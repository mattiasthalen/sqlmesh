from __future__ import annotations

import datetime as dt
import logging
import time
import typing as t
from typing import Any, List, Optional, Sequence, Tuple

import requests
from sqlglot import exp
from sqlmesh.core.engine_adapter.mixins import (
    GetCurrentCatalogFromFunctionMixin,
    HiveMetastoreTablePropertiesMixin,
    RowDiffMixin,
)
from sqlmesh.core.engine_adapter.shared import (
    CatalogSupport,
    CommentCreationTable,
    CommentCreationView,
    InsertOverwriteStrategy,
    set_catalog,
)
from sqlmesh.core.schema_diff import SchemaDiffer

if t.TYPE_CHECKING:
    from azure.core.credentials import AccessToken

logger = logging.getLogger(__name__)

DEFAULT_POLL_WAIT = 10
DEFAULT_POLL_STATEMENT_WAIT = 5
AZURE_CREDENTIAL_SCOPE = "https://analysis.windows.net/powerbi/api/.default"


def is_token_refresh_necessary(unix_timestamp: int) -> bool:
    """Check if Azure token needs refresh based on expiration time."""
    dt_object = dt.datetime.fromtimestamp(unix_timestamp)
    local_time = time.localtime(time.time())
    difference = dt_object - dt.datetime.fromtimestamp(time.mktime(local_time))
    return int(difference.total_seconds() / 60) < 5


def get_cli_access_token() -> AccessToken:
    """Get Azure access token using CLI credentials."""
    from azure.identity import AzureCliCredential

    return AzureCliCredential().get_token(AZURE_CREDENTIAL_SCOPE)


def get_sp_access_token(tenant_id: str, client_id: str, client_secret: str) -> AccessToken:
    """Get Azure access token using service principal credentials."""
    from azure.identity import ClientSecretCredential

    return ClientSecretCredential(tenant_id, client_id, client_secret).get_token(
        AZURE_CREDENTIAL_SCOPE
    )


class FabricSparkCredentials:
    """Credentials for Microsoft Fabric Spark connection."""

    def __init__(
        self,
        workspace_id: str,
        lakehouse_id: str,
        database: str,
        endpoint: str = "https://api.fabric.microsoft.com/v1",
        authentication: str = "az_cli",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        access_token: Optional[str] = None,
        spark_config: Optional[dict] = None,
        connect_retries: int = 1,
        connect_timeout: int = 10,
        **kwargs: Any,
    ) -> None:
        self.workspace_id = workspace_id
        self.lakehouse_id = lakehouse_id
        self.database = database
        self.endpoint = endpoint
        self.authentication = authentication
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.access_token = access_token
        self.spark_config = spark_config or {"name": "sqlmesh-session"}
        self.connect_retries = connect_retries
        self.connect_timeout = connect_timeout

    @property
    def lakehouse_endpoint(self) -> str:
        """Get the Livy API endpoint for the lakehouse."""
        return f"{self.endpoint}/workspaces/{self.workspace_id}/lakehouses/{self.lakehouse_id}/livyapi/versions/2023-12-01"


class LivySession:
    """Manages a Livy session for Fabric Spark."""

    def __init__(self, credentials: FabricSparkCredentials) -> None:
        self.credentials = credentials
        self.session_id: Optional[int] = None
        self.access_token: Optional[AccessToken] = None
        self._ensure_token()

    def _ensure_token(self) -> None:
        """Ensure we have a valid access token."""
        if self.access_token and not is_token_refresh_necessary(self.access_token.expires_on):
            return

        if self.credentials.access_token:
            # Use provided token (assume it's valid)
            from azure.core.credentials import AccessToken

            self.access_token = AccessToken(
                token=self.credentials.access_token,
                expires_on=int(time.time()) + 3600,  # Assume 1 hour validity
            )
        elif self.credentials.authentication == "az_cli":
            self.access_token = get_cli_access_token()
        elif self.credentials.authentication == "service_principal":
            if not all(
                [
                    self.credentials.tenant_id,
                    self.credentials.client_id,
                    self.credentials.client_secret,
                ]
            ):
                raise ValueError(
                    "Service principal authentication requires tenant_id, client_id, and client_secret"
                )
            self.access_token = get_sp_access_token(
                self.credentials.tenant_id,  # type: ignore
                self.credentials.client_id,  # type: ignore
                self.credentials.client_secret,  # type: ignore
            )
        else:
            raise ValueError(
                f"Unsupported authentication method: {self.credentials.authentication}"
            )

    def _headers(self) -> dict:
        """Get HTTP headers for API requests."""
        self._ensure_token()
        if not self.access_token:
            raise ValueError("No access token available")
        return {
            "Authorization": f"Bearer {self.access_token.token}",
            "Content-Type": "application/json",
        }

    def create_session(self) -> int:
        """Create a new Livy session."""
        if self.session_id:
            return self.session_id

        url = f"{self.credentials.lakehouse_endpoint}/sessions"
        payload = self.credentials.spark_config.copy()

        response = requests.post(url, json=payload, headers=self._headers())
        response.raise_for_status()

        session_data = response.json()
        self.session_id = session_data["id"]

        # Wait for session to be ready
        self._wait_for_session_ready()
        if not self.session_id:
            raise ValueError("Failed to create session")
        return self.session_id

    def _wait_for_session_ready(self) -> None:
        """Wait for the Livy session to be ready."""
        if not self.session_id:
            raise ValueError("No session ID available")

        url = f"{self.credentials.lakehouse_endpoint}/sessions/{self.session_id}"

        while True:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()

            session_data = response.json()
            state = session_data.get("state")

            if state == "idle":
                logger.info(f"Livy session {self.session_id} is ready")
                break
            elif state in ["error", "dead", "killed"]:
                raise RuntimeError(f"Livy session failed with state: {state}")

            logger.debug(f"Waiting for session {self.session_id}, current state: {state}")
            time.sleep(DEFAULT_POLL_WAIT)

    def execute_statement(self, code: str) -> dict:
        """Execute a SQL statement in the Livy session."""
        if not self.session_id:
            self.create_session()

        url = f"{self.credentials.lakehouse_endpoint}/sessions/{self.session_id}/statements"
        # Fabric Livy requires statements to be wrapped in spark.sql() with kind="spark"
        # Use .show() to get formatted results that we can parse
        wrapped_code = f'spark.sql("{code.replace('"', '\\"')}").show()'
        payload = {"code": wrapped_code, "kind": "spark"}

        response = requests.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            logger.error(f"HTTP {response.status_code} error for {url}: {response.text}")
        response.raise_for_status()

        statement_data = response.json()
        statement_id = statement_data["id"]

        return self._wait_for_statement_completion(statement_id)

    def _wait_for_statement_completion(self, statement_id: int) -> dict:
        """Wait for a statement to complete and return results."""
        url = f"{self.credentials.lakehouse_endpoint}/sessions/{self.session_id}/statements/{statement_id}"

        while True:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()

            statement_data = response.json()
            state = statement_data.get("state")

            if state == "available":
                return statement_data
            if state in ["error", "cancelled"]:
                error_msg = statement_data.get("output", {}).get("evalue", "Unknown error")
                raise RuntimeError(f"Statement failed: {error_msg}")

            time.sleep(DEFAULT_POLL_STATEMENT_WAIT)

    def close(self) -> None:
        """Close the Livy session."""
        if self.session_id:
            url = f"{self.credentials.lakehouse_endpoint}/sessions/{self.session_id}"
            try:
                response = requests.delete(url, headers=self._headers())
                response.raise_for_status()
                logger.info(f"Closed Livy session {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to close session {self.session_id}: {e}")
            finally:
                self.session_id = None


class FabricSparkCursor:
    """Database cursor implementation for Fabric Spark."""

    def __init__(self, livy_session: LivySession) -> None:
        self.livy_session = livy_session
        self._last_result: Optional[dict] = None
        self._last_output: Optional[List[Tuple]] = None
        self._last_output_cursor: int = 0
        self.description: Optional[
            Sequence[
                Tuple[str, Any, Optional[int], Optional[int], Optional[int], Optional[int], bool]
            ]
        ] = None

    def execute(self, query: str, parameters: Optional[Any] = None) -> None:
        """Execute a SQL query."""
        if parameters:
            raise NotImplementedError("Parameterized queries are not supported")

        # Ensure the query uses the correct database/schema
        if not query.lower().strip().startswith(("use ", "create database", "create schema")):
            use_statement = f"USE {self.livy_session.credentials.database}"
            self.livy_session.execute_statement(use_statement)

        result = self.livy_session.execute_statement(query)
        self._process_result(result)

    def _process_result(self, result: dict) -> None:
        """Process the result from Livy session."""
        self._last_result = result
        self._last_output = None
        self._last_output_cursor = 0

        output = result.get("output", {})
        data = output.get("data", {})

        if "application/json" in data:
            # Parse JSON result
            json_data = data["application/json"]
            if isinstance(json_data, dict) and "data" in json_data:
                rows = json_data["data"]
                self._last_output = [tuple(row) for row in rows]

                # Set description if available
                if "schema" in json_data:
                    schema = json_data["schema"]
                    self.description = [
                        (field["name"], field["type"], None, None, None, None, True)
                        for field in schema.get("fields", [])
                    ]
        elif "text/plain" in data:
            # Handle text output from .show() or other text results
            text_output = data["text/plain"]
            # Try to parse Spark DataFrame .show() output
            if self._is_spark_show_output(text_output):
                self._last_output = self._parse_spark_show_output(text_output)
            else:
                self._last_output = [(text_output,)]

    def _is_spark_show_output(self, text: str) -> bool:
        """Check if text output is from Spark DataFrame .show()"""
        return "+---" in text or "only showing top" in text.lower()

    def _parse_spark_show_output(self, text: str) -> List[Tuple]:
        """Parse Spark DataFrame .show() output into rows"""
        lines = text.strip().split("\n")
        rows = []

        # Find data rows (skip header and separator lines)
        in_data = False
        for line in lines:
            line = line.strip()
            if line.startswith("+") and "---" in line:
                in_data = not in_data  # Toggle when we see separator
                continue
            if in_data and line.startswith("|") and line.endswith("|"):
                # Parse row data
                values = [val.strip() for val in line[1:-1].split("|")]
                # Convert values to appropriate types
                parsed_values: List[Any] = []
                for val in values:
                    if val.isdigit():
                        parsed_values.append(int(val))
                    elif val.replace(".", "").isdigit():
                        parsed_values.append(float(val))
                    else:
                        parsed_values.append(val)
                rows.append(tuple(parsed_values))

        return rows

    def fetchone(self) -> Optional[Tuple]:
        """Fetch one row from the result."""
        result = self._fetch(size=1)
        return result[0] if result else None

    def fetchmany(self, size: int = 1) -> List[Tuple]:
        """Fetch multiple rows from the result."""
        return self._fetch(size=size)

    def fetchall(self) -> List[Tuple]:
        """Fetch all remaining rows from the result."""
        return self._fetch()

    def _fetch(self, size: Optional[int] = None) -> List[Tuple]:
        """Internal fetch implementation."""
        if size and size < 0:
            raise ValueError("The size argument can't be negative")

        if self._last_output is None:
            return []

        if self._last_output_cursor >= len(self._last_output):
            return []

        if size is None:
            size = len(self._last_output) - self._last_output_cursor

        output = self._last_output[self._last_output_cursor : self._last_output_cursor + size]
        self._last_output_cursor += size

        return output

    def close(self) -> None:
        """Close the cursor."""
        pass


class FabricSparkConnection:
    """Database connection implementation for Fabric Spark."""

    def __init__(self, credentials: FabricSparkCredentials) -> None:
        self.credentials = credentials
        self.livy_session = LivySession(credentials)

    def cursor(self) -> FabricSparkCursor:
        """Create a new cursor."""
        return FabricSparkCursor(self.livy_session)

    def commit(self) -> None:
        """Commit transaction (no-op for Spark)."""
        pass

    def rollback(self) -> None:
        """Rollback transaction (no-op for Spark)."""
        pass

    def close(self) -> None:
        """Close the connection."""
        self.livy_session.close()


def fabricspark_connect(
    workspace_id: str,
    lakehouse_id: str,
    database: str,
    endpoint: str = "https://msitapi.fabric.microsoft.com/v1",
    authentication: str = "az_cli",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None,
    access_token: Optional[str] = None,
    spark_config: Optional[dict] = None,
    connect_retries: int = 1,
    connect_timeout: int = 10,
    **kwargs: Any,
) -> FabricSparkConnection:
    """Create a connection to Microsoft Fabric Spark."""
    credentials = FabricSparkCredentials(
        workspace_id=workspace_id,
        lakehouse_id=lakehouse_id,
        database=database,
        endpoint=endpoint,
        authentication=authentication,
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        access_token=access_token,
        spark_config=spark_config,
        connect_retries=connect_retries,
        connect_timeout=connect_timeout,
        **kwargs,
    )
    return FabricSparkConnection(credentials)


@set_catalog()
class FabricSparkEngineAdapter(
    GetCurrentCatalogFromFunctionMixin, HiveMetastoreTablePropertiesMixin, RowDiffMixin
):
    """
    Engine adapter for Microsoft Fabric Spark.

    This adapter provides connectivity to Microsoft Fabric's Spark compute
    through Livy sessions with Azure authentication.
    """

    DIALECT = "spark"
    SUPPORTS_TRANSACTIONS = False
    INSERT_OVERWRITE_STRATEGY = InsertOverwriteStrategy.INSERT_OVERWRITE
    COMMENT_CREATION_TABLE = CommentCreationTable.IN_SCHEMA_DEF_NO_CTAS
    COMMENT_CREATION_VIEW = CommentCreationView.IN_SCHEMA_DEF_NO_COMMANDS
    SUPPORTS_REPLACE_TABLE = False
    QUOTE_IDENTIFIERS_IN_VIEWS = False

    SCHEMA_DIFFER = SchemaDiffer(
        parameterized_type_defaults={
            # default decimal precision varies across backends
            exp.DataType.build("DECIMAL", dialect=DIALECT).this: [(), (0,)],
        },
    )

    @property
    def connection(self) -> FabricSparkConnection:
        return self._connection_pool.get()

    @property
    def use_serverless(self) -> bool:
        """Fabric Spark uses managed compute similar to serverless."""
        return True

    @property
    def catalog_support(self) -> CatalogSupport:
        return CatalogSupport.FULL_SUPPORT

    def _get_temp_table(
        self, table: t.Union[exp.Table, str], table_only: bool = False, quoted: bool = True
    ) -> exp.Table:
        """
        Returns the name of the temp table that should be used for the given table name.
        Fabric Spark has similar temporary table limitations as regular Spark.
        """
        table = super()._get_temp_table(table, table_only=table_only)
        table_name_id = table.args["this"]
        # Fabric Spark may also have issues with temp tables starting with __temp
        table_name_id.set("this", table_name_id.this.replace("__temp_", "fabric_temp_"))
        return table

    def get_current_catalog(self) -> t.Optional[str]:
        """Get the current catalog from the Fabric connection."""
        # For Fabric, this would typically be the lakehouse name
        try:
            return self.fetchone(exp.select(exp.func("current_catalog")))[0]  # type: ignore
        except Exception:
            # Fallback if current_catalog() is not available
            return "spark_catalog"

    def set_current_catalog(self, catalog_name: str) -> None:
        """Set the current catalog for the Fabric connection."""
        try:
            self.execute(f"USE CATALOG {catalog_name}")
        except Exception as e:
            logger.warning(f"Failed to set catalog {catalog_name}: {e}")

    def get_current_database(self) -> str:
        """Get the current database (schema) from Fabric Spark."""
        # Fabric uses lakehouse concepts, but Spark SQL still has database/schema
        return self.fetchone(exp.select(exp.func("current_database")))[0]  # type: ignore
