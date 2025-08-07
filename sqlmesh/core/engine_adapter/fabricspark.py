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
    DataObject,
    DataObjectType,
    InsertOverwriteStrategy,
    SourceQuery,
    set_catalog,
    to_schema,
)
from sqlmesh.core.schema_diff import SchemaDiffer

if t.TYPE_CHECKING:
    import pandas as pd
    from azure.core.credentials import AccessToken
    from sqlmesh.core._typing import SchemaName, TableName
    from sqlmesh.core.engine_adapter._typing import QueryOrDF

logger = logging.getLogger(__name__)

DEFAULT_POLL_WAIT = 2  # Reduced from 10 to 2 seconds for faster session startup
DEFAULT_POLL_STATEMENT_WAIT = 1  # Reduced from 5 to 1 second for faster queries
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
        # Use direct SQL execution like dbt-fabricspark does
        payload = {"code": code, "kind": "sql"}

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

        # Check if the query executed successfully
        if output.get("status") != "ok":
            error_msg = output.get("evalue", "Unknown error")
            raise RuntimeError(f"Query failed: {error_msg}")

        data = output.get("data", {})

        if "application/json" in data:
            # Parse JSON result (direct from Livy like dbt does)
            json_data = data["application/json"]
            if isinstance(json_data, dict) and "data" in json_data:
                rows = json_data["data"]
                # Convert string values to appropriate Python types
                converted_rows = []
                for row in rows:
                    converted_row = []
                    for val in row:
                        converted_row.append(self._convert_value(val))
                    converted_rows.append(tuple(converted_row))
                self._last_output = converted_rows

                # Set description if available
                if "schema" in json_data:
                    schema = json_data["schema"]
                    self.description = [
                        (
                            field["name"],
                            field["type"],
                            None,
                            None,
                            None,
                            None,
                            field.get("nullable", True),
                        )
                        for field in schema.get("fields", [])
                    ]
            else:
                # Handle case where there's no data (e.g., DDL statements)
                self._last_output = []
                self.description = None
        elif "text/plain" in data:
            # Fallback for text output (shouldn't happen with kind="sql")
            text_output = data["text/plain"]
            self._last_output = [(text_output,)]
        else:
            # No data returned (e.g., successful DDL)
            self._last_output = []
            self.description = None

    def _convert_value(self, val: Any) -> Any:
        """Convert string values from Livy JSON to appropriate Python types."""
        if val is None:
            return None
        if isinstance(val, str):
            # Handle null values
            if val.lower() in ("null", "none", ""):
                return None
            # Handle boolean values
            if val.lower() in ("true", "false"):
                return val.lower() == "true"
            # Handle integers (including negative)
            if val.lstrip("-").isdigit():
                return int(val)
            # Handle floats (including negative)
            if val.replace(".", "", 1).replace("-", "", 1).isdigit() and "." in val:
                return float(val)
            # Keep as string for everything else
            return val
        # Already the right type
        return val

    def _is_spark_show_output(self, text: str) -> bool:
        """Check if text output is from Spark DataFrame .show()"""
        return "+---" in text or "only showing top" in text.lower()

    def _parse_spark_show_output(self, text: str) -> List[Tuple]:
        """Parse Spark DataFrame .show() output into rows"""
        lines = text.strip().split("\n")
        rows = []

        # Find data rows (skip header and separator lines)
        # Spark .show() format: +---+, |header|, +---+, |data|, +---+
        header_seen = False
        for line in lines:
            line = line.strip()
            if line.startswith("+") and "---" in line:
                continue  # Skip separator lines
            if line.startswith("|") and line.endswith("|"):
                if not header_seen:
                    # Skip the header row
                    header_seen = True
                    continue
                # Parse data row
                values = [val.strip() for val in line[1:-1].split("|")]
                # Convert values to appropriate types
                parsed_values: List[Any] = []
                for val in values:
                    # Handle null values
                    if val.lower() in ("null", "none", ""):
                        parsed_values.append(None)
                    # Handle boolean values
                    elif val.lower() in ("true", "false"):
                        parsed_values.append(val.lower() == "true")
                    # Handle integers (including negative)
                    elif val.lstrip("-").isdigit():
                        parsed_values.append(int(val))
                    # Handle floats (including negative)
                    elif val.replace(".", "", 1).replace("-", "", 1).isdigit() and "." in val:
                        parsed_values.append(float(val))
                    else:
                        # Keep as string for everything else
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
    SUPPORTS_MATERIALIZED_VIEWS = True
    INSERT_OVERWRITE_STRATEGY = InsertOverwriteStrategy.INSERT_OVERWRITE
    COMMENT_CREATION_TABLE = CommentCreationTable.UNSUPPORTED
    COMMENT_CREATION_VIEW = CommentCreationView.UNSUPPORTED
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
    def comments_enabled(self) -> bool:
        """Fabric Spark does not reliably support table/column comments."""
        return False

    @property
    def catalog_support(self) -> CatalogSupport:
        return CatalogSupport.SINGLE_CATALOG_ONLY

    def get_current_catalog(self) -> t.Optional[str]:
        """Get the current catalog from the Fabric connection."""
        # In Fabric Spark, we want to use the lakehouse name as the catalog name
        # instead of the generic 'spark_catalog' returned by current_catalog()
        # The lakehouse name is stored in the database field of credentials
        return self.connection.credentials.database

    def get_current_database(self) -> str:
        """Get the current database (schema) from Fabric Spark."""
        # Fabric uses lakehouse concepts, but Spark SQL still has database/schema
        result = self.fetchone("SELECT current_database()")
        return result[0] if result else "default"  # type: ignore

    def fetchdf(
        self, query: t.Union[exp.Expression, str], quote_identifiers: bool = False
    ) -> "pd.DataFrame":
        """Fetch a Pandas DataFrame from a query."""
        import pandas as pd

        results = self.fetchall(query, quote_identifiers=quote_identifiers)
        if not results:
            return pd.DataFrame()

        # Get column names and types from cursor description if available
        if hasattr(self.cursor, "description") and self.cursor.description:
            columns = [desc[0] for desc in self.cursor.description]
            column_types = (
                [desc[1] for desc in self.cursor.description]
                if len(self.cursor.description[0]) > 1
                else None
            )
        else:
            # Fallback: use generic column names
            columns = [f"col_{i}" for i in range(len(results[0]) if results else 0)]
            column_types = None

        df = pd.DataFrame(results, columns=columns)

        # Convert values to appropriate pandas types based on column types
        if column_types and self.cursor.description:
            for i, (col_name, col_type) in enumerate(zip(columns, column_types)):
                col_type_str = str(col_type).upper() if col_type else ""

                if col_type and col_type_str in ("TIMESTAMP", "DATETIME", "DATE"):
                    # Convert None values to pd.NaT for timestamp/datetime columns
                    df[col_name] = df[col_name].apply(lambda x: pd.NaT if x is None else x)
                    # Handle specific data type conversions
                    if col_type_str == "DATE":
                        try:
                            # Convert date strings to datetime.date objects
                            def convert_date(x: t.Any) -> t.Any:
                                if pd.isna(x) or x is pd.NaT:
                                    return pd.NaT
                                if isinstance(x, str):
                                    # Parse date string to datetime.date
                                    dt = pd.to_datetime(x, errors="coerce")
                                    return dt.date() if not pd.isna(dt) else pd.NaT
                                return x

                            df[col_name] = df[col_name].apply(convert_date)
                        except:
                            pass  # Keep original values if conversion fails
                    elif col_type_str in ("TIMESTAMP", "DATETIME"):
                        try:
                            converted = pd.to_datetime(df[col_name], errors="coerce")
                            # Convert timezone-aware datetime to timezone-naive for compatibility
                            if hasattr(converted.dtype, "tz") and converted.dtype.tz is not None:
                                converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
                            df[col_name] = converted
                        except:
                            pass  # Keep original values if conversion fails

                elif col_type and col_type_str in ("BOOLEAN", "BOOL"):
                    # For boolean columns, keep them as numeric (0, 1) for table diff operations
                    # Only convert to strings if explicitly needed for specific use cases
                    try:

                        def convert_boolean(x: t.Any) -> t.Any:
                            if x is None:
                                return None
                            # Keep as numeric values for arithmetic operations
                            if isinstance(x, (int, float)):
                                return 1 if x else 0
                            if isinstance(x, bool):
                                return 1 if x else 0
                            return x

                        df[col_name] = df[col_name].apply(convert_boolean)
                    except:
                        pass  # Keep original values if conversion fails

        # Fallback: Apply data type transformations based on data inspection when type info unavailable
        for col_name in df.columns:
            # Check if column contains boolean-like numeric values (0.0, 1.0) that should be strings
            if not df[col_name].empty:
                sample_values = df[col_name].dropna()
                if len(sample_values) > 0:
                    # Check if all non-null values are exactly 0 or 1 (as int or float)
                    all_boolean_like = all(
                        isinstance(v, (int, float)) and v in (0, 1, 0.0, 1.0) for v in sample_values
                    )
                    if all_boolean_like:
                        # This looks like boolean data - keep as numeric for arithmetic operations
                        def convert_to_numeric_bool(x: t.Any) -> t.Any:
                            if x is None or pd.isna(x):
                                return None
                            return 1 if (x == 1 or x == 1.0) else 0

                        df[col_name] = df[col_name].apply(convert_to_numeric_bool)

                elif col_type and col_type_str in (
                    "INT",
                    "INTEGER",
                    "BIGINT",
                    "SMALLINT",
                    "TINYINT",
                ):
                    # Convert string integers to actual integers
                    try:
                        df[col_name] = pd.to_numeric(
                            df[col_name], errors="coerce", downcast="integer"
                        )
                    except:
                        pass  # Keep original values if conversion fails

                elif col_type and col_type_str in ("FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"):
                    # Convert string floats to actual floats
                    try:
                        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                    except:
                        pass  # Keep original values if conversion fails

        return df

    def _get_data_objects(
        self, schema_name: "SchemaName", object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """Get data objects (tables, views) from the specified schema."""
        schema_name_str = to_schema(schema_name).sql(dialect=self.dialect)

        # Fabric Spark: Use SHOW TABLES approach since SHOW TABLE EXTENDED may not work reliably
        data_objects = []
        catalog = self.get_current_catalog()

        try:
            # Use SHOW TABLES IN schema_name format for better reliability
            if schema_name_str and schema_name_str != "default":
                show_sql = f"SHOW TABLES IN {schema_name_str}"
            else:
                show_sql = "SHOW TABLES"

            logger.debug(f"Executing metadata query: {show_sql}")
            basic_results = self.fetchall(show_sql)
            logger.debug(f"SHOW TABLES returned {len(basic_results)} rows")

            for row in basic_results:  # type: ignore
                logger.debug(f"Processing SHOW TABLES row: {row}")
                # Basic SHOW TABLES format: (database, tableName, isTemporary)
                # In Fabric Spark, sometimes it's just (tableName,) or (database, tableName)
                if len(row) >= 1:
                    if len(row) == 1:
                        # Just table name
                        db_name = schema_name_str
                        table_name = row[0]
                        is_temp = False
                    elif len(row) == 2:
                        # Database and table name
                        raw_db_name = row[0] if row[0] else schema_name_str
                        # Extract just the schema name from fully qualified database name
                        # FabricSpark may return "`catalog`.database.schema" but we only want "schema"
                        if raw_db_name and "." in raw_db_name:
                            # Parse the fully qualified name and extract just the last component (schema)
                            try:
                                parsed_table = exp.to_table(raw_db_name, dialect=self.dialect)
                                # If this parses as a 3-part identifier, the schema is the name part
                                # If it's a 2-part identifier, the schema is also the name part
                                db_name = parsed_table.name if parsed_table.name else raw_db_name
                            except Exception:
                                # Fallback: split on dots and take the last part
                                parts = raw_db_name.replace("`", "").split(".")
                                db_name = parts[-1] if parts else raw_db_name
                        else:
                            db_name = raw_db_name
                        table_name = row[1]
                        is_temp = False
                    else:
                        # Full format with temp flag
                        raw_db_name = row[0] if row[0] else schema_name_str
                        # Extract just the schema name from fully qualified database name
                        # FabricSpark may return "`catalog`.database.schema" but we only want "schema"
                        if raw_db_name and "." in raw_db_name:
                            # Parse the fully qualified name and extract just the last component (schema)
                            try:
                                parsed_table = exp.to_table(raw_db_name, dialect=self.dialect)
                                # If this parses as a 3-part identifier, the schema is the name part
                                # If it's a 2-part identifier, the schema is also the name part
                                db_name = parsed_table.name if parsed_table.name else raw_db_name
                            except Exception:
                                # Fallback: split on dots and take the last part
                                parts = raw_db_name.replace("`", "").split(".")
                                db_name = parts[-1] if parts else raw_db_name
                        else:
                            db_name = raw_db_name
                        table_name = row[1]
                        is_temp = len(row) > 2 and str(row[2]).lower() == "true"

                    # Skip temporary tables
                    if is_temp:
                        continue

                    if table_name:
                        # Filter by object_names if specified
                        if object_names is not None and table_name not in object_names:
                            continue

                        # Check if this is a materialized lake view created by SQLMesh
                        object_type = DataObjectType.TABLE
                        try:
                            # Try to get table properties to determine if it's a materialized lake view
                            desc_sql = (
                                f"DESCRIBE TABLE EXTENDED {db_name or schema_name_str}.{table_name}"
                            )
                            desc_results = self.fetchall(desc_sql)
                            # Look through all the description rows for table properties
                            full_description = "\n".join([str(row) for row in desc_results])

                            # Check if this table has the materialized lake view property
                            if "lakehouse_materialized_view" in full_description.lower():
                                # Check the SQLMesh view type property to distinguish between views and materialized views
                                if "sqlmesh.view.type" in full_description.lower():
                                    # More precise check for the view type value - handle different spacing and quote patterns
                                    normalized_desc = (
                                        full_description.lower()
                                        .replace(" ", "")
                                        .replace("'", "")
                                        .replace('"', "")
                                    )
                                    if "sqlmesh.view.type=materialized_view" in normalized_desc:
                                        object_type = DataObjectType.MATERIALIZED_VIEW
                                    elif "sqlmesh.view.type=view" in normalized_desc:
                                        object_type = DataObjectType.VIEW
                                    else:
                                        # If we have the property but can't parse it, default to materialized view
                                        object_type = DataObjectType.MATERIALIZED_VIEW
                                else:
                                    # Default to materialized view if no SQLMesh property found
                                    object_type = DataObjectType.MATERIALIZED_VIEW
                        except Exception:
                            pass  # Ignore errors getting table description

                        logger.debug(
                            f"Adding data object: catalog={catalog}, schema={db_name}, name={table_name}, type={object_type}"
                        )
                        data_objects.append(
                            DataObject(
                                catalog=catalog,
                                schema=db_name or schema_name_str,
                                name=table_name,
                                type=object_type,
                            )
                        )
        except Exception as e:
            logger.warning(f"Failed to get data objects for schema {schema_name_str}: {e}")

        logger.debug(f"Returning {len(data_objects)} data objects")
        return data_objects

    def create_view(
        self,
        view_name: "TableName",
        query_or_df: "QueryOrDF",
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        replace: bool = True,
        materialized: bool = False,
        materialized_properties: t.Optional[t.Dict[str, t.Any]] = None,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        view_properties: t.Optional[t.Dict[str, exp.Expression]] = None,
        **create_kwargs: t.Any,
    ) -> None:
        """Create a materialized lake view instead of a regular view for Fabric Spark."""

        # Fabric Spark doesn't support regular views, only materialized lake views
        # Convert all view creation to materialized lake view creation
        logger.info(f"Creating materialized lake view {view_name} instead of regular view")

        # Import here to avoid circular imports
        import pandas as pd

        if materialized_properties and not materialized:
            from sqlmesh.utils.errors import SQLMeshError

            raise SQLMeshError("Materialized properties are only supported for materialized views")

        query_or_df = self._native_df_to_pandas_df(query_or_df)

        if isinstance(query_or_df, pd.DataFrame):
            values: t.List[t.Tuple[t.Any, ...]] = list(
                query_or_df.itertuples(index=False, name=None)
            )
            columns_to_types = columns_to_types or self._columns_to_types(query_or_df)
            if not columns_to_types:
                from sqlmesh.utils.errors import SQLMeshError

                raise SQLMeshError("columns_to_types must be provided for dataframes")
            query_or_df = self._values_to_sql(
                values,
                columns_to_types,
                batch_start=0,
                batch_end=len(values),
            )

        source_queries, columns_to_types = self._get_source_queries_and_columns_to_types(
            query_or_df, columns_to_types, batch_size=0, target_table=view_name
        )
        if len(source_queries) != 1:
            from sqlmesh.utils.errors import SQLMeshError

            raise SQLMeshError("Only one source query is supported for creating views")

        schema: t.Union[exp.Table, exp.Schema] = exp.to_table(view_name)
        if columns_to_types:
            schema = self._build_schema_exp(
                exp.to_table(view_name), columns_to_types, column_descriptions, is_view=True
            )

        properties = create_kwargs.pop("properties", None)
        if not properties:
            properties = exp.Properties(expressions=[])

        # Force materialized lake view for Fabric Spark
        properties.append("expressions", exp.MaterializedProperty())

        # Add lake view type property - this is specific to Fabric Spark
        properties.append(
            "expressions", exp.Property(this="TYPE", value=exp.Literal.string("LAKEHOUSE"))
        )

        if view_properties:
            table_type = self._pop_creatable_type_from_properties(view_properties)
            if table_type:
                properties.append("expressions", table_type)

        if not self.SUPPORTS_VIEW_SCHEMA and isinstance(schema, exp.Schema):
            schema = schema.this

        create_view_properties = self._build_view_properties_exp(
            view_properties,
            (
                table_description
                if self.COMMENT_CREATION_VIEW.supports_schema_def and self.comments_enabled
                else None
            ),
            physical_cluster=create_kwargs.pop("physical_cluster", None),
        )
        if create_view_properties:
            for view_property in create_view_properties.expressions:
                properties.append("expressions", view_property)

        if properties.expressions:
            create_kwargs["properties"] = properties

        with source_queries[0] as query:
            # For Fabric Spark, create a materialized lake view
            view_name_sql = exp.to_table(view_name).sql(dialect=self.dialect)

            if replace:
                # Drop existing materialized view if it exists (they are stored as tables)
                try:
                    self.execute(f"DROP TABLE IF EXISTS {view_name_sql}")
                except Exception:
                    pass  # Ignore if doesn't exist

            # Create materialized lake view using Fabric Spark syntax
            # Reference: https://learn.microsoft.com/en-us/fabric/data-engineering/materialized-lake-views/create-materialized-lake-view

            # Add a property to distinguish between regular views and materialized views
            view_type = "materialized_view" if materialized else "view"
            create_sql = (
                f"CREATE OR REPLACE TABLE {view_name_sql} "
                f"TBLPROPERTIES ("
                f"'lakehouse.table.type' = 'lakehouse_materialized_view', "
                f"'sqlmesh.view.type' = '{view_type}'"
                f") AS {query.sql(dialect=self.dialect)}"
            )
            self.execute(create_sql)

    def drop_view(
        self,
        view_name: "TableName",
        ignore_if_not_exists: bool = True,
        materialized: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Drop a view or materialized view in Fabric Spark."""
        # According to Microsoft docs, materialized lake views should be dropped using DROP VIEW
        # Reference: https://learn.microsoft.com/en-us/fabric/data-engineering/materialized-lake-views/create-materialized-lake-view#drop-a-materialized-lake-view
        view_name_sql = exp.to_table(view_name).sql(dialect=self.dialect)

        exists_clause = "IF EXISTS" if ignore_if_not_exists else ""
        # Fabric Spark doesn't support CASCADE in DROP VIEW

        # In Fabric Spark, materialized lake views are stored as tables with special properties
        # So we need to drop them as tables, not as views
        drop_sql = f"DROP TABLE {exists_clause} {view_name_sql}".strip()
        self.execute(drop_sql)

    def _insert_overwrite_by_time_partition(
        self,
        table_name: TableName,
        source_queries: t.List[SourceQuery],
        columns_to_types: t.Dict[str, exp.DataType],
        where: exp.Condition,
        **kwargs: t.Any,
    ) -> None:
        """
        Override time partition overwrite to use DELETE_INSERT strategy.

        FabricSpark's INSERT_OVERWRITE strategy replaces the entire table,
        but for time-partitioned operations we need to replace only the
        specified time range. This method uses DELETE + INSERT approach
        to achieve the correct behavior.
        """
        # Force DELETE_INSERT strategy for time-partitioned overwrites
        return self._insert_overwrite_by_condition(
            table_name,
            source_queries,
            columns_to_types,
            where,
            insert_overwrite_strategy_override=InsertOverwriteStrategy.DELETE_INSERT,
            **kwargs,
        )

    def _normalize_boolean_value(self, expr: exp.Expression) -> exp.Expression:
        """
        Normalize boolean values for FabricSpark.

        FabricSpark returns boolean values as floats (1.0/0.0) when cast to INT,
        which when converted to VARCHAR becomes "1.0"/"0.0" instead of "1"/"0".

        Use a CASE expression to directly convert boolean to string "1" or "0".
        """
        return exp.Case(
            ifs=[
                exp.If(this=expr, true=exp.Literal.string("1")),
            ],
            default=exp.Literal.string("0"),
        )
