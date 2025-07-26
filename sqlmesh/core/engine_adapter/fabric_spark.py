from __future__ import annotations

import contextlib
import logging
import time
import typing as t

import requests
from sqlglot import exp

from sqlmesh.core.dialect import to_schema
from sqlmesh.core.engine_adapter.shared import (
    CatalogSupport,
    DataObject,
    DataObjectType,
    InsertOverwriteStrategy,
)
from sqlmesh.core.engine_adapter.base import EngineAdapter
from sqlmesh.core.schema_diff import SchemaDiffer
from sqlmesh.utils.errors import SQLMeshError

if t.TYPE_CHECKING:
    import pandas as pd

    from sqlmesh.core._typing import SchemaName, SessionProperties

logger = logging.getLogger(__name__)


class FabricSparkEngineAdapter(EngineAdapter):
    """
    Microsoft Fabric Spark engine adapter using Livy REST API.

    This adapter provides integration with Microsoft Fabric Spark through:
    - Livy REST API for Spark SQL execution
    - Fabric REST API for lakehouse management (creation/deletion)
    - Full catalog support for Fabric lakehouses
    """

    DIALECT = "spark"  # Use Spark dialect since fabric-spark isn't available in SQLGlot
    INSERT_OVERWRITE_STRATEGY = InsertOverwriteStrategy.INSERT_OVERWRITE
    SUPPORTS_TRANSACTIONS = False
    SUPPORTS_CLONING = False
    SUPPORTS_MATERIALIZED_VIEWS = True
    SUPPORTS_MATERIALIZED_VIEW_SCHEMA = True

    SCHEMA_DIFFER = SchemaDiffer(
        support_positional_add=True,
        support_nested_operations=True,
        support_nested_drop=True,
        array_element_selector="element",
        parameterized_type_defaults={
            exp.DataType.build("DECIMAL", dialect="spark").this: [(10, 0), (0,)],
        },
    )

    def __init__(self, connection_factory: t.Callable[[], t.Any], **kwargs: t.Any) -> None:
        super().__init__(connection_factory, **kwargs)
        self._livy_session_id: t.Optional[t.Union[int, str]] = None
        self._session_headers: t.Dict[str, str] = {}
        self._fabric_headers: t.Dict[str, str] = {}
        self._lakehouse_id_cache: t.Optional[str] = None
        self._setup_authentication()

    def _setup_authentication(self) -> None:
        """Setup authentication headers for Livy and Fabric APIs."""
        client_id = self._extra_config.get("client_id")
        client_secret = self._extra_config.get("client_secret")
        tenant_id = self._extra_config.get("tenant_id")

        if not all([client_id, client_secret, tenant_id]):
            raise SQLMeshError(
                "client_id, client_secret, and tenant_id are required for Fabric Spark connection"
            )

        # Get access token using client credentials flow
        access_token = self._get_access_token(
            t.cast(str, client_id), t.cast(str, client_secret), t.cast(str, tenant_id)
        )

        self._session_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        self._fabric_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def _get_access_token(self, client_id: str, client_secret: str, tenant_id: str) -> str:
        """Get access token using Azure AD client credentials flow."""
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        }

        response = requests.post(token_url, data=token_data, timeout=30)

        if response.status_code != 200:
            raise SQLMeshError(f"Failed to get access token: {response.text}")

        token_response = response.json()
        return token_response["access_token"]

    @property
    def workspace_id(self) -> str:
        """Get workspace ID from connection config."""
        workspace_id = self._extra_config.get("workspace_id")
        if not workspace_id:
            raise SQLMeshError("workspace_id is required for Fabric Spark connection")
        return workspace_id

    def _get_livy_endpoint(self) -> str:
        """Get Livy endpoint URL."""
        return f"https://api.fabric.microsoft.com/v1/workspaces/{self.workspace_id}/lakehouses/{self.default_lakehouse_id}/livyapi/versions/2023-12-01"

    @property
    def fabric_endpoint(self) -> str:
        """Get Fabric API endpoint URL."""
        return f"https://api.fabric.microsoft.com/v1/workspaces/{self.workspace_id}"

    @property
    def default_lakehouse_name(self) -> str:
        """Get default lakehouse name from connection config."""
        lakehouse_name = self._extra_config.get("lakehouse")
        if not lakehouse_name:
            raise SQLMeshError("lakehouse is required for Fabric Spark connection")
        return lakehouse_name

    @property
    def default_lakehouse_id(self) -> str:
        """Get default lakehouse ID by looking up the name."""
        if self._lakehouse_id_cache is None:
            lakehouse_id = self._get_lakehouse_id(self.default_lakehouse_name)
            if not lakehouse_id:
                raise SQLMeshError(f"Lakehouse '{self.default_lakehouse_name}' not found")
            self._lakehouse_id_cache = lakehouse_id
        return self._lakehouse_id_cache

    @property
    def catalog_support(self) -> CatalogSupport:
        return CatalogSupport.FULL_SUPPORT

    def get_current_catalog(self) -> t.Optional[str]:
        """Return the current catalog (lakehouse) name."""
        return self.default_lakehouse_name

    @property
    def _use_spark_session(self) -> bool:
        return False  # We use Livy REST API instead

    def _create_livy_session(self) -> t.Union[int, str]:
        """Create a new Livy session and return the session ID."""
        if self._livy_session_id is not None:
            return self._livy_session_id

        session_config = {
            "name": f"sqlmesh-session-{int(time.time())}",
            "kind": "spark",
            "conf": {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
            },
        }

        response = requests.post(
            f"{self._get_livy_endpoint()}/sessions",
            headers=self._session_headers,
            json=session_config,
            timeout=60,
        )

        if response.status_code not in (200, 201):
            raise SQLMeshError(f"Failed to create Livy session: {response.text}")

        session_data = response.json()
        # Handle different response formats from Fabric API
        if "id" in session_data:
            session_id = session_data["id"]
        else:
            raise SQLMeshError(f"Unexpected session response format: {response.text}")

        # Wait for session to be ready
        self._wait_for_session_ready(session_id)

        self._livy_session_id = session_id
        return session_id

    def _wait_for_async_operation(self, location_url: str, timeout: int = 300) -> None:
        """Wait for an async Fabric operation to complete by polling the location URL."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(location_url, headers=self._fabric_headers, timeout=30)

            if response.status_code == 200:
                # Operation completed successfully
                return
            if response.status_code == 202:
                # Still in progress, continue polling
                time.sleep(5)
                continue
            else:
                raise SQLMeshError(f"Async operation failed: {response.text}")

        raise SQLMeshError(f"Async operation did not complete within {timeout} seconds")

    def _wait_for_session_ready(self, session_id: t.Union[int, str], timeout: int = 300) -> None:
        """Wait for Livy session to be in 'idle' state."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self._get_livy_endpoint()}/sessions/{session_id}",
                headers=self._session_headers,
                timeout=30,
            )

            if response.status_code != 200:
                raise SQLMeshError(f"Failed to get session status: {response.text}")

            session_data = response.json()
            state = session_data.get("state")

            if state == "idle":
                return
            if state in ["error", "dead", "killed"]:
                raise SQLMeshError(f"Livy session failed with state: {state}")

            time.sleep(5)

        raise SQLMeshError(f"Livy session did not become ready within {timeout} seconds")

    def _execute_livy_statement(self, sql: str) -> t.Dict[str, t.Any]:
        """Execute a SQL statement through Livy and return the result."""
        session_id = self._create_livy_session()

        statement_config = {"code": sql, "kind": "sql"}

        response = requests.post(
            f"{self._get_livy_endpoint()}/sessions/{session_id}/statements",
            headers=self._session_headers,
            json=statement_config,
            timeout=60,
        )

        if response.status_code != 201:
            raise SQLMeshError(f"Failed to submit statement: {response.text}")

        statement_data = response.json()
        statement_id = statement_data["id"]

        # Wait for statement completion
        return self._wait_for_statement_completion(session_id, statement_id)

    def _wait_for_statement_completion(
        self, session_id: t.Union[int, str], statement_id: t.Union[int, str], timeout: int = 600
    ) -> t.Dict[str, t.Any]:
        """Wait for Livy statement to complete and return the result."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self._get_livy_endpoint()}/sessions/{session_id}/statements/{statement_id}",
                headers=self._session_headers,
                timeout=30,
            )

            if response.status_code != 200:
                raise SQLMeshError(f"Failed to get statement status: {response.text}")

            statement_data = response.json()
            state = statement_data.get("state")

            if state == "available":
                output = statement_data.get("output", {})
                if output.get("status") == "error":
                    error_name = output.get("ename", "Unknown")
                    error_value = output.get("evalue", "Unknown error")
                    raise SQLMeshError(f"SQL execution failed: {error_name}: {error_value}")
                return statement_data
            if state in ["error", "cancelled"]:
                raise SQLMeshError(f"Statement failed with state: {state}")

            time.sleep(2)

        raise SQLMeshError(f"Statement did not complete within {timeout} seconds")

    @property
    def cursor(self) -> t.Any:
        """Return a fake cursor object for compatibility."""
        return self

    def execute(
        self,
        expressions: t.Union[str, exp.Expression, t.Sequence[exp.Expression]],
        ignore_unsupported_errors: bool = False,
        quote_identifiers: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Execute SQL statement through Livy."""
        # Convert to single SQL string
        if isinstance(expressions, (list, tuple)):
            # Join multiple expressions
            sql = ";\n".join(
                expr.sql(dialect=self.dialect) if isinstance(expr, exp.Expression) else str(expr)
                for expr in expressions
            )
        elif isinstance(expressions, exp.Expression):
            sql = expressions.sql(dialect=self.dialect)
        else:
            sql = str(expressions)

        logger.debug(f"Executing SQL: {sql}")
        result = self._execute_livy_statement(sql)

        # Store result for potential fetchone/fetchall calls
        self._last_result = result

    def fetchone(
        self,
        query: t.Union[exp.Expression, str],
        ignore_unsupported_errors: bool = False,
        quote_identifiers: bool = False,
    ) -> t.Optional[t.Tuple[t.Any, ...]]:
        """Fetch one row from the query."""
        self.execute(
            query,
            ignore_unsupported_errors=ignore_unsupported_errors,
            quote_identifiers=quote_identifiers,
        )

        if not hasattr(self, "_last_result"):
            return None

        output = self._last_result.get("output", {})
        data = output.get("data", {})

        if "text/plain" in data:
            # Parse text output - this is a simplified implementation
            text_data = data["text/plain"]
            if isinstance(text_data, list) and text_data:
                # Return first row
                return tuple(text_data[0]) if isinstance(text_data[0], list) else (text_data[0],)

        return None

    def fetchall(
        self,
        query: t.Union[exp.Expression, str],
        ignore_unsupported_errors: bool = False,
        quote_identifiers: bool = False,
    ) -> t.List[t.Tuple[t.Any, ...]]:
        """Fetch all rows from the query."""
        self.execute(
            query,
            ignore_unsupported_errors=ignore_unsupported_errors,
            quote_identifiers=quote_identifiers,
        )

        if not hasattr(self, "_last_result"):
            return []

        output = self._last_result.get("output", {})
        data = output.get("data", {})

        if "text/plain" in data:
            # Parse text output - this is a simplified implementation
            text_data = data["text/plain"]
            if isinstance(text_data, list):
                return [tuple(row) if isinstance(row, list) else (row,) for row in text_data]

        return []

    def fetchdf(
        self, query: t.Union[exp.Expression, str], quote_identifiers: bool = False
    ) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame from a query or expression.
        """
        import pandas as pd

        result = self._execute_livy_statement(
            query.sql(dialect=self.dialect) if isinstance(query, exp.Expression) else query
        )

        output = result.get("output", {})
        data = output.get("data", {})

        # This is a simplified implementation - in practice, you'd need to parse
        # the Livy output format more carefully
        if "application/json" in data:
            json_data = data["application/json"]
            return pd.DataFrame(json_data)
        if "text/plain" in data:
            text_data = data["text/plain"]
            if isinstance(text_data, list) and text_data:
                # Try to create DataFrame from list data
                return pd.DataFrame(text_data)

        return pd.DataFrame()

    def create_catalog(self, catalog_name: t.Union[str, exp.Identifier]) -> None:
        """Create a lakehouse (catalog) using Fabric REST API."""
        # Convert to string if Identifier
        if isinstance(catalog_name, exp.Identifier):
            catalog_name_str = catalog_name.this
        else:
            catalog_name_str = catalog_name

        # Check if lakehouse already exists
        if self._lakehouse_exists(catalog_name_str):
            return

        lakehouse_config = {
            "displayName": catalog_name_str,
            "type": "Lakehouse",
            "enableSchemas": True,
        }

        response = requests.post(
            f"{self.fabric_endpoint}/items",
            headers=self._fabric_headers,
            json=lakehouse_config,
            timeout=60,
        )

        if response.status_code == 202:
            # Handle async lakehouse creation - poll the location header for completion
            location = response.headers.get("Location")
            if not location:
                raise SQLMeshError(
                    "Received 202 but no Location header for polling lakehouse creation"
                )

            # Wait for lakehouse creation to complete
            self._wait_for_async_operation(location)
            return
        if response.status_code not in (200, 201):
            if "already exists" in response.text.lower():
                return
            raise SQLMeshError(f"Failed to create lakehouse: {response.text}")

    def drop_catalog(self, catalog_name: t.Union[str, exp.Identifier]) -> None:
        """Drop a lakehouse (catalog) using Fabric REST API."""
        # Convert to string if Identifier
        if isinstance(catalog_name, exp.Identifier):
            catalog_name_str = catalog_name.this
        else:
            catalog_name_str = catalog_name

        # Get lakehouse ID
        lakehouse_id = self._get_lakehouse_id(catalog_name_str)
        if not lakehouse_id:
            return  # Catalog doesn't exist, nothing to do

        response = requests.delete(
            f"{self.fabric_endpoint}/items/{lakehouse_id}", headers=self._fabric_headers, timeout=60
        )

        if response.status_code not in (200, 204):
            if "not found" in response.text.lower():
                return
            raise SQLMeshError(f"Failed to delete lakehouse: {response.text}")

    def create_schema(
        self,
        schema_name: SchemaName,
        ignore_if_exists: bool = True,
        **kwargs: t.Any,
    ) -> None:
        """Create a schema in the default lakehouse."""
        schema = to_schema(schema_name)
        if not schema.db:
            raise SQLMeshError("Schema name is required")

        # In Fabric, we create a schema within the lakehouse using SQL
        create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema.db}"
        self.execute(create_schema_sql)

    def drop_schema(
        self,
        schema_name: SchemaName,
        ignore_if_not_exists: bool = True,
        cascade: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Drop a schema from the default lakehouse."""
        schema = to_schema(schema_name)
        if not schema.db:
            raise SQLMeshError("Schema name is required")

        # In Fabric, we drop a schema within the lakehouse using SQL
        drop_schema_sql = f"DROP SCHEMA {'IF EXISTS ' if ignore_if_not_exists else ''}{schema.db} {'CASCADE' if cascade else 'RESTRICT'}"
        self.execute(drop_schema_sql)

    def _lakehouse_exists(self, lakehouse_name: str) -> bool:
        """Check if a lakehouse exists."""
        return self._get_lakehouse_id(lakehouse_name) is not None

    def _get_lakehouse_id(self, lakehouse_name: str) -> t.Optional[str]:
        """Get lakehouse ID by name using the dedicated lakehouse list endpoint."""
        response = requests.get(
            f"{self.fabric_endpoint}/lakehouses", headers=self._fabric_headers, timeout=30
        )

        if response.status_code != 200:
            return None

        items = response.json().get("value", [])
        for item in items:
            if item.get("displayName") == lakehouse_name:
                return item.get("id")

        return None

    def _get_data_objects(
        self, schema_name: SchemaName, object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """
        Returns all the data objects that exist in the given schema.
        """
        schema = to_schema(schema_name)
        catalog_name = schema.catalog or self.default_lakehouse_name
        schema_db = schema.db or "default"

        # Use SHOW TABLES SQL command through Livy
        if schema.catalog:
            show_tables_sql = f"SHOW TABLES FROM {catalog_name}.{schema_db}"
        else:
            show_tables_sql = f"SHOW TABLES IN {schema_db}"

        if object_names:
            # Filter by specific table names if provided
            names_filter = "', '".join(object_names)
            show_tables_sql += f" LIKE '{names_filter}'"

        try:
            result = self._execute_livy_statement(show_tables_sql)
            output = result.get("output", {})
            data = output.get("data", {})

            objects = []
            if "text/plain" in data:
                text_data = data["text/plain"]
                if isinstance(text_data, list):
                    for row in text_data:
                        if isinstance(row, list) and len(row) >= 2:
                            table_name = row[1]  # Table name is typically in second column
                            objects.append(
                                DataObject(
                                    catalog=catalog_name,
                                    schema=schema_db,
                                    name=table_name,
                                    type=DataObjectType.TABLE,  # Fabric lakehouses primarily contain tables
                                )
                            )

            return objects
        except Exception as e:
            logger.warning(f"Failed to list tables in schema '{schema_name}': {e}")
            return []

    def close(self) -> None:
        """Close the Livy session and release resources."""
        if self._livy_session_id is not None:
            try:
                requests.delete(
                    f"{self._get_livy_endpoint()}/sessions/{self._livy_session_id}",
                    headers=self._session_headers,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"Failed to close Livy session: {e}")
            finally:
                self._livy_session_id = None

        super().close()

    def _begin_session(self, properties: SessionProperties) -> t.Any:
        """Begin a new session - ensure Livy session is created."""
        self._create_livy_session()

    def _end_session(self) -> None:
        """End the current session."""
        # Session cleanup is handled in close()
        pass

    @contextlib.contextmanager
    def transaction(self, condition: t.Optional[bool] = None) -> t.Iterator[None]:
        """Return a context manager for transactions - no-op for Livy."""
        # Use a proper generator context manager since Fabric doesn't support transactions
        yield

    def commit(self) -> None:
        """Commit transaction - no-op for Livy."""
        pass

    def rollback(self) -> None:
        """Rollback transaction - no-op for Livy."""
        pass
