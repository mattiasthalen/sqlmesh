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
        # Fabric Spark uses lakehouse as catalog name
        try:
            result = self.fetchone("SELECT current_catalog()")
            return result[0] if result else None  # type: ignore
        except Exception:
            # Fallback to lakehouse name from connection config
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

        # Get column names from cursor description if available
        if hasattr(self.cursor, "description") and self.cursor.description:
            columns = [desc[0] for desc in self.cursor.description]
        else:
            # Fallback: use generic column names
            columns = [f"col_{i}" for i in range(len(results[0]) if results else 0)]

        return pd.DataFrame(results, columns=columns)

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
                        db_name = row[0] if row[0] else schema_name_str
                        table_name = row[1]
                        is_temp = False
                    else:
                        # Full format with temp flag
                        db_name = row[0] if row[0] else schema_name_str
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
                                # Treat materialized lake views as regular views for SQLMesh compatibility
                                object_type = DataObjectType.VIEW
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
            create_sql = (
                f"CREATE OR REPLACE TABLE {view_name_sql} "
                f"TBLPROPERTIES ('lakehouse.table.type' = 'lakehouse_materialized_view') "
                f"AS {query.sql(dialect=self.dialect)}"
            )
            self.execute(create_sql)
