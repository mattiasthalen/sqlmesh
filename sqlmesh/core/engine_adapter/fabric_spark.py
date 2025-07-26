from __future__ import annotations

import contextlib
import logging
import random
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

    # Class-level session pool shared across all adapter instances
    # Format: {lakehouse_id: session_id}
    _shared_session_pool: t.Dict[str, t.Union[int, str]] = {}

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
        return CatalogSupport.SINGLE_CATALOG_ONLY

    def get_current_catalog(self) -> t.Optional[str]:
        """Return the current catalog (lakehouse) name."""
        return self.default_lakehouse_name

    @property
    def _use_spark_session(self) -> bool:
        return False  # We use Livy REST API instead

    def _create_livy_session(self) -> t.Union[int, str]:
        """Create a new Livy session and return the session ID."""
        lakehouse_id = self.default_lakehouse_id

        # Check instance-level session first
        if self._livy_session_id is not None:
            if self._check_session_status(self._livy_session_id):
                return self._livy_session_id
            # Session is no longer active, clear it
            self._livy_session_id = None

        # Check shared session pool
        shared_session_id = self._shared_session_pool.get(lakehouse_id)
        if shared_session_id is not None:
            if self._check_session_status(shared_session_id):
                logger.info(
                    f"Reusing shared session {shared_session_id} for lakehouse {lakehouse_id}"
                )
                self._livy_session_id = shared_session_id
                return shared_session_id
            # Session is no longer active, remove from shared pool
            self._shared_session_pool.pop(lakehouse_id, None)

        # Try to discover existing idle sessions
        reusable_session_id = self._find_idle_session()
        if reusable_session_id:
            logger.info(f"Discovered reusable session {reusable_session_id}")
            self._livy_session_id = reusable_session_id
            self._shared_session_pool[lakehouse_id] = reusable_session_id
            return reusable_session_id

        session_config = {
            "name": f"sqlmesh-session-{int(time.time())}",
            "kind": "spark",
            "conf": {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
            },
        }

        # Retry logic with exponential backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self._get_livy_endpoint()}/sessions",
                    headers=self._session_headers,
                    json=session_config,
                    timeout=60,
                )

                # Microsoft Fabric API might return different status codes for successful session creation
                # Check if we have a valid response with session ID regardless of status code
                try:
                    response_data = response.json()
                    if "id" in response_data and response.status_code < 500:
                        # Got a session ID, consider it success even with unexpected status codes
                        break
                except (ValueError, KeyError):
                    pass

                if response.status_code in (200, 201, 202):
                    break

                # Check if it's a rate limit error and we can retry
                if response.status_code == 430 or "rate limit" in response.text.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limit hit, retrying in {wait_time:.2f} seconds "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue

                # Non-retryable error or last attempt
                raise SQLMeshError(f"Failed to create Livy session: {response.text}")

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Request failed, retrying in {wait_time:.2f} seconds "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                raise SQLMeshError(
                    f"Failed to create Livy session after {max_retries} attempts: {e}"
                )

        session_data = response.json()
        # Handle different response formats from Fabric API
        if "id" in session_data:
            session_id = session_data["id"]
        else:
            raise SQLMeshError(f"Unexpected session response format: {response.text}")

        # Wait for session to be ready
        self._wait_for_session_ready(session_id)

        # Store in both instance and shared pool
        self._livy_session_id = session_id
        self._shared_session_pool[lakehouse_id] = session_id
        logger.info(f"Created and stored new session {session_id} for lakehouse {lakehouse_id}")
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

    def _wait_for_session_ready(self, session_id: t.Union[int, str], timeout: int = 120) -> None:
        """Wait for Livy session to be in 'idle' state."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self._get_livy_endpoint()}/sessions/{session_id}",
                headers=self._session_headers,
                timeout=30,
            )

            if response.status_code == 404:
                # Session might not be visible yet or was cleaned up
                logger.debug(f"Session {session_id} not found (404), continuing to wait...")
                time.sleep(5)
                continue
            elif response.status_code != 200:
                raise SQLMeshError(f"Failed to get session status: {response.text}")

            session_data = response.json()
            state = session_data.get("state")

            logger.debug(f"Session {session_id} state: {state}")

            if state == "idle":
                return
            if state in ["error", "dead", "killed"]:
                raise SQLMeshError(f"Livy session failed with state: {state}")
            elif state in ["not_started", "starting", "busy", "shutting_down"]:
                # These are normal intermediate states, continue waiting
                logger.debug(
                    f"Session {session_id} in intermediate state '{state}', continuing to wait..."
                )
                time.sleep(5)
            elif state == "success":
                # Session stopped successfully, but we need it to be idle
                raise SQLMeshError(f"Livy session stopped unexpectedly with state: {state}")
            else:
                # Unknown state, log it but continue waiting
                logger.warning(
                    f"Unknown session state '{state}' for session {session_id}, continuing to wait..."
                )
                time.sleep(5)

        raise SQLMeshError(
            f"Livy session {session_id} did not become ready within {timeout} seconds. Last state: {session_data.get('state', 'unknown')}"
        )

    def _check_session_status(self, session_id: t.Union[int, str]) -> bool:
        """Check if a Livy session is still active."""
        try:
            response = requests.get(
                f"{self._get_livy_endpoint()}/sessions/{session_id}",
                headers=self._session_headers,
                timeout=30,
            )

            if response.status_code == 404:
                return False  # Session doesn't exist
            if response.status_code != 200:
                logger.warning(f"Failed to check session status: {response.text}")
                return False

            session_data = response.json()
            state = session_data.get("state")

            # Session is active if it's in any non-terminal state
            return state in ["not_started", "starting", "idle", "busy", "shutting_down"]

        except requests.RequestException as e:
            logger.warning(f"Error checking session status: {e}")
            return False

    def _find_idle_session(self) -> t.Optional[t.Union[int, str]]:
        """Find an existing idle SQLMesh session that can be reused."""
        try:
            response = requests.get(
                f"{self._get_livy_endpoint()}/sessions",
                headers=self._session_headers,
                timeout=30,
            )

            if response.status_code != 200:
                logger.debug(f"Failed to list sessions: {response.text}")
                return None

            sessions_data = response.json()
            # Handle both possible response formats: {"sessions": [...]} or {"items": [...]}
            sessions = sessions_data.get("sessions", sessions_data.get("items", []))

            # Look for an idle session with SQLMesh naming pattern
            for session in sessions:
                session_name = session.get("name", "")
                # The API response uses "livyState" not "state"
                session_state = session.get("livyState", session.get("state"))
                session_id = session.get("id")

                # Reuse sessions in any usable state, not just idle
                if (
                    session_name.startswith("sqlmesh-session-")
                    and session_state in ["not_started", "starting", "idle"]
                    and session_id is not None
                ):
                    logger.info(f"Reusing existing session {session_id} in state '{session_state}'")
                    return session_id

            return None

        except requests.RequestException as e:
            logger.debug(f"Error listing sessions: {e}")
            return None

    def _is_session_active(self) -> bool:
        """Indicates whether or not a session is active."""
        # Check instance session first
        if self._livy_session_id is not None and self._check_session_status(self._livy_session_id):
            return True

        # Check shared session pool
        lakehouse_id = self.default_lakehouse_id
        shared_session_id = self._shared_session_pool.get(lakehouse_id)
        if shared_session_id is not None and self._check_session_status(shared_session_id):
            # Update instance to use the shared session
            self._livy_session_id = shared_session_id
            return True

        return False

    def _execute_livy_statement(self, sql: str) -> t.Dict[str, t.Any]:
        """Execute a SQL statement through Livy and return the result."""
        session_id = self._create_livy_session()

        statement_config = {"code": sql, "kind": "sql"}

        # Retry logic with exponential backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self._get_livy_endpoint()}/sessions/{session_id}/statements",
                    headers=self._session_headers,
                    json=statement_config,
                    timeout=60,
                )

                # Check if we got a valid statement response regardless of status code
                try:
                    response_data = response.json()
                    if "id" in response_data and response.status_code < 500:
                        # Got a statement ID, consider it success
                        break
                except (ValueError, KeyError):
                    pass

                if response.status_code == 201:
                    break

                # Check if it's a rate limit error and we can retry
                if response.status_code == 430 or "rate limit" in response.text.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Statement submission rate limit hit, retrying in {wait_time:.2f} seconds "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue

                # Non-retryable error or last attempt
                raise SQLMeshError(f"Failed to submit statement: {response.text}")

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Statement submission request failed, retrying in {wait_time:.2f} seconds "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                raise SQLMeshError(f"Failed to submit statement after {max_retries} attempts: {e}")

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
        """Create catalog operation not supported - single lakehouse only."""
        if isinstance(catalog_name, exp.Identifier):
            catalog_name_str = catalog_name.this
        else:
            catalog_name_str = catalog_name

        if catalog_name_str != self.default_lakehouse_name:
            raise SQLMeshError(
                f"Cannot create catalog '{catalog_name_str}'. "
                f"Fabric Spark adapter supports single lakehouse only: '{self.default_lakehouse_name}'"
            )

    def drop_catalog(self, catalog_name: t.Union[str, exp.Identifier]) -> None:
        """Drop catalog operation not supported - single lakehouse only."""
        if isinstance(catalog_name, exp.Identifier):
            catalog_name_str = catalog_name.this
        else:
            catalog_name_str = catalog_name

        if catalog_name_str != self.default_lakehouse_name:
            raise SQLMeshError(
                f"Cannot drop catalog '{catalog_name_str}'. "
                f"Fabric Spark adapter supports single lakehouse only: '{self.default_lakehouse_name}'"
            )

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
        # Retry logic for lakehouse lookup (authentication issues, rate limits)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.fabric_endpoint}/lakehouses", headers=self._fabric_headers, timeout=30
                )

                if response.status_code == 200:
                    items = response.json().get("value", [])
                    for item in items:
                        if item.get("displayName") == lakehouse_name:
                            return item.get("id")
                    return None  # Lakehouse not found in list

                # Check for auth or rate limit errors
                if response.status_code in (401, 403):
                    logger.warning(
                        f"Authentication error getting lakehouse list: {response.status_code}"
                    )
                    if attempt < max_retries - 1:
                        # Re-authenticate and retry
                        self._setup_authentication()
                        time.sleep(1)
                        continue
                elif response.status_code == 430 or "rate limit" in response.text.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limit getting lakehouse list, retrying in {wait_time:.2f} seconds "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue

                logger.warning(f"Failed to get lakehouse list: HTTP {response.status_code}")
                return None

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Request failed getting lakehouse list, retrying in {wait_time:.2f} seconds "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                logger.warning(f"Failed to get lakehouse list after {max_retries} attempts: {e}")
                return None

        return None

    def _get_data_objects(
        self, schema_name: SchemaName, object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """
        Returns all the data objects that exist in the given schema.
        """
        schema = to_schema(schema_name)

        # Only support queries within the default lakehouse
        if schema.catalog and schema.catalog != self.default_lakehouse_name:
            raise SQLMeshError(
                f"Cannot query catalog '{schema.catalog}'. "
                f"Fabric Spark adapter supports single lakehouse only: '{self.default_lakehouse_name}'"
            )

        schema_db = schema.db or "default"
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
                                    catalog=self.default_lakehouse_name,
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
        # Only clear the instance reference, but keep shared sessions alive
        # for other adapter instances to reuse
        if self._livy_session_id is not None:
            logger.debug(f"Releasing instance reference to session {self._livy_session_id}")
            self._livy_session_id = None

        super().close()

    @classmethod
    def cleanup_shared_sessions(cls) -> None:
        """Clean up all shared sessions. Use sparingly, only when needed."""
        logger.info(f"Cleaning up {len(cls._shared_session_pool)} shared sessions")
        cls._shared_session_pool.clear()

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
