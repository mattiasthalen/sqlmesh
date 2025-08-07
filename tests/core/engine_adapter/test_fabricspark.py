# type: ignore
import typing as t
from unittest.mock import patch

import pytest
from pytest_mock.plugin import MockerFixture

from sqlmesh.core.engine_adapter.fabricspark import FabricSparkEngineAdapter
from sqlmesh.core.config.connection import FabricSparkConnectionConfig, parse_connection_config
from sqlmesh.utils.errors import ConfigError

pytestmark = [pytest.mark.fabricspark, pytest.mark.engine]


@pytest.fixture
def adapter(make_mocked_engine_adapter: t.Callable) -> FabricSparkEngineAdapter:
    return make_mocked_engine_adapter(FabricSparkEngineAdapter)


def test_dialect(adapter: FabricSparkEngineAdapter):
    """Test that the adapter uses the correct dialect."""
    assert adapter.DIALECT == "spark"


def test_supports_transactions(adapter: FabricSparkEngineAdapter):
    """Test that Fabric Spark does not support transactions."""
    assert adapter.SUPPORTS_TRANSACTIONS is False


def test_supports_replace_table(adapter: FabricSparkEngineAdapter):
    """Test that Fabric Spark does not support REPLACE TABLE."""
    assert adapter.SUPPORTS_REPLACE_TABLE is False


def test_use_serverless(adapter: FabricSparkEngineAdapter):
    """Test that Fabric Spark is considered serverless."""
    assert adapter.use_serverless is True


def test_get_current_catalog(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test getting current catalog."""
    # The implementation directly returns database from credentials
    # Mock the database property to return a string value
    adapter.connection.credentials.database = "test_catalog"

    result = adapter.get_current_catalog()

    # Should return the database name from connection credentials
    assert result == "test_catalog"

    # Test with different catalog name
    adapter.connection.credentials.database = "another_catalog"
    result = adapter.get_current_catalog()
    assert result == "another_catalog"


def test_set_current_catalog_not_supported(adapter: FabricSparkEngineAdapter):
    """Test that setting current catalog raises NotImplementedError for single-catalog engines."""
    with pytest.raises(NotImplementedError):
        adapter.set_current_catalog("test_catalog")


def test_get_current_database(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test getting current database."""
    fetchone_mock = mocker.patch.object(adapter, "fetchone", return_value=("test_database",))

    result = adapter.get_current_database()

    assert result == "test_database"
    fetchone_mock.assert_called_once()


@patch("sqlmesh.core.engine_adapter.fabricspark.requests.post")
@patch("sqlmesh.core.engine_adapter.fabricspark.get_cli_access_token")
def test_livy_session_creation(mock_get_token, mock_post):
    """Test Livy session creation."""
    from sqlmesh.core.engine_adapter.fabricspark import FabricSparkCredentials, LivySession
    from azure.core.credentials import AccessToken

    # Mock access token
    mock_token = AccessToken(token="test-token", expires_on=9999999999)
    mock_get_token.return_value = mock_token

    # Mock session creation response
    mock_post.return_value.json.return_value = {"id": 123, "state": "idle"}
    mock_post.return_value.raise_for_status.return_value = None

    credentials = FabricSparkCredentials(
        workspace_id="test-workspace", lakehouse_id="test-lakehouse", database="test_db"
    )

    session = LivySession(credentials)

    # Mock the ready check
    with patch.object(session, "_wait_for_session_ready"):
        session_id = session.create_session()

    assert session_id == 123
    assert session.session_id == 123


@patch("sqlmesh.core.engine_adapter.fabricspark.requests")
@patch("sqlmesh.core.engine_adapter.fabricspark.get_cli_access_token")
def test_cursor_execute(mock_get_token, mock_requests):
    """Test cursor SQL execution."""
    from sqlmesh.core.engine_adapter.fabricspark import (
        FabricSparkCredentials,
        LivySession,
        FabricSparkCursor,
    )
    from azure.core.credentials import AccessToken

    # Mock access token
    mock_token = AccessToken(token="test-token", expires_on=9999999999)
    mock_get_token.return_value = mock_token

    # Mock session creation
    mock_requests.post.return_value.json.return_value = {"id": 123}
    mock_requests.post.return_value.raise_for_status.return_value = None
    mock_requests.get.return_value.json.return_value = {"state": "idle"}
    mock_requests.get.return_value.raise_for_status.return_value = None

    credentials = FabricSparkCredentials(
        workspace_id="test-workspace", lakehouse_id="test-lakehouse", database="test_db"
    )

    session = LivySession(credentials)
    cursor = FabricSparkCursor(session)

    # Mock statement execution
    with patch.object(session, "execute_statement") as mock_execute:
        mock_execute.return_value = {
            "output": {
                "status": "ok",
                "data": {
                    "application/json": {
                        "data": [["value1", "value2"]],
                        "schema": {
                            "fields": [
                                {"name": "col1", "type": "string"},
                                {"name": "col2", "type": "string"},
                            ]
                        },
                    }
                },
            }
        }

        cursor.execute("SELECT * FROM test_table")

        # Should execute USE statement first, then the query
        assert mock_execute.call_count == 2
        assert "USE test_db" in str(mock_execute.call_args_list[0])
        assert "SELECT * FROM test_table" in str(mock_execute.call_args_list[1])


def test_connection_types():
    """Test that all connection classes are properly defined."""
    from sqlmesh.core.engine_adapter.fabricspark import (
        FabricSparkCredentials,
        FabricSparkConnection,
        fabricspark_connect,
    )

    # Test credentials
    creds = FabricSparkCredentials(
        workspace_id="test-ws", lakehouse_id="test-lh", database="test_db"
    )
    assert creds.workspace_id == "test-ws"
    assert creds.lakehouse_id == "test-lh"
    assert creds.database == "test_db"
    assert "test-ws" in creds.lakehouse_endpoint
    assert "test-lh" in creds.lakehouse_endpoint

    # Test connection factory
    with patch("sqlmesh.core.engine_adapter.fabricspark.LivySession"):
        conn = fabricspark_connect(
            workspace_id="test-ws", lakehouse_id="test-lh", database="test_db"
        )
        assert isinstance(conn, FabricSparkConnection)


class TestFabricSparkConnectionConfig:
    """Test cases for FabricSparkConnectionConfig."""

    def test_basic_config_default(self):
        """Test basic configuration with default authentication (service_principal)."""
        config = FabricSparkConnectionConfig(
            workspace_id="12345678-1234-1234-1234-123456789abc",
            lakehouse_id="87654321-4321-4321-4321-cba987654321",
            database="test_lakehouse",
        )

        assert config.type_ == "fabricspark"
        assert config.DIALECT == "spark"
        assert config.authentication == "service_principal"
        assert config.workspace_id == "12345678-1234-1234-1234-123456789abc"
        assert config.lakehouse_id == "87654321-4321-4321-4321-cba987654321"
        assert config.database == "test_lakehouse"
        assert config.concurrent_tasks == 1
        assert config.pre_ping is False

    def test_service_principal_config_valid(self):
        """Test service principal configuration with all required fields."""
        config = FabricSparkConnectionConfig(
            workspace_id="12345678-1234-1234-1234-123456789abc",
            lakehouse_id="87654321-4321-4321-4321-cba987654321",
            database="test_lakehouse",
            authentication="service_principal",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id",
        )

        assert config.authentication == "service_principal"
        assert config.client_id == "test-client-id"
        assert config.client_secret == "test-client-secret"
        assert config.tenant_id == "test-tenant-id"

    def test_service_principal_config_missing_fields(self):
        """Test service principal configuration with missing required fields."""
        with pytest.raises(ConfigError, match="Service principal authentication requires"):
            FabricSparkConnectionConfig(
                workspace_id="12345678-1234-1234-1234-123456789abc",
                lakehouse_id="87654321-4321-4321-4321-cba987654321",
                database="test_lakehouse",
                authentication="service_principal",
                client_id="test-client-id",
                # Missing client_secret and tenant_id
            )

    def test_connection_kwargs_keys(self):
        """Test that all expected connection kwargs are included."""
        config = FabricSparkConnectionConfig(
            workspace_id="12345678-1234-1234-1234-123456789abc",
            lakehouse_id="87654321-4321-4321-4321-cba987654321",
            database="test_lakehouse",
        )

        expected_keys = {
            "workspace_id",
            "lakehouse_id",
            "database",
            "endpoint",
            "authentication",
            "client_id",
            "client_secret",
            "tenant_id",
            "access_token",
            "spark_config",
            "connect_retries",
            "connect_timeout",
        }

        assert config._connection_kwargs_keys == expected_keys

    def test_parse_connection_config_dict(self):
        """Test parsing from dictionary configuration."""
        config_dict = {
            "type": "fabricspark",
            "workspace_id": "12345678-1234-1234-1234-123456789abc",
            "lakehouse_id": "87654321-4321-4321-4321-cba987654321",
            "database": "test_lakehouse",
            "authentication": "service_principal",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "tenant_id": "test-tenant-id",
        }

        config = parse_connection_config(config_dict)
        assert isinstance(config, FabricSparkConnectionConfig)
        assert config.type_ == "fabricspark"
        assert config.workspace_id == "12345678-1234-1234-1234-123456789abc"

    def test_engine_adapter_property(self):
        """Test that the correct engine adapter is returned."""
        config = FabricSparkConnectionConfig(
            workspace_id="12345678-1234-1234-1234-123456789abc",
            lakehouse_id="87654321-4321-4321-4321-cba987654321",
            database="test_lakehouse",
        )

        assert config._engine_adapter == FabricSparkEngineAdapter


def test_insert_overwrite_by_time_partition_behavior(
    adapter: FabricSparkEngineAdapter, mocker: MockerFixture
):
    """Test insert_overwrite_by_time_partition behavior to reproduce the reported issue."""
    import pandas as pd
    from sqlglot import exp

    table_name = "test_table"

    # Mock fetchone to return table metadata
    mocker.patch.object(adapter, "fetchone", return_value=("test_database",))

    # Mock columns method to return table schema
    columns_to_types = {"id": exp.DataType.build("int"), "ds": exp.DataType.build("string")}
    mocker.patch.object(adapter, "columns", return_value=columns_to_types)

    # Mock fetchall to simulate table contents at different stages
    fetchall_results = []

    def mock_fetchall(*args, **kwargs):
        # Return the next result in sequence
        if fetchall_results:
            return fetchall_results.pop(0)
        return []

    mocker.patch.object(adapter, "fetchall", side_effect=mock_fetchall)

    # Mock execute to capture SQL statements
    executed_statements = []

    def mock_execute(sql, *args, **kwargs):
        executed_statements.append(str(sql))

    mocker.patch.object(adapter, "execute", side_effect=mock_execute)

    # Test data setup - simulate the test scenario
    initial_data = pd.DataFrame(
        [
            {"id": 1, "ds": "2022-01-01"},
            {"id": 2, "ds": "2022-01-02"},
            {"id": 3, "ds": "2022-01-03"},
        ]
    )

    # First operation: insert_overwrite_by_time_partition with start="2022-01-02", end="2022-01-03"
    # This should keep data where ds NOT IN ('2022-01-02', '2022-01-03') and then add back [2,3] for [2022-01-02,2022-01-03]
    adapter.insert_overwrite_by_time_partition(
        table_name,
        initial_data,
        start="2022-01-02",
        end="2022-01-03",
        time_formatter=lambda x, _: exp.Literal.string(x),
        time_column="ds",
        columns_to_types=columns_to_types,
    )

    print("First operation executed statements:")
    for stmt in executed_statements:
        print(f"  {stmt}")

    # Second operation data
    overwrite_data = pd.DataFrame(
        [
            {"id": 10, "ds": "2022-01-03"},
            {"id": 4, "ds": "2022-01-04"},
            {"id": 5, "ds": "2022-01-05"},
        ]
    )

    executed_statements.clear()

    # Second operation: insert_overwrite_by_time_partition with start="2022-01-03", end="2022-01-05"
    # This should keep data where ds NOT IN ('2022-01-03', '2022-01-04', '2022-01-05') and add [10,4,5] for [2022-01-03,2022-01-04,2022-01-05]
    # Final expected result: id=2,ds=2022-01-02 (from before range) + [10,4,5] for [2022-01-03,2022-01-04,2022-01-05]
    adapter.insert_overwrite_by_time_partition(
        table_name,
        overwrite_data,
        start="2022-01-03",
        end="2022-01-05",
        time_formatter=lambda x, _: exp.Literal.string(x),
        time_column="ds",
        columns_to_types=columns_to_types,
    )

    print("Second operation executed statements:")
    for stmt in executed_statements:
        print(f"  {stmt}")

    # With the fix, we should see DELETE + INSERT statements instead of INSERT OVERWRITE
    # First operation should have 2 statements: DELETE ... WHERE ds BETWEEN ... and INSERT ...
    # Second operation should have 2 statements: DELETE ... WHERE ds BETWEEN ... and INSERT ...

    # Verify that DELETE statements are present (indicating DELETE_INSERT strategy is being used)
    delete_statements = [
        stmt for stmt in executed_statements if stmt.strip().upper().startswith("DELETE")
    ]
    insert_statements = [
        stmt for stmt in executed_statements if stmt.strip().upper().startswith("INSERT")
    ]

    assert len(delete_statements) > 0, f"Expected DELETE statements but got: {executed_statements}"
    assert len(insert_statements) > 0, f"Expected INSERT statements but got: {executed_statements}"

    # The DELETE should target the specific time range
    delete_stmt = delete_statements[0]
    assert "WHERE ds BETWEEN" in delete_stmt or 'WHERE "ds" BETWEEN' in delete_stmt, (
        f"DELETE statement should have time range condition: {delete_stmt}"
    )

    # The INSERT should not be INSERT OVERWRITE (which would replace entire table)
    insert_stmt = insert_statements[0]
    assert "INSERT OVERWRITE" not in insert_stmt, (
        f"Should use INSERT not INSERT OVERWRITE: {insert_stmt}"
    )


def test_insert_overwrite_by_time_partition_integration_scenario(
    adapter: FabricSparkEngineAdapter, mocker: MockerFixture
):
    """Test the full integration scenario that was failing in the original test."""
    import pandas as pd
    from sqlglot import exp

    table_name = "test_table"

    # Mock the necessary adapter methods
    mocker.patch.object(adapter, "fetchone", return_value=("test_database",))

    columns_to_types = {"id": exp.DataType.build("int"), "ds": exp.DataType.build("string")}
    mocker.patch.object(adapter, "columns", return_value=columns_to_types)

    # Track all executed statements
    executed_statements = []

    def mock_execute(sql, *args, **kwargs):
        executed_statements.append(str(sql))

    mocker.patch.object(adapter, "execute", side_effect=mock_execute)

    # Simulate the exact integration test scenario

    # Step 1: Initial data insert (ids 1,2,3 for dates 2022-01-01,02,03)
    initial_data = pd.DataFrame(
        [
            {"id": 1, "ds": "2022-01-01"},
            {"id": 2, "ds": "2022-01-02"},
            {"id": 3, "ds": "2022-01-03"},
        ]
    )

    # First overwrite: start="2022-01-02", end="2022-01-03"
    # Should delete data for 2022-01-02 and 2022-01-03, then insert ids [2,3] back
    # Expected remaining after first op: all initial data (since we're inserting the same data back)
    adapter.insert_overwrite_by_time_partition(
        table_name,
        initial_data,
        start="2022-01-02",
        end="2022-01-03",
        time_formatter=lambda x, _: exp.Literal.string(x),
        time_column="ds",
        columns_to_types=columns_to_types,
    )

    # Should have DELETE and INSERT statements
    first_op_statements = executed_statements.copy()
    executed_statements.clear()

    assert len(first_op_statements) == 2, (
        f"Expected 2 statements (DELETE + INSERT), got {len(first_op_statements)}"
    )
    assert first_op_statements[0].strip().upper().startswith("DELETE"), (
        f"First should be DELETE: {first_op_statements[0]}"
    )
    assert first_op_statements[1].strip().upper().startswith("INSERT"), (
        f"Second should be INSERT: {first_op_statements[1]}"
    )

    # Step 2: Second overwrite with new data
    # New data: ids [10,4,5] for dates [2022-01-03,04,05]
    # Range: start="2022-01-03", end="2022-01-05"
    # Should delete data for 2022-01-03,04,05 and insert new data
    # Expected final result: id=2,ds=2022-01-02 (preserved from before) + [10,4,5] for [03,04,05]
    overwrite_data = pd.DataFrame(
        [
            {"id": 10, "ds": "2022-01-03"},
            {"id": 4, "ds": "2022-01-04"},
            {"id": 5, "ds": "2022-01-05"},
        ]
    )

    adapter.insert_overwrite_by_time_partition(
        table_name,
        overwrite_data,
        start="2022-01-03",
        end="2022-01-05",
        time_formatter=lambda x, _: exp.Literal.string(x),
        time_column="ds",
        columns_to_types=columns_to_types,
    )

    # Should have DELETE and INSERT statements for the second operation
    second_op_statements = executed_statements.copy()

    assert len(second_op_statements) == 2, (
        f"Expected 2 statements (DELETE + INSERT), got {len(second_op_statements)}"
    )
    assert second_op_statements[0].strip().upper().startswith("DELETE"), (
        f"First should be DELETE: {second_op_statements[0]}"
    )
    assert second_op_statements[1].strip().upper().startswith("INSERT"), (
        f"Second should be INSERT: {second_op_statements[1]}"
    )

    # Verify the DELETE statements target the correct time ranges
    first_delete = first_op_statements[0]
    second_delete = second_op_statements[0]

    # First DELETE should target 2022-01-02 to 2022-01-03 range
    assert "2022-01-02" in first_delete and "2022-01-03" in first_delete, (
        f"First DELETE should target 2022-01-02 to 2022-01-03: {first_delete}"
    )

    # Second DELETE should target 2022-01-03 to 2022-01-05 range
    assert "2022-01-03" in second_delete and "2022-01-05" in second_delete, (
        f"Second DELETE should target 2022-01-03 to 2022-01-05: {second_delete}"
    )

    # The key insight: With DELETE_INSERT strategy, data outside the time range is preserved
    # This is exactly what the integration test expects - that id=2,ds=2022-01-02 survives
    # the second operation because 2022-01-02 is outside the 2022-01-03 to 2022-01-05 range
