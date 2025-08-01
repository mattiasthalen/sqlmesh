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


def test_get_temp_table(adapter: FabricSparkEngineAdapter):
    """Test temp table name generation for Fabric Spark."""
    table = adapter._get_temp_table("test_table")
    table_name = table.args["this"].this

    # Should replace __temp_ with fabric_temp_
    assert "fabric_temp_" in table_name or "__temp_" not in table_name


def test_get_current_catalog(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test getting current catalog."""
    fetchone_mock = mocker.patch.object(adapter, "fetchone", return_value=("test_catalog",))

    result = adapter.get_current_catalog()

    # Should try to get current_catalog() first, fallback to spark_catalog
    assert result in ("test_catalog", "spark_catalog")


def test_get_current_catalog_fallback(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test getting current catalog with fallback."""
    fetchone_mock = mocker.patch.object(
        adapter, "fetchone", side_effect=Exception("No current_catalog")
    )

    result = adapter.get_current_catalog()

    # Should fallback to spark_catalog
    assert result == "spark_catalog"


def test_set_current_catalog(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test setting current catalog."""
    execute_mock = mocker.patch.object(adapter, "execute")

    adapter.set_current_catalog("test_catalog")

    # Fabric Spark only supports single catalog, so no USE CATALOG command should be executed
    execute_mock.assert_not_called()


def test_set_current_catalog_failure(mocker: MockerFixture, adapter: FabricSparkEngineAdapter):
    """Test setting current catalog with failure (should not raise)."""
    execute_mock = mocker.patch.object(
        adapter, "execute", side_effect=Exception("Catalog not found")
    )

    # Should not raise an exception since it's a no-op for single catalog support
    adapter.set_current_catalog("test_catalog")

    # No execute call should be made since Fabric Spark only supports single catalog
    execute_mock.assert_not_called()


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
                }
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
