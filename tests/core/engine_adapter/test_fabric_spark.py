# type: ignore
import os
from unittest.mock import patch

import pytest
from sqlglot import exp

from sqlmesh.core.engine_adapter import FabricSparkEngineAdapter
from sqlmesh.core.engine_adapter.shared import CatalogSupport
from sqlmesh.utils.errors import SQLMeshError

pytestmark = [pytest.mark.fabric_spark, pytest.mark.engine]


@pytest.fixture
def fabric_adapter():
    """Create a FabricSparkEngineAdapter with mocked configuration."""

    def connection_factory():
        return None

    with patch("sqlmesh.core.engine_adapter.fabric_spark.requests") as mock_requests:
        # Mock the token request
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"access_token": "test-access-token"}

        # Mock the lakehouse lookup request
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = {
            "value": [{"displayName": "test-lakehouse", "id": "test-lakehouse-id"}]
        }

        extra_config = {
            "workspace_id": "test-workspace-id",
            "lakehouse": "test-lakehouse",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "tenant_id": "test-tenant-id",
        }

        adapter = FabricSparkEngineAdapter(connection_factory=connection_factory, **extra_config)
        yield adapter


def test_adapter_properties(fabric_adapter):
    """Test basic adapter properties and configuration."""
    assert fabric_adapter.workspace_id == "test-workspace-id"
    assert fabric_adapter.default_lakehouse_name == "test-lakehouse"
    assert fabric_adapter.DIALECT == "spark"
    assert fabric_adapter.catalog_support == CatalogSupport.FULL_SUPPORT
    assert fabric_adapter.SUPPORTS_MATERIALIZED_VIEWS is True
    assert fabric_adapter.SUPPORTS_TRANSACTIONS is False


def test_endpoint_urls(fabric_adapter):
    """Test API endpoint URL generation."""
    expected_fabric_endpoint = "https://api.fabric.microsoft.com/v1/workspaces/test-workspace-id"
    assert fabric_adapter.fabric_endpoint == expected_fabric_endpoint

    expected_livy_endpoint = (
        f"{expected_fabric_endpoint}/lakehouses/test-lakehouse-id/livyapi/versions/2023-12-01"
    )
    assert fabric_adapter._get_livy_endpoint() == expected_livy_endpoint


def test_cursor_property(fabric_adapter):
    """Test cursor property returns self for compatibility."""
    assert fabric_adapter.cursor is fabric_adapter


def test_missing_config_validation():
    """Test validation of required configuration parameters."""

    def connection_factory():
        return None

    # Missing client credentials should raise error during initialization
    with pytest.raises(SQLMeshError, match="client_id, client_secret, and tenant_id are required"):
        FabricSparkEngineAdapter(
            connection_factory=connection_factory,
            workspace_id="test-workspace",
            lakehouse="test-lakehouse",
        )


def test_transaction_methods(fabric_adapter):
    """Test transaction-related methods are no-ops."""
    # These should not raise errors and do nothing
    fabric_adapter.commit()
    fabric_adapter.rollback()

    # Transaction context manager should work and do nothing
    with fabric_adapter.transaction():
        pass


def test_sql_generation(fabric_adapter):
    """Test SQL generation for common operations."""
    # Test that SQL expressions are generated with correct dialect
    select_expr = exp.select("*").from_("test_table")
    sql = select_expr.sql(dialect=fabric_adapter.dialect)
    assert "SELECT * FROM test_table" in sql


def test_lakehouse_name_validation(fabric_adapter):
    """Test lakehouse name property validation."""
    # Should return the configured lakehouse name
    assert fabric_adapter.default_lakehouse_name == "test-lakehouse"

    # Test with adapter that has no lakehouse configured
    def connection_factory():
        return None

    with patch("sqlmesh.core.engine_adapter.fabric_spark.requests") as mock_requests:
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"access_token": "test-token"}

        adapter_no_lakehouse = FabricSparkEngineAdapter(
            connection_factory=connection_factory,
            workspace_id="test-workspace",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id",
        )

        with pytest.raises(SQLMeshError, match="lakehouse is required"):
            _ = adapter_no_lakehouse.default_lakehouse_name


def test_result_parsing_methods(fabric_adapter):
    """Test result parsing without external API calls."""

    # Mock the _execute_livy_statement method to avoid actual API calls
    def mock_execute(sql):
        return {"output": {"data": {"text/plain": []}}}

    fabric_adapter._execute_livy_statement = mock_execute

    # Test fetchone/fetchall with no result
    assert fabric_adapter.fetchone("SELECT 1") is None
    assert fabric_adapter.fetchall("SELECT 1") == []

    # Test with mock result data
    def mock_execute_with_data(sql):
        return {"output": {"data": {"text/plain": [["value1", "value2"], ["value3", "value4"]]}}}

    fabric_adapter._execute_livy_statement = mock_execute_with_data

    # fetchone should return first row
    row = fabric_adapter.fetchone("SELECT * FROM test")
    assert row == ("value1", "value2")

    # fetchall should return all rows
    rows = fabric_adapter.fetchall("SELECT * FROM test")
    assert len(rows) == 2
    assert rows[0] == ("value1", "value2")
    assert rows[1] == ("value3", "value4")


# Integration tests specific to Fabric Spark functionality
@pytest.mark.slow
@pytest.mark.fabric_spark
def test_livy_session_creation():
    """Test Livy session creation functionality specific to Fabric Spark."""
    pytest.importorskip("requests")

    from sqlmesh.core.config.connection import FabricSparkConnectionConfig

    # Create a connection config using environment variables
    connection_config = FabricSparkConnectionConfig(
        workspace_id=os.getenv("FABRIC_WORKSPACE_ID"),
        lakehouse=os.getenv("FABRIC_LAKEHOUSE"),
        client_id=os.getenv("FABRIC_CLIENT_ID"),
        client_secret=os.getenv("FABRIC_CLIENT_SECRET"),
        tenant_id=os.getenv("FABRIC_TENANT_ID"),
    )

    # Create adapter directly
    adapter = connection_config.create_engine_adapter()

    try:
        # Test session creation
        session_id = adapter._create_livy_session()

        # Verify we got a valid session ID
        assert isinstance(session_id, int)
        assert session_id > 0
        assert adapter._livy_session_id == session_id

        # Test that creating another session reuses the existing one
        session_id_2 = adapter._create_livy_session()
        assert session_id_2 == session_id

    finally:
        # Clean up
        adapter.close()


@pytest.mark.slow
@pytest.mark.fabric_spark
def test_spark_sql_execution():
    """Test basic Spark SQL execution through Fabric's Livy API."""
    pytest.importorskip("requests")

    from sqlmesh.core.config.connection import FabricSparkConnectionConfig

    # Create a connection config using environment variables
    connection_config = FabricSparkConnectionConfig(
        workspace_id=os.getenv("FABRIC_WORKSPACE_ID"),
        lakehouse=os.getenv("FABRIC_LAKEHOUSE"),
        client_id=os.getenv("FABRIC_CLIENT_ID"),
        client_secret=os.getenv("FABRIC_CLIENT_SECRET"),
        tenant_id=os.getenv("FABRIC_TENANT_ID"),
    )

    # Create adapter directly
    adapter = connection_config.create_engine_adapter()

    try:
        # Test basic SQL execution - SELECT 1
        result = adapter.fetchone("SELECT 1 as test_value")
        assert result is not None
        assert len(result) == 1
        assert result[0] == 1

        # Test fetchall with a VALUES query (standard Spark SQL)
        values_query = (
            "SELECT * FROM VALUES (1, 'first'), (2, 'second'), (3, 'third') AS t(id, name)"
        )
        results = adapter.fetchall(values_query)

        assert len(results) == 3
        assert results[0] == (1, "first")
        assert results[1] == (2, "second")
        assert results[2] == (3, "third")

        # Test SHOW DATABASES (should work in any Spark environment)
        databases = adapter.fetchall("SHOW DATABASES")
        assert len(databases) > 0  # Should have at least the default database

        # Test temporary view operations
        adapter.execute(
            "CREATE OR REPLACE TEMPORARY VIEW test_temp_view AS SELECT 42 as answer, 'hello' as greeting"
        )

        temp_view_result = adapter.fetchone("SELECT answer, greeting FROM test_temp_view")
        assert temp_view_result == (42, "hello")

    finally:
        # Clean up
        try:
            adapter.execute("DROP VIEW IF EXISTS test_temp_view")
        except:
            pass  # Ignore cleanup errors
        adapter.close()
