# type: ignore
import typing as t
import pytest
from sqlglot import exp
from sqlglot.optimizer.qualify_columns import quote_identifiers

from sqlmesh.core.engine_adapter.fabricspark import FabricSparkEngineAdapter
import sqlmesh.core.dialect as d
from sqlmesh.core.model import load_sql_based_model
from sqlmesh.core.plan import Plan
from tests.core.engine_adapter.integration import TestContext
from pytest import FixtureRequest
from tests.core.engine_adapter.integration import (
    TestContext,
    generate_pytest_params,
    ENGINES_BY_NAME,
    IntegrationTestEngine,
)


@pytest.fixture(params=list(generate_pytest_params(ENGINES_BY_NAME["fabricspark"])))
def ctx(
    request: FixtureRequest,
    create_test_context: t.Callable[[IntegrationTestEngine, str, str], t.Iterable[TestContext]],
) -> t.Iterable[TestContext]:
    yield from create_test_context(*request.param)


@pytest.fixture
def engine_adapter(ctx: TestContext) -> FabricSparkEngineAdapter:
    assert isinstance(ctx.engine_adapter, FabricSparkEngineAdapter)
    return ctx.engine_adapter


def test_basic_table_operations(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test basic table creation, insertion, and querying operations."""
    test_table = ctx.table("basic_test_table")

    # Create table
    engine_adapter.execute(
        f"CREATE TABLE {test_table.sql(dialect=ctx.dialect)} (id INT, name STRING, created_date DATE)"
    )

    # Insert data
    engine_adapter.execute(
        f"INSERT INTO {test_table.sql(dialect=ctx.dialect)} VALUES (1, 'test', '2023-01-01')"
    )

    # Query data
    result = engine_adapter.fetchall(f"SELECT * FROM {test_table.sql(dialect=ctx.dialect)}")
    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][1] == "test"

    # Drop table
    engine_adapter.drop_table(test_table)


def test_temp_table_creation(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test temporary table creation and naming."""
    test_table = ctx.table("temp_test")
    temp_table = engine_adapter._get_temp_table(test_table)

    # Should use fabric_temp_ prefix instead of __temp_
    assert "fabric_temp_" in temp_table.sql() or "__temp_" not in temp_table.sql()


def test_catalog_operations(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test catalog-related operations."""
    # Test getting current catalog
    current_catalog = engine_adapter.get_current_catalog()
    assert current_catalog is not None

    # Test getting current database
    current_database = engine_adapter.get_current_database()
    assert current_database is not None


def test_data_types_support(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test various Spark data types are supported."""
    test_table = ctx.table("data_types_test")

    # Create table with various data types
    engine_adapter.execute(f"""
        CREATE TABLE {test_table.sql(dialect=ctx.dialect)} (
            int_col INT,
            bigint_col BIGINT,
            float_col FLOAT,
            double_col DOUBLE,
            decimal_col DECIMAL(10,2),
            string_col STRING,
            boolean_col BOOLEAN,
            date_col DATE,
            timestamp_col TIMESTAMP,
            array_col ARRAY<INT>,
            map_col MAP<STRING, INT>,
            struct_col STRUCT<field1: STRING, field2: INT>
        )
    """)

    # Insert sample data
    engine_adapter.execute(f"""
        INSERT INTO {test_table.sql(dialect=ctx.dialect)} VALUES (
            1,
            123456789,
            1.23,
            4.56789,
            123.45,
            'test_string',
            true,
            '2023-01-01',
            '2023-01-01 12:00:00',
            array(1, 2, 3),
            map('key1', 1, 'key2', 2),
            named_struct('field1', 'value1', 'field2', 42)
        )
    """)

    # Query and verify data
    result = engine_adapter.fetchone(f"SELECT * FROM {test_table.sql(dialect=ctx.dialect)}")
    assert result is not None
    assert result[0] == 1  # int_col
    assert result[5] == "test_string"  # string_col
    assert result[6] is True  # boolean_col

    # Drop table
    engine_adapter.drop_table(test_table)


def test_incremental_model_workflow(ctx: TestContext):
    """Test full incremental model workflow with Fabric Spark."""
    model_name = ctx.table("incremental_test")

    sqlmesh = ctx.create_context()

    # Create incremental model
    sqlmesh.upsert_model(
        load_sql_based_model(
            d.parse(
                f"""
                MODEL (
                    name {model_name.sql(dialect="spark")},
                    kind INCREMENTAL_BY_TIME_RANGE (
                        time_column ds
                    ),
                    start '2023-01-01',
                    cron '@daily',
                    dialect 'spark'
                );

                SELECT 
                    1 as id, 
                    'test' as name,
                    @start_ds as ds
                """
            )
        )
    )

    # Test plan creation and execution
    plan: Plan = sqlmesh.plan(auto_apply=True, no_prompts=True)
    assert len(plan.snapshots) == 1

    # Verify table was created
    target_table = exp.to_table(list(plan.snapshots.values())[0].table_name())
    quote_identifiers(target_table)

    # Query the created table
    result = sqlmesh.engine_adapter.fetchall(
        f"SELECT * FROM {target_table.sql(dialect=ctx.dialect)}"
    )
    assert len(result) > 0


def test_fabric_spark_specific_features(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test Fabric Spark specific functionality."""
    # Test that it's recognized as serverless
    assert engine_adapter.use_serverless is True

    # Test that transactions are not supported
    assert engine_adapter.SUPPORTS_TRANSACTIONS is False

    # Test that REPLACE TABLE is not supported
    assert engine_adapter.SUPPORTS_REPLACE_TABLE is False

    # Test dialect is spark
    assert engine_adapter.DIALECT == "spark"


def test_livy_session_connectivity(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test that Livy session connectivity works properly."""
    # Test basic SQL execution through Livy
    result = engine_adapter.fetchone("SELECT 1 as test_col")
    assert result is not None
    assert result[0] == 1

    # Test database context is set correctly
    current_db = engine_adapter.get_current_database()
    assert current_db is not None

    # Test simple aggregation
    result = engine_adapter.fetchone("SELECT COUNT(*) FROM (SELECT 1 as col UNION ALL SELECT 2)")
    assert result[0] == 2


def test_table_metadata_operations(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test table metadata retrieval operations."""
    test_table = ctx.table("metadata_test")

    # Create table
    engine_adapter.execute(f"""
        CREATE TABLE {test_table.sql(dialect=ctx.dialect)} (
            id INT COMMENT 'Primary key',
            name STRING COMMENT 'Name field',
            created_at TIMESTAMP
        ) COMMENT 'Test table for metadata'
    """)

    # Get table metadata
    data_objects = engine_adapter.get_data_objects(test_table.db, {test_table.name})
    assert len(data_objects) == 1

    table_metadata = data_objects[0]
    assert table_metadata.name == test_table.name
    assert table_metadata.type.is_table

    # Get column information
    columns = engine_adapter.columns(test_table)
    assert "id" in columns
    assert "name" in columns
    assert "created_at" in columns

    # Verify column types
    assert columns["id"].is_type("INT")
    assert columns["name"].is_type("STRING")
    assert columns["created_at"].is_type("TIMESTAMP")

    # Drop table
    engine_adapter.drop_table(test_table)


def test_error_handling(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test error handling for invalid operations."""
    # Test invalid SQL should raise appropriate error
    with pytest.raises(Exception):
        engine_adapter.execute("INVALID SQL STATEMENT")

    # Test querying non-existent table should raise error
    with pytest.raises(Exception):
        engine_adapter.fetchall("SELECT * FROM non_existent_table_12345")


def test_azure_authentication_context(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test that Azure authentication context is properly set up."""
    # This test verifies that the connection was established successfully
    # If authentication failed, we wouldn't have gotten this far

    # Test a simple query to verify authentication worked
    result = engine_adapter.fetchone("SELECT current_user()")
    assert result is not None

    # Test workspace/lakehouse context
    result = engine_adapter.fetchone("SELECT current_database()")
    assert result is not None


@pytest.mark.skip(reason="Requires actual Fabric workspace with shortcuts configured")
def test_fabric_shortcuts_integration(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test integration with Fabric shortcuts (requires real setup)."""
    # This test would require actual shortcuts to be configured
    # in the Fabric workspace, so it's skipped by default

    # Example test that would work with proper shortcuts:
    # result = engine_adapter.fetchall("SELECT * FROM shortcut_table LIMIT 1")
    # assert result is not None
    pass
