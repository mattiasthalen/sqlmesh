from __future__ import annotations

import typing as t
from sqlglot import exp
from sqlmesh.core.engine_adapter.mssql import MSSQLEngineAdapter
from sqlmesh.core.engine_adapter.shared import InsertOverwriteStrategy, SourceQuery

if t.TYPE_CHECKING:
    from sqlmesh.core._typing import SchemaName, TableName
    from sqlmesh.core.engine_adapter._typing import QueryOrDF


class FabricWarehouseAdapter(MSSQLEngineAdapter):
    """
    Adapter for Microsoft Fabric Warehouses.
    """

    DIALECT = "tsql"
    SUPPORTS_INDEXES = False
    SUPPORTS_TRANSACTIONS = False

    INSERT_OVERWRITE_STRATEGY = InsertOverwriteStrategy.DELETE_INSERT

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        self.database = kwargs.get("database")

        super().__init__(*args, **kwargs)

        if not self.database:
            raise ValueError(
                "The 'database' parameter is required in the connection config for the FabricWarehouseAdapter."
            )
        try:
            self.execute(f"USE [{self.database}]")
        except Exception as e:
            raise RuntimeError(f"Failed to set database context to '{self.database}'. Reason: {e}")

    def _get_schema_name(self, name: t.Union[TableName, SchemaName]) -> str:
        """Extracts the schema name from a sqlglot object or string."""
        table = exp.to_table(name)

        # Handle case where db part contains the schema name
        if table.db:
            if isinstance(table.db, exp.Identifier):
                return table.db.name
            if isinstance(table.db, str):
                return table.db

        # Handle case where the name is just a schema name (no table part)
        if table.this and hasattr(table.this, "name") and not table.db:
            return table.this.name

        # If no schema is specified, default to 'dbo' (SQL Server default schema)
        # This handles cases like temporary table names without schema qualification
        return "dbo"

    def create_schema(
        self,
        schema_name: SchemaName,
        ignore_if_exists: bool = True,
        warn_on_error: bool = True,
        properties: t.Optional[t.List[exp.Expression]] = None,
    ) -> None:
        """
        Creates a schema in a Microsoft Fabric Warehouse.

        Overridden to handle Fabric's specific T-SQL requirements.
        T-SQL's `CREATE SCHEMA` command does not support `IF NOT EXISTS`, so this
        implementation first checks for the schema's existence in the
        `INFORMATION_SCHEMA.SCHEMATA` view.
        """
        # Extract schema name if it's a complex object
        if isinstance(schema_name, (exp.Table, exp.Identifier)):
            schema_name_str = self._get_schema_name(schema_name)
        else:
            schema_name_str = str(schema_name)

        # Validate schema name is not empty
        if not schema_name_str or schema_name_str.strip() == "":
            raise ValueError(f"Invalid empty schema name extracted from: {schema_name}")

        sql = (
            exp.select("1")
            .from_(f"{self.database}.INFORMATION_SCHEMA.SCHEMATA")
            .where(f"SCHEMA_NAME = '{schema_name_str}'")
        )
        if self.fetchone(sql):
            return
        self.execute(f"USE [{self.database}]")
        self.execute(f"CREATE SCHEMA [{schema_name_str}]")

    def _create_table_from_columns(
        self,
        table_name: TableName,
        columns_to_types: t.Dict[str, exp.DataType],
        primary_key: t.Optional[t.Tuple[str, ...]] = None,
        exists: bool = True,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Creates a table, ensuring the schema exists first and that all
        object names are fully qualified with the database.
        """
        table_exp = exp.to_table(table_name)
        schema_name = self._get_schema_name(table_name)

        self.create_schema(schema_name)

        fully_qualified_table_name = f"[{self.database}].[{schema_name}].[{table_exp.name}]"

        column_defs = ", ".join(
            f"[{col}] {kind.sql(dialect=self.dialect)}" for col, kind in columns_to_types.items()
        )

        create_table_sql = f"CREATE TABLE {fully_qualified_table_name} ({column_defs})"

        if not exists:
            self.execute(create_table_sql)
            return

        if not self.table_exists(table_name):
            self.execute(create_table_sql)

        if table_description and self.comments_enabled:
            qualified_table_for_comment = self._fully_qualify(table_name)
            self._create_table_comment(qualified_table_for_comment, table_description)
            if column_descriptions and self.comments_enabled:
                self._create_column_comments(qualified_table_for_comment, column_descriptions)

    def table_exists(self, table_name: TableName) -> bool:
        """
        Checks if a table exists.

        Overridden to query the uppercase `INFORMATION_SCHEMA` required
        by case-sensitive Fabric environments.
        """
        table = exp.to_table(table_name)
        schema = self._get_schema_name(table_name)

        sql = (
            exp.select("1")
            .from_(f"{self.database}.INFORMATION_SCHEMA.TABLES")
            .where(f"TABLE_NAME = '{table.alias_or_name}'")
            .where(f"TABLE_SCHEMA = '{schema}'")
        )

        result = self.fetchone(sql, quote_identifiers=True)

        return result[0] == 1 if result else False

    def _fully_qualify(self, name: t.Union[TableName, SchemaName]) -> exp.Table:
        """Ensures an object name is prefixed with the configured database."""
        table = exp.to_table(name)
        # Ensure we always have a catalog (database) set
        catalog = table.catalog or exp.to_identifier(self.database)
        # Ensure we have a schema (db) set - if not provided, use default
        schema = table.db or exp.to_identifier("dbo")
        return exp.Table(this=table.this, db=schema, catalog=catalog)

    def create_view(
        self,
        view_name: TableName,
        query_or_df: QueryOrDF,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        replace: bool = True,
        materialized: bool = False,
        materialized_properties: t.Optional[t.Dict[str, t.Any]] = None,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        view_properties: t.Optional[t.Dict[str, exp.Expression]] = None,
        **create_kwargs: t.Any,
    ) -> None:
        """
        Creates a view from a query or DataFrame.

        Overridden to ensure that the view name and all tables referenced
        in the source query are fully qualified with the database name,
        as required by Fabric.
        """
        view_schema = self._get_schema_name(view_name)
        self.create_schema(view_schema)

        qualified_view_name = self._fully_qualify(view_name)

        if isinstance(query_or_df, exp.Expression):
            for table in query_or_df.find_all(exp.Table):
                if not table.catalog:
                    qualified_table = self._fully_qualify(table)
                    table.replace(qualified_table)

        return super().create_view(
            qualified_view_name,
            query_or_df,
            columns_to_types,
            replace,
            materialized,
            table_description=table_description,
            column_descriptions=column_descriptions,
            view_properties=view_properties,
            **create_kwargs,
        )

    def columns(
        self, table_name: TableName, include_pseudo_columns: bool = False
    ) -> t.Dict[str, exp.DataType]:
        """
        Fetches column names and types for the target table.

        Overridden to query the uppercase `INFORMATION_SCHEMA.COLUMNS` view
        required by case-sensitive Fabric environments.
        """
        table = exp.to_table(table_name)
        schema = self._get_schema_name(table_name)
        sql = (
            exp.select("COLUMN_NAME", "DATA_TYPE")
            .from_(f"{self.database}.INFORMATION_SCHEMA.COLUMNS")
            .where(f"TABLE_NAME = '{table.name}'")
            .where(f"TABLE_SCHEMA = '{schema}'")
            .order_by("ORDINAL_POSITION")
        )
        df = self.fetchdf(sql)
        return {
            str(row.COLUMN_NAME): exp.DataType.build(str(row.DATA_TYPE), dialect=self.dialect)
            for row in df.itertuples()
        }

    def _insert_overwrite_by_condition(
        self,
        table_name: TableName,
        source_queries: t.List[SourceQuery],
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        where: t.Optional[exp.Condition] = None,
        insert_overwrite_strategy_override: t.Optional[InsertOverwriteStrategy] = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Implements the insert overwrite strategy for Fabric.

        Overridden to enforce a `DELETE`/`INSERT` strategy, as Fabric's
        `MERGE` statement has limitations.
        """

        columns_to_types = columns_to_types or self.columns(table_name)

        self.delete_from(table_name, where=where or exp.true())

        for source_query in source_queries:
            with source_query as query:
                query = self._order_projections_and_filter(query, columns_to_types)
                self._insert_append_query(
                    table_name,
                    query,
                    columns_to_types=columns_to_types,
                    order_projections=False,
                )

    def _to_sql(self, expression: exp.Expression, quote: bool = True, **kwargs: t.Any) -> str:
        """
        Override SQL generation to fix Fabric-specific issues.

        Specifically, replace unqualified information_schema references with
        database-qualified ones for compatibility with Fabric Warehouse.
        """
        sql = super()._to_sql(expression, quote=quote, **kwargs)

        import re

        # Fix the IF NOT EXISTS pattern to use database-qualified INFORMATION_SCHEMA
        # Only replace if not already database-qualified to avoid double-qualification

        # For information_schema.tables:
        # Match: information_schema.tables (not preceded by a database name)
        # Don't match: db.information_schema.tables or db.INFORMATION_SCHEMA.TABLES
        if self.database and re.search(r"\binformation_schema\.tables\b", sql, flags=re.IGNORECASE):
            # Check if it's already qualified (has database prefix)
            escaped_db = re.escape(self.database)
            if not re.search(
                rf"\b{escaped_db}\.information_schema\.tables\b",
                sql,
                flags=re.IGNORECASE,
            ):
                # Replace unqualified references only
                pattern = r"\binformation_schema\.tables\b"
                replacement = f"{self.database}.INFORMATION_SCHEMA.TABLES"
                sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

        # Same for information_schema.schemata
        if self.database and re.search(
            r"\binformation_schema\.schemata\b", sql, flags=re.IGNORECASE
        ):
            # Check if it's already qualified (has database prefix)
            escaped_db = re.escape(self.database)
            if not re.search(
                rf"\b{escaped_db}\.information_schema\.schemata\b",
                sql,
                flags=re.IGNORECASE,
            ):
                # Replace unqualified references only
                pattern = r"\binformation_schema\.schemata\b"
                replacement = f"{self.database}.INFORMATION_SCHEMA.SCHEMATA"
                sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

        return sql
