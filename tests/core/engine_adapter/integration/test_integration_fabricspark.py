# type: ignore
import typing as t
import pytest
import time
from unittest.mock import patch

from sqlmesh.core.engine_adapter.fabricspark import FabricSparkEngineAdapter
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


def test_token_authentication_validation(
    ctx: TestContext, engine_adapter: FabricSparkEngineAdapter
):
    """Test that Azure access token authentication is properly validated."""
    connection = engine_adapter.connection
    livy_session = connection.livy_session

    # Verify access token exists and is valid
    livy_session._ensure_token()
    assert livy_session.access_token is not None
    assert livy_session.access_token.token is not None
    assert len(livy_session.access_token.token) > 0
    assert livy_session.access_token.expires_on > time.time()

    # Test that authentication headers are properly formatted
    headers = livy_session._headers()
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Bearer ")
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"


def test_session_establishment_and_retrieval(
    ctx: TestContext, engine_adapter: FabricSparkEngineAdapter
):
    """Test Livy session establishment and retrieval."""
    connection = engine_adapter.connection
    livy_session = connection.livy_session

    # Test session creation/retrieval
    session_id = livy_session.create_session()
    assert session_id is not None
    assert isinstance(session_id, str)
    assert len(session_id) > 0

    # Verify session is properly stored
    assert livy_session.session_id == session_id

    # Test that we can execute statements through the session
    result = livy_session.execute_statement("SELECT 1 as test_session")
    assert result is not None
    assert isinstance(result, dict)

    # Test that session is reused for subsequent calls
    second_session_id = livy_session.create_session()
    assert second_session_id == session_id


def test_session_token_refresh_renewal(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test automatic token refresh and renewal mechanism."""
    connection = engine_adapter.connection
    livy_session = connection.livy_session

    # Get initial token
    livy_session._ensure_token()
    initial_token = livy_session.access_token.token
    initial_expires = livy_session.access_token.expires_on

    # Mock token expiration to test refresh logic
    with patch(
        "sqlmesh.core.engine_adapter.fabricspark.is_token_refresh_necessary"
    ) as mock_refresh_check:
        # First call - token is valid (no refresh needed)
        mock_refresh_check.return_value = False
        livy_session._ensure_token()
        assert livy_session.access_token.token == initial_token

        # Second call - token needs refresh
        mock_refresh_check.return_value = True

        # Mock the token refresh based on authentication method
        if livy_session.credentials.authentication == "az_cli":
            with patch(
                "sqlmesh.core.engine_adapter.fabricspark.get_cli_access_token"
            ) as mock_get_token:
                from azure.core.credentials import AccessToken

                mock_token = AccessToken(
                    token="refreshed_token_123", expires_on=int(time.time()) + 3600
                )
                mock_get_token.return_value = mock_token

                livy_session._ensure_token()
                assert livy_session.access_token.token == "refreshed_token_123"
                assert livy_session.access_token.expires_on > initial_expires

        elif livy_session.credentials.authentication == "service_principal":
            with patch(
                "sqlmesh.core.engine_adapter.fabricspark.get_sp_access_token"
            ) as mock_get_token:
                from azure.core.credentials import AccessToken

                mock_token = AccessToken(
                    token="refreshed_sp_token_456", expires_on=int(time.time()) + 3600
                )
                mock_get_token.return_value = mock_token

                livy_session._ensure_token()
                assert livy_session.access_token.token == "refreshed_sp_token_456"
                assert livy_session.access_token.expires_on > initial_expires


def test_authentication_error_handling(ctx: TestContext, engine_adapter: FabricSparkEngineAdapter):
    """Test proper handling of authentication errors."""
    connection = engine_adapter.connection
    livy_session = connection.livy_session

    # Test unsupported authentication method
    original_auth = livy_session.credentials.authentication
    try:
        livy_session.credentials.authentication = "unsupported_method"
        livy_session.access_token = None  # Force token refresh

        with pytest.raises(ValueError, match="Unsupported authentication method"):
            livy_session._ensure_token()
    finally:
        # Restore original authentication method
        livy_session.credentials.authentication = original_auth


def test_azure_authentication_context_verification(
    ctx: TestContext, engine_adapter: FabricSparkEngineAdapter
):
    """Test that Azure authentication context is properly established."""
    # Verify connection was established successfully (authentication worked)
    result = engine_adapter.fetchone("SELECT 1 as auth_test")
    assert result is not None
    assert result[0] == 1

    # Test workspace/lakehouse context is accessible
    result = engine_adapter.fetchone("SELECT current_database()")
    assert result is not None
    assert result[0] is not None

    # Test user context is properly set
    result = engine_adapter.fetchone("SELECT current_user()")
    assert result is not None
    assert result[0] is not None
