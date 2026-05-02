"""
pytest tests with fixtures, parametrization, mocker, monkeypatch, tmp_path.
Demonstrates conftest patterns and test organization.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Generator


# ============================================================================
# Fixtures with different scopes
# ============================================================================

@pytest.fixture(scope="session")
def session_config():
    """Session-scoped fixture: initialized once per test session."""
    return {
        "api_key": "test-key-12345",
        "base_url": "https://api.test.example.com",
        "timeout": 30,
    }


@pytest.fixture(scope="module")
def database():
    """Module-scoped fixture: initialized once per test module."""
    class MockDB:
        def __init__(self):
            self.connected = False
            self.records = []

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

        def insert(self, record):
            self.records.append(record)

        def query(self, table):
            return self.records

    db = MockDB()
    db.connect()
    yield db
    db.disconnect()


@pytest.fixture
def user_data():
    """Function-scoped fixture: initialized for each test."""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "active": True,
    }


@pytest.fixture(params=["json", "csv", "xml"])
def export_format(request):
    """Parametrized fixture: generates multiple fixture instances."""
    return request.param


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture using tmp_path to create temporary files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Create test files
    (data_dir / "input.txt").write_text("test input data\n")
    (data_dir / "config.json").write_text('{"key": "value"}')

    yield data_dir

    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_api_client(mocker):
    """Fixture using mocker to create mock objects."""
    client = mocker.MagicMock()
    client.get.return_value = {"status": 200, "data": []}
    client.post.return_value = {"status": 201, "id": 123}
    client.delete.return_value = {"status": 204}
    return client


@pytest.fixture
def mock_http_calls(mocker):
    """Fixture that mocks external HTTP calls."""
    with patch("requests.get") as mock_get, \
         patch("requests.post") as mock_post:
        mock_get.return_value.json.return_value = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": 999, "status": "created"}
        yield mock_get, mock_post


# ============================================================================
# Test classes and functions
# ============================================================================

class TestUserManagement:
    """Test class demonstrating class-scoped setup."""

    def setup_method(self):
        """Run before each test method."""
        self.test_users = []

    def teardown_method(self):
        """Run after each test method."""
        self.test_users.clear()

    def test_add_user(self, user_data):
        """Basic test with fixture."""
        self.test_users.append(user_data)
        assert len(self.test_users) == 1
        assert self.test_users[0]["username"] == "testuser"

    def test_user_email_validation(self, user_data):
        """Test user data validation."""
        assert "@" in user_data["email"]
        assert user_data["active"] is True

    @pytest.mark.parametrize("field,value", [
        ("username", "alice"),
        ("username", "bob_123"),
        ("email", "alice@example.com"),
        ("email", "bob+test@domain.co.uk"),
    ])
    def test_valid_user_fields(self, field, value, user_data):
        """Parametrized test with multiple inputs."""
        user_data[field] = value
        assert user_data[field] == value


class TestDatabaseOperations:
    """Tests using module-scoped database fixture."""

    def test_insert_record(self, database):
        """Test database insert operation."""
        assert database.connected is True
        database.insert({"name": "Test Record", "value": 42})
        assert len(database.records) == 1

    def test_query_records(self, database):
        """Test database query operation."""
        records = database.query("test_table")
        assert isinstance(records, list)

    @pytest.mark.parametrize("count", [1, 5, 10])
    def test_insert_multiple(self, database, count):
        """Insert multiple records and verify count."""
        for i in range(count):
            database.insert({"id": i, "value": i * 10})
        assert len(database.records) >= count


class TestFileOperations:
    """Tests using tmp_path fixture for file operations."""

    def test_read_temp_file(self, temp_data_dir: Path):
        """Read file from temporary directory."""
        input_file = temp_data_dir / "input.txt"
        assert input_file.exists()
        content = input_file.read_text()
        assert "test input data" in content

    def test_write_new_file(self, temp_data_dir: Path):
        """Write and read file in temporary directory."""
        output_file = temp_data_dir / "output.txt"
        output_file.write_text("Generated output\n")
        assert output_file.exists()
        assert output_file.read_text() == "Generated output\n"

    def test_config_parsing(self, temp_data_dir: Path):
        """Parse JSON config from temporary directory."""
        import json
        config_file = temp_data_dir / "config.json"
        config = json.loads(config_file.read_text())
        assert config["key"] == "value"


class TestMocking:
    """Tests demonstrating mocking patterns."""

    def test_api_client_mock(self, mock_api_client):
        """Test with mocked API client."""
        result = mock_api_client.get("/users")
        assert result["status"] == 200
        mock_api_client.get.assert_called_once_with("/users")

    def test_http_requests_mock(self, mock_http_calls):
        """Test with mocked HTTP requests."""
        import requests
        mock_get, mock_post = mock_http_calls

        users = requests.get("/api/users").json()
        assert len(users["users"]) == 2
        assert users["users"][0]["name"] == "Alice"

        new_user = requests.post("/api/users", json={"name": "Charlie"}).json()
        assert new_user["status"] == "created"
        assert new_user["id"] == 999

    def test_monkeypatch_environment(self, monkeypatch):
        """Test using monkeypatch to modify environment."""
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("DEBUG", "true")

        import os
        assert os.getenv("API_KEY") == "test-key"
        assert os.getenv("DEBUG") == "true"

    def test_monkeypatch_dict(self, monkeypatch):
        """Test using monkeypatch to modify dict items."""
        test_dict = {"key": "original"}
        monkeypatch.setitem(test_dict, "key", "modified")
        assert test_dict["key"] == "modified"


class TestExportFormats:
    """Test with parametrized fixture across multiple tests."""

    def test_export_supported_format(self, export_format):
        """Test exporting to various formats."""
        assert export_format in ["json", "csv", "xml"]
        # Each test runs 3 times: once for each format

    def test_export_file_creation(self, export_format, temp_data_dir: Path):
        """Test that export files are created with correct extension."""
        extensions = {"json": ".json", "csv": ".csv", "xml": ".xml"}
        filename = f"export{extensions[export_format]}"
        filepath = temp_data_dir / filename
        filepath.write_text(f"exported in {export_format} format")
        assert filepath.exists()


class TestWithFixtureRequestParam:
    """Demonstrate fixture.request for indirect parametrization."""

    @pytest.fixture
    def config_file(self, request, tmp_path: Path):
        """Fixture that's parametrized via request.param."""
        config = request.param if hasattr(request, "param") else {}
        filepath = tmp_path / "config.json"
        import json
        filepath.write_text(json.dumps(config))
        return filepath

    @pytest.mark.parametrize("config_file", [
        {"debug": True, "timeout": 10},
        {"debug": False, "timeout": 30},
    ], indirect=True)
    def test_different_configs(self, config_file: Path):
        """Test with different configurations."""
        import json
        config = json.loads(config_file.read_text())
        assert "debug" in config
        assert "timeout" in config


# ============================================================================
# Conftest pattern (typically in conftest.py in real projects)
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Automatically run before all tests."""
    print("\n[Setup] Test environment initialized")
    yield
    print("\n[Teardown] Test environment cleaned up")


def pytest_configure(config):
    """pytest hook called after command line options are parsed."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def session_config():
    """Available to all tests in the session."""
    return {"api_url": "https://api.test.local"}
