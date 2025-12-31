import pytest
from unittest.mock import MagicMock
import datetime
import db_check  # Import your script

# ==========================================
# 🧪 TEST HELPER FUNCTIONS
# ==========================================

def test_get_bq_quoted_name():
    """Verify BigQuery table names are properly quoted with backticks."""
    # Scenario 1: Table only
    assert db_check.get_bq_quoted_name(None, "mytable") == "`mytable`"
    
    # Scenario 2: Project + Dataset + Table
    assert db_check.get_bq_quoted_name("my-project.dataset", "table") == "`my-project`.`dataset`.`table`"
    
    # Scenario 3: Dataset + Table
    assert db_check.get_bq_quoted_name("dataset", "table") == "`dataset`.`table`"

def test_clean_ai_response():
    """Verify AI markdown cleaning works correctly."""
    # Scenario 1: Indented header (common AI response format)
    raw_text = """
    # Description
    This is the description.
    """
    assert db_check.clean_ai_response(raw_text) == "This is the description."
    
    # Scenario 2: Inline bold label
    assert db_check.clean_ai_response("**Column:** id") == "id"
    
    # Scenario 3: Mixed content
    assert db_check.clean_ai_response("**Info** Some value") == "Some value"
    
    # Scenario 4: Clean text remains unchanged
    assert db_check.clean_ai_response("Clean text") == "Clean text"

# ==========================================
# 🧪 TEST CHECK LOGIC (Nulls, Freshness, Unique)
# ==========================================

@pytest.fixture
def mock_connection():
    """Creates a fake database connection object."""
    return MagicMock()

def test_check_freshness_success(mock_connection):
    """Test a timestamp that is recent (< 24h)."""
    # Setup: Return a time 1 hour ago
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    mock_connection.execute.return_value.scalar.return_value = one_hour_ago
    
    result = db_check.check_freshness(mock_connection, "schema", "table", "ts_col", dtype="TIMESTAMP")
    
    assert result is not None
    assert "1.0h ago" in result[0]
    assert result[1] == "✅"

def test_check_freshness_old(mock_connection):
    """Test a timestamp that is old (> 24h)."""
    # Setup: Return a time 48 hours ago
    two_days_ago = datetime.datetime.now() - datetime.timedelta(hours=48)
    mock_connection.execute.return_value.scalar.return_value = two_days_ago
    
    result = db_check.check_freshness(mock_connection, "schema", "table", "ts_col", dtype="TIMESTAMP")
    
    assert result is not None
    assert "48.0h ago" in result[0]
    assert result[1] == "⚠️ OLD"

def test_check_freshness_pure_date(mock_connection):
    """Test that pure 'datetime.date' objects don't crash the script."""
    # Setup: Return a date object (no time info)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    mock_connection.execute.return_value.scalar.return_value = yesterday
    
    # Should automatically convert to datetime at midnight and calculate hours
    result = db_check.check_freshness(mock_connection, "schema", "table", "date_col", dtype="DATE")
    
    assert result is not None
    assert "ago" in result[0]

def test_check_freshness_bad_format_string(mock_connection):
    """Test handling of weird string formats."""
    # Setup: Return a string that isn't a standard ISO date
    mock_connection.execute.return_value.scalar.return_value = "Not A Date"
    
    # FIX: Use a column name that looks like a date (e.g., 'created_at')
    # If we used 'bad_col', the script would skip it entirely and return None.
    result = db_check.check_freshness(mock_connection, "schema", "table", "created_at", dtype="STRING")
    
    # Should catch the error gracefully and warn
    assert result is not None
    assert "Bad fmt" in result[0]
    assert result[1] == "⚠️"

def test_check_nulls(mock_connection):
    """Test Null check logic."""
    mock_connection.execute.return_value.scalar.return_value = 0
    assert db_check.check_nulls(mock_connection, "s", "t", "c") == ("PASS", "✅")
    
    mock_connection.execute.return_value.scalar.return_value = 5
    res = db_check.check_nulls(mock_connection, "s", "t", "c")
    assert "FAIL (5)" in res[0]
    assert res[1] == "❌"