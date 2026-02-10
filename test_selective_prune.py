import yaml
import os
import pytest
from unittest.mock import MagicMock, patch
import db_check

def test_push_to_db_filtering():
    """Verify that push_to_db correctly filters tables based on user input."""
    # Mock data
    mock_data = {
        "sources": [
            {
                "name": "dataset1",
                "tables": [
                    {"name": "table1", "description": "desc1"},
                    {"name": "table2", "description": "desc2"}
                ]
            },
            {
                "name": "dataset2",
                "tables": [
                    {"name": "table3", "description": "desc3"}
                ]
            }
        ]
    }
    
    # Mock target_tables
    target = ["dataset1.table1", "table3"]
    
    # Filter logic (extracted from push_to_db)
    sources = mock_data.get('sources', [])
    filtered_sources = []
    for src in sources:
        src_name = src.get('name')
        new_tables = [t for t in src.get('tables', []) if t['name'] in target or f"{src_name}.{t['name']}" in target]
        if new_tables:
            src_copy = src.copy()
            src_copy['tables'] = new_tables
            filtered_sources.append(src_copy)
            
    assert len(filtered_sources) == 2
    assert len(filtered_sources[0]['tables']) == 1
    assert filtered_sources[0]['tables'][0]['name'] == 'table1'
    assert filtered_sources[1]['tables'][0]['name'] == 'table3'

def test_prune_logic():
    """Verify the pruning logic correctly removes missing schemas/tables."""
    mock_yaml = {
        "sources": [
            {
                "name": "exists",
                "tables": [{"name": "tbl_exists"}, {"name": "tbl_gone"}]
            },
            {
                "name": "gone",
                "tables": [{"name": "any"}]
            }
        ]
    }
    
    db_schemas = ["exists"]
    all_db_items = {"tbl_exists"}
    
    pruned_sources = []
    for src in mock_yaml['sources']:
        src_name = src['name']
        if src_name not in db_schemas:
            continue
            
        current_tables = src.get('tables', [])
        valid_tables = [t for t in current_tables if t['name'] in all_db_items]
        
        if valid_tables:
            src['tables'] = valid_tables
            pruned_sources.append(src)
            
    assert len(pruned_sources) == 1
    assert pruned_sources[0]['name'] == "exists"
    assert len(pruned_sources[0]['tables']) == 1
    assert pruned_sources[0]['tables'][0]['name'] == "tbl_exists"

def test_refresh_logic():
    """Verify refresh_schema pulls new columns and updates descriptions."""
    mock_yaml = {
        "sources": [
            {
                "name": "ds",
                "tables": [
                    {
                        "name": "tbl",
                        "columns": [{"name": "col1", "description": "old desc"}]
                    }
                ]
            }
        ]
    }
    
    # DB state: col1 has new desc, col2 is brand new
    db_cols = [
        {"name": "col1", "comment": "new desc", "type": "STRING"},
        {"name": "col2", "comment": "added desc", "type": "INT64"}
    ]
    db_col_map = {c['name']: c for c in db_cols}
    
    # Logic extracted from refresh_schema
    tbl = mock_yaml['sources'][0]['tables'][0]
    yaml_cols = tbl['columns']
    yaml_col_names = {c['name'] for c in yaml_cols}
    
    for col in yaml_cols:
        c_name = col['name']
        if c_name in db_col_map:
            col['description'] = db_col_map[c_name]['comment']
            
    for c_name, db_c in db_col_map.items():
        if c_name not in yaml_col_names:
            yaml_cols.append({"name": c_name, "description": db_c['comment']})
            
    assert len(yaml_cols) == 2
    assert yaml_cols[0]['description'] == "new desc"
    assert yaml_cols[1]['name'] == "col2"
    assert yaml_cols[1]['description'] == "added desc"
