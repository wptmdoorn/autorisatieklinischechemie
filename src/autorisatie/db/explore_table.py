import sqlite3
from typing import List, Dict

def get_tables(db_path: str) -> List[str]:
    """Get list of all tables in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query sqlite_master table for all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    conn.close()
    return tables

def get_table_info(db_path: str, table_name: str) -> List[Dict]:
    """Get detailed information about columns in a specific table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column information
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    # Format column information
    column_info = []
    for col in columns:
        column_info.append({
            'cid': col[0],          # Column ID
            'name': col[1],         # Column name
            'type': col[2],         # Data type
            'notnull': col[3],      # Not null constraint
            'default': col[4],      # Default value
            'pk': col[5]            # Primary key
        })
    
    conn.close()
    return column_info

def get_sample_rows(db_path: str, table_name: str, n: int = 5) -> List[Dict]:
    """Get n random sample rows from the specified table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column names first
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Get random rows
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {n};")
    rows = cursor.fetchall()
    
    # Format into list of dictionaries
    formatted_rows = []
    for row in rows:
        formatted_rows.append(dict(zip(columns, row)))
    
    conn.close()
    return formatted_rows

def explore_database(db_path: str) -> None:
    """Explore and print information about all tables in the database."""
    tables = get_tables(db_path)
    
    print("Database Tables:")
    print("-" * 50)
    
    for table in tables:
        print(f"\nTable: {table}")
        print("Columns:")
        columns = get_table_info(db_path, table)
        
        for col in columns:
            pk_indicator = "(Primary Key)" if col['pk'] else ""
            null_indicator = "NOT NULL" if col['notnull'] else "NULL allowed"
            print(f"  - {col['name']}: {col['type']} {null_indicator} {pk_indicator}")
            if col['default']:
                print(f"    Default: {col['default']}")
        
        print("\nSample Rows:")
        print("-" * 30)
        sample_rows = get_sample_rows(db_path, table)
        for row in sample_rows:
            print(row)
        print()

explore_database("data/clean/2013_sample_5000.db")
