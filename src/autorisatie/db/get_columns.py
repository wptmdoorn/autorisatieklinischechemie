import sqlite3

def get_columns(db_path: str, table_name: str):
    """Get the column names of a specified table in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Extract column names from the result
        column_names = [column[1] for column in columns]
        
        conn.close()
        return column_names
    except sqlite3.Error as e:
        print(f"Error retrieving columns: {str(e)}")
        return []

if __name__ == "__main__":
    db_path = "data/raw/2013.db"  # Update with your actual database path
    table_name = "labosys"
    columns = get_columns(db_path, table_name)
    print("Columns in the table:", columns) 