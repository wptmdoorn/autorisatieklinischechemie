import sqlite3

def check_schema(db_path: str):
    """Check the schema of the labosys table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # print all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(cursor.fetchall())
        
        cursor.execute("PRAGMA table_info(labosys)")
        columns = cursor.fetchall()
        
        print("Current schema of labosys table:")
        for col in columns:
            print(f"Column: {col[1]}, Type: {col[2]}")
        
        conn.close()
    except Exception as e:
        print(f"Error checking schema: {str(e)}")

if __name__ == "__main__":
    db_path = "data/raw/2023.db"  # Update with your actual database path
    check_schema(db_path)