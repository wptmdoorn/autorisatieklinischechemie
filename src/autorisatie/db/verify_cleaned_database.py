import sqlite3
import logging
import random

logger = logging.getLogger(__name__)

def connect_db(db_path: str) -> sqlite3.Connection:
    """Create a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def verify_database(db_path: str):
    """Verify the cleaned database."""
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Check for required columns
        required_columns = ['Patientnummer', 'Testcode', 'VerzamelDatum', 'VerzamelTijd', 'VerzamelDatetime', 'resultaat', 'test']
        cursor.execute("PRAGMA table_info(labosys)")
        columns = [col[1] for col in cursor.fetchall()]

        missing_columns = [col for col in required_columns if col not in columns]
        if missing_columns:
            print(f"Missing columns: {', '.join(missing_columns)}")
        else:
            print("All required columns are present.")

        # Count total records
        cursor.execute("SELECT COUNT(*) FROM labosys")
        total_records = cursor.fetchone()[0]
        print(f"Total records in the cleaned database: {total_records}")

        # Count unique VerzamelDatetime values
        cursor.execute("SELECT COUNT(DISTINCT VerzamelDatetime) FROM labosys")
        unique_verzamel_datetime = cursor.fetchone()[0]
        print(f"Unique VerzamelDatetime values: {unique_verzamel_datetime}")

        # Count records with null or empty values in critical columns
        cursor.execute("SELECT COUNT(*) FROM labosys WHERE resultaat IS NULL OR resultaat = ''")
        invalid_resultaat_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM labosys WHERE test IS NULL OR test = ''")
        invalid_test_count = cursor.fetchone()[0]

        print(f"Records with null or empty 'resultaat': {invalid_resultaat_count}")
        print(f"Records with null or empty 'test': {invalid_test_count}")

        # Display a few random rows for inspection
        print("\nSample of processed rows:")
        sample_size = 5  # Number of rows to display
        cursor.execute("SELECT * FROM labosys")
        rows = cursor.fetchall()
        
        # Randomly select rows to display
        if len(rows) > sample_size:
            sampled_rows = random.sample(rows, sample_size)
        else:
            sampled_rows = rows  # If fewer rows than sample size, take all

        # Print the sampled rows in a user-friendly format
        for row in sampled_rows:
            print(row)

        conn.close()

    except Exception as e:
        logger.error(f"Error verifying database: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = "data/clean/2013_sample_50000.db"  # Update with your actual cleaned database path
    verify_database(db_path) 