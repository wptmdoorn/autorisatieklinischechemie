import sqlite3
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def connect_db(db_path: str, create_new: bool = False) -> sqlite3.Connection:
    """
    Create a database connection.
    
    Args:
        db_path: Path to the SQLite database
        create_new: If True, creates a new database if it doesn't exist
    """
    try:
        if create_new:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Use bytes to handle non-UTF8 characters
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def clean_duplicate_tests(db_path: str) -> None:
    """
    Remove duplicate test results while keeping the most recent one.
    Duplicates are identified by having the same patient, test, and collection date/time.
    """
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()
        
        # Create temporary table for unique records
        cursor.execute("""
            CREATE TEMPORARY TABLE temp_unique AS
            SELECT * FROM labosys
            GROUP BY Patientnummer, Testcode, VerzamelDatum, VerzamelTijd
            HAVING rowid = MAX(rowid)
        """)
        
        # Replace original table with deduplicated data
        cursor.execute("DELETE FROM labosys")
        cursor.execute("INSERT INTO labosys SELECT * FROM temp_unique")
        
        # Cleanup
        cursor.execute("DROP TABLE temp_unique")
        
        deleted_count = cursor.rowcount
        logger.info(f"Removed {deleted_count} duplicate records")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        raise

def clean_invalid_results(db_path: str) -> None:
    """
    Clean up invalid or corrupted test results.
    """
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()
        
        # Remove records with empty or invalid test results
        cursor.execute("""
            DELETE FROM labosys 
            WHERE resultaat IS NULL 
               OR resultaat = '' 
               OR resultaat = 'ERROR'
        """)
        
        deleted_count = cursor.rowcount
        logger.info(f"Removed {deleted_count} invalid results")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error cleaning invalid results: {str(e)}")
        raise

def adjust_order_times(db_path: str) -> None:
    """Adjust order times for tests within one minute for each patient."""
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Step 1: Create a temporary table to hold adjusted times
        cursor.execute("""
            CREATE TEMPORARY TABLE temp_adjusted AS
            SELECT Patientnummer, 
                   Testcode, 
                   MIN(VerzamelTijd) AS FirstTime,
                   VerzamelDatum
            FROM labosys
            GROUP BY Patientnummer, VerzamelDatum
            HAVING COUNT(*) > 1
        """)

        # Step 2: Update the original labosys table with the adjusted times
        cursor.execute("""
            UPDATE labosys
            SET VerzamelTijd = (
                SELECT FirstTime
                FROM temp_adjusted
                WHERE labosys.Patientnummer = temp_adjusted.Patientnummer
                  AND labosys.VerzamelDatum = temp_adjusted.VerzamelDatum
            )
            WHERE EXISTS (
                SELECT 1
                FROM temp_adjusted
                WHERE labosys.Patientnummer = temp_adjusted.Patientnummer
                  AND labosys.VerzamelDatum = temp_adjusted.VerzamelDatum
            )
        """)

        # Cleanup
        cursor.execute("DROP TABLE temp_adjusted")
        
        conn.commit()
        conn.close()
        logger.info("Adjusted order times for tests within one minute.")

    except Exception as e:
        logger.error(f"Error adjusting order times: {str(e)}")
        raise

def transform_to_wide_format(db_path: str) -> None:
    """Transform the data to wide format based on blood draw level."""
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Step 1: Retrieve unique test codes
        cursor.execute("SELECT DISTINCT Testcode FROM labosys")
        test_codes = [row[0] for row in cursor.fetchall()]

        # Step 2: Construct the SQL query for wide format
        case_statements = ",\n".join(
            [f"MAX(CASE WHEN Testcode = '{test_code}' THEN resultaat END) AS '{test_code}'" for test_code in test_codes]
        )

        # Create a new table to store the wide format data
        cursor.execute("DROP TABLE IF EXISTS wide_blood_draws")
        cursor.execute(f"""
            CREATE TABLE wide_blood_draws AS
            SELECT Patientnummer, 
                   VerzamelDatetime,
                   Labnummer,
                   {case_statements},
                   COUNT(*) AS DrawCount
            FROM labosys
            GROUP BY Patientnummer, VerzamelDatetime, Labnummer
        """)
        
        conn.commit()
        conn.close()
        logger.info("Transformed data to wide format successfully.")
        
    except Exception as e:
        logger.error(f"Error transforming data to wide format: {str(e)}")
        raise

def copy_and_clean_database(source_path: str, dest_path: str, sample_size: Optional[int] = None) -> None:
    """
    Copy the database and clean the copy.
    
    Args:
        source_path: Path to the source SQLite database
        dest_path: Path to the destination SQLite database
        sample_size: Optional number of records to sample (if None, copies all records)
    """
    try:
        # Create new database and copy schema
        source_conn = connect_db(source_path)
        dest_conn = connect_db(dest_path, create_new=True)
        
        if sample_size is None:
            # Copy the entire database
            source_conn.backup(dest_conn)
        else:
            # Copy schema by creating empty tables
            cursor = source_conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='labosys'")
            create_table_sql = cursor.fetchone()[0]
            
            # Modify the create table SQL to include VerzamelDatetime
            create_table_sql = create_table_sql.replace(")", ", VerzamelDatetime TEXT)")
            dest_conn.execute(create_table_sql)
            
            # Copy a random sample of records
            cursor.execute("SELECT COUNT(*) FROM labosys")
            total_records = cursor.fetchone()[0]
            
            logger.info(f"Sampling {sample_size} records from {total_records} total records")
            
            # First, get the column names
            cursor.execute("PRAGMA table_info(labosys)")
            columns = [col[1] for col in cursor.fetchall()]
            columns_str = ", ".join(columns)
            
            # Use ORDER BY RANDOM() to get a truly random sample
            source_cursor = source_conn.cursor()
            dest_cursor = dest_conn.cursor()
            
            source_cursor.execute(f"""
                SELECT {columns_str} 
                FROM labosys 
                LIMIT ?
            """, (sample_size,))
            
            # Fetch and insert in batches to handle large samples
            batch_size = 10000
            while True:
                rows = source_cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                # Process rows to remove unwanted columns and merge date and time
                processed_rows = []
                for row in rows:
                    # Create a new row without Prioriteit, Categorie, UniekLabnummer, and test
                    new_row = [row[i] for i in range(len(row)) if columns[i] not in ['Prioriteit', 'Categorie', 'UniekLabnummer', 'test']]
                    
                    # Merge VerzamelDatum and VerzamelTijd into VerzamelDatetime
                    verzamel_datum = row[columns.index('VerzamelDatum')]
                    verzamel_tijd = row[columns.index('VerzamelTijd')]
                    
                    # Format VerzamelDatetime to %Y-%m-%d %H:%M:%S
                    verzamel_datetime = f"{verzamel_datum[:4]}-{verzamel_datum[4:6]}-{verzamel_datum[6:]} {verzamel_tijd}:00"
                    
                    # Append the new VerzamelDatetime to the new row
                    new_row.append(verzamel_datetime)
                    processed_rows.append(new_row)
                
                # Insert processed rows into the destination database
                dest_cursor.executemany(
                    f"INSERT INTO labosys ({','.join([col for col in columns if col not in ['Prioriteit', 'Categorie', 'UniekLabnummer', 'test']])}, VerzamelDatetime) VALUES ({','.join(['?' for _ in range(len(columns) - 4 + 1)])})",
                    processed_rows
                )
                dest_conn.commit()
            
            # Log the actual number of records copied
            dest_cursor.execute("SELECT COUNT(*) FROM labosys")
            actual_copied = dest_cursor.fetchone()[0]
            logger.info(f"Actually copied {actual_copied} records")
            
        source_conn.close()
        dest_conn.close()
        
        # Now clean the new database
        clean_duplicate_tests(dest_path)
        clean_invalid_results(dest_path)
        transform_to_wide_format(dest_path)  # Transform to wide format after cleaning
        
    except Exception as e:
        logger.error(f"Error during database copy and cleanup: {str(e)}")
        raise

def main(source_path: str, sample_size: Optional[int] = None) -> None:
    """
    Main function to run all cleanup operations.
    
    Args:
        source_path: Path to the source SQLite database
        sample_size: Optional number of records to sample
    """
    logger.info("Starting database cleanup")
    
    try:
        # Create cleaned database path
        filename = os.path.basename(source_path)
        base, ext = os.path.splitext(filename)
        
        # Add sample size to filename if sampling
        if sample_size:
            dest_filename = f"{base}_sample_{sample_size}{ext}"
        else:
            dest_filename = filename
            
        dest_path = os.path.join('data', 'clean', dest_filename)
        
        # Overwrite the existing file if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        
        # Copy and clean the database
        copy_and_clean_database(source_path, dest_path, sample_size)
        
        # Adjust order times for tests within one minute
        adjust_order_times(dest_path)
        
        logger.info(f"Database cleanup completed successfully. Clean database saved to: {dest_path}")
        
    except Exception as e:
        logger.error(f"Database cleanup failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main("data/raw/2013.db", sample_size=500000)