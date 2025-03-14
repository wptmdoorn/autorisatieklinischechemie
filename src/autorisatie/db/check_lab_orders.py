import sqlite3
import logging

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

def check_lab_orders_within_one_hour(db_path: str):
    """Check how many lab orders are within 1 hour of each other for each patient."""
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Query to find lab orders within 1 hour for each patient using the existing VerzamelDatetime
        query = """
        SELECT Patientnummer, COUNT(*) AS OrderCount
        FROM (
            SELECT Patientnummer, 
                   VerzamelDatetime,
                   LAG(VerzamelDatetime) OVER (PARTITION BY Patientnummer ORDER BY VerzamelDatetime) AS PreviousOrder
            FROM labosys
        )
        WHERE PreviousOrder IS NOT NULL AND 
              (strftime('%Y-%m-%d %H:%M:%S', VerzamelDatetime) - 
               strftime('%Y-%m-%d %H:%M:%S', PreviousOrder)) < 3600
        GROUP BY Patientnummer
        """
        
        cursor.execute(query)
        results = cursor.fetchall()

        if results:
            print("Lab Orders Within 1 Hour for Each Patient:")
            for patient_id, order_count in results:
                print(f"Patient ID: {patient_id}, Orders within 1 hour: {order_count}")
        else:
            print("No lab orders found within 1 hour for any patient.")

        conn.close()

    except Exception as e:
        logger.error(f"Error checking lab orders: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = "data/clean/2013_sample_50000.db"  # Update with your actual cleaned database path
    check_lab_orders_within_one_hour(db_path) 