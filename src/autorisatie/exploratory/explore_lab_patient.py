import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def connect_db(db_path: str) -> sqlite3.Connection:
    """Create a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def plot_tests_over_time(db_path: str):
    """Plot measurements for 5 patients with 5 different tests over time."""
    try:
        conn = connect_db(db_path)

        # Step 1: Identify the 5 most prevalent tests
        prevalence_query = """
        SELECT Testcode, COUNT(*) AS TestCount
        FROM labosys
        GROUP BY Testcode
        ORDER BY TestCount DESC
        LIMIT 5
        """
        prevalent_tests = pd.read_sql_query(prevalence_query, conn)
        conn.close()

        # Step 2: Prepare the query to get data for 5 patients for these tests
        test_codes = prevalent_tests['Testcode'].tolist()
        test_code_conditions = ', '.join([f'"{test}"' for test in test_codes])

        # Step 3: Retrieve data for 5 patients
        conn = connect_db(db_path)
        query = f"""
        SELECT Patientnummer, VerzamelDatum, VerzamelTijd, Testcode, resultaat
        FROM labosys
        WHERE Testcode IN ({test_code_conditions}) AND Patientnummer IN (
            SELECT DISTINCT Patientnummer FROM labosys LIMIT 5
        )
        """
        
        # Load data into a DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Check if we have data
        if df.empty:
            print("No data found for the selected tests and patients.")
            return

        # Combine date and time into a single datetime column
        df['VerzamelDatetime'] = pd.to_datetime(df['VerzamelDatum'].astype(str) + ' ' + df['VerzamelTijd'])

        # Melt the DataFrame to long format for easier plotting
        df_melted = df[['Patientnummer', 'VerzamelDatetime', 'Testcode', 'resultaat']]

        # Drop rows with NaN results
        df_melted = df_melted.dropna(subset=['resultaat'])

        # Create a line plot for the tests over time
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=df_melted, x='VerzamelDatetime', y='resultaat', hue='Patientnummer', style='Testcode', markers=True)
        plt.title('Measurements for 5 Patients with 5 Different Tests Over Time')
        plt.xlabel('Date and Time')
        plt.ylabel('Resultaat')
        plt.xticks(rotation=45)
        plt.legend(title='Patientnummer and Testcode')
        plt.tight_layout()
        plt.show()

        # Output the prevalent tests and their counts
        print("5 Most Prevalent Tests:")
        print(prevalent_tests)

    except Exception as e:
        print(f"Error plotting measured tests: {str(e)}")
        raise

if __name__ == "__main__":
    db_path = "data/clean/2013_sample_50000.db"  # Update with your actual cleaned database path
    plot_tests_over_time(db_path)
