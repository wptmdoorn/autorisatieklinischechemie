import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def connect_db(db_path: str) -> sqlite3.Connection:
    """Create a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def analyze_correlation(db_path: str):
    """Analyze the correlation between the top 10 parameters from wide_blood_draws."""
    try:
        conn = connect_db(db_path)

        # Step 1: Identify the 10 most prevalent parameters
        prevalence_query = """
        SELECT Testcode, COUNT(*) AS TestCount
        FROM labosys
        GROUP BY Testcode
        ORDER BY TestCount DESC
        LIMIT 10
        """
        prevalent_tests = pd.read_sql_query(prevalence_query, conn)
        top_tests = prevalent_tests['Testcode'].tolist()
        conn.close()

        print(top_tests)

        # Step 2: Prepare the query to get data for the top 10 parameters
        test_code_conditions = ', '.join([f'"{test}"' for test in top_tests])
        print(test_code_conditions)

        # Step 3: Retrieve data for the top 10 parameters from wide_blood_draws
        # Select the relevant columns (top-10 tests)

        conn = connect_db(db_path)
        query = f"""
        SELECT {test_code_conditions}
        FROM wide_blood_draws
        """
        
        # Load data into a DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Check if we have data
        if df.empty:
            print("No data found for the selected tests.")
            return

        # Step 4: Calculate the correlation matrix
        df = df.apply(pd.to_numeric, errors='coerce')
        correlation_matrix = df[["KRESE", "KALSE",  "NATSE", "URESE", "LEUC", "TRC"]].corr()

        # Step 5: Visualize the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Top 10 Parameters from wide_blood_draws')
        plt.show()

        # Step 6: Provide a combination plot for each correlation
        top_tests = ["KRESE", "KALSE",  "NATSE", "URESE", "LEUC", "TRC"]
        for test in zip(top_tests, top_tests[1:]):
            #plt.figure(figsize=(12, 6))
            # Create joint plot combining scatter and density
            g = sns.JointGrid(data=df, x=test[0], y=test[1])
            g.plot_joint(sns.scatterplot, alpha=0.5)
            g.plot_marginals(sns.kdeplot, fill=True)
            g.fig.suptitle(f'{test[0]} vs {test[1]}')
            # Set x and y axis limits to cover 99% of data
            x_data = df[test[0]].dropna()
            y_data = df[test[1]].dropna()
            x_1p, x_99p = np.percentile(x_data, [1, 99])
            y_1p, y_99p = np.percentile(y_data, [1, 99])
            plt.xlim(x_1p, x_99p)
            plt.ylim(y_1p, y_99p)
            plt.show()


    except Exception as e:
        print(f"Error analyzing correlation: {str(e)}")
        raise

if __name__ == "__main__":
    db_path = "data/clean/2013_sample_500000.db"  # Update with your actual cleaned database path
    analyze_correlation(db_path) 