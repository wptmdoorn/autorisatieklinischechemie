import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from sklearn.metrics import r2_score
import numpy as np
import os

def connect_db(db_path: str) -> sqlite3.Connection:
    """Create a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def find_top_coexisting_pairs(db_path: str, top_n: int = 20):
    """Find the top N co-existing test pairs."""
    conn = connect_db(db_path)
    
    # Step 1: Retrieve data from wide_blood_draws
    query = "SELECT * FROM wide_blood_draws"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Step 2: Find co-existing pairs
    test_columns = list(df.columns[3:])  # Assuming first three columns are Patientnummer, VerzamelDatetime, Labnummer
    test_columns.remove("ARCHSE")
    test_columns.remove("DrawCount")
    coexisting_counts = {}

    for test1, test2 in combinations(test_columns, 2):
        # Count co-occurrences
        co_count = df[[test1, test2]].dropna().shape[0]
        if co_count > 0:
            coexisting_counts[(test1, test2)] = co_count

    # Step 3: Get the top N pairs
    top_pairs = sorted(coexisting_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_pairs, df

def calculate_r_squared(df, test1, test2):
    """Calculate the R-squared value for two tests."""
    # Drop NaN values for the two tests
    valid_data = df[[test1, test2]].dropna()
    if valid_data.shape[0] < 2:  # Need at least two points to calculate R-squared
        return None
    x = valid_data[test1].values
    y = valid_data[test2].values
    return r2_score(x, y)

def create_scatter_plots(db_path: str, top_pairs, df):
    """Create scatter plots for the top co-existing pairs and save to PDF."""
    pdf_path = "out/correlation/correlation_scatter_plots.pdf"

    df = df.apply(pd.to_numeric, errors='coerce')

    with PdfPages(pdf_path) as pdf:
        for (test1, test2), _ in top_pairs:
            r_squared = calculate_r_squared(df, test1, test2)
            if r_squared is not None:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=df, x=test1, y=test2)
                plt.title(f'Scatter Plot of {test1} vs {test2}\nR-squared: {r_squared:.2f}')
                plt.xlabel(test1)
                plt.ylabel(test2)
                plt.grid()
                pdf.savefig()  # Save the current figure into the PDF
                plt.close()

    print(f"Scatter plots saved to {pdf_path}")

if __name__ == "__main__":
    db_path = "data/clean/2013_sample_5000.db"  # Update with your actual cleaned database path
    top_pairs, df = find_top_coexisting_pairs(db_path, top_n=20)
    print(top_pairs)
    create_scatter_plots(db_path, top_pairs, df) 