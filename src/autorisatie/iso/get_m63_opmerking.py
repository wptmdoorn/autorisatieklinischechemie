import sqlite3
import pandas as pd

def get_m63_info(db_paths: list[str], table_name: str):
    """Get the column names of a specified table in the database."""
    try:
        all_data = []
        
        # Loop through each database path
        for db_path in db_paths:
            try:
                conn = sqlite3.connect(db_path)
                conn.text_factory = lambda data: str(data, encoding="latin1")
                cursor = conn.cursor()
                
                cursor.execute(f"""SELECT test, resultaat, Opmerkingrapport 
                               FROM {table_name}
                               """)
                
                print(f'Processing database: {db_path}')
                data = cursor.fetchall()
                
                # keep rows with 'iso' or 'm063' in any of these columns
                filtered_data = [row for row in data if 
                        any('accreditatie' in str(col).lower() or
                            '15189' in str(col).lower() or
                            'm063' in str(col).lower() or
                            'mo63' in str(col).lower()
                            for col in row)]

                # additionele filters
                filtered_data = [row for row in filtered_data if
                        'sanquin' not in str(row[0]).lower() and
                        'm069' not in str(row[1]).lower() and
                        'm168' not in str(row[1]).lower()]
                
                all_data.extend(filtered_data)
                conn.close()
                
            except sqlite3.Error as e:
                print(f"Error processing database {db_path}: {str(e)}")
                continue
        
        # Create a DataFrame from all collected data (with test and resultaat columns

        df = pd.DataFrame(all_data, columns=['test', 'resultaat', 'Opmerkingrapport'])

        # remove opmerkingrapport column
        df = df.drop(columns=['Opmerkingrapport'], errors='ignore')

        # Count unique combinations and add a new column 'n'
        df['n'] = df.fillna('NA').groupby(['test', 'resultaat'])['test'].transform('size')
        
        # Drop duplicates to keep only unique combinations with their counts
        df = df.drop_duplicates()

        # Arrange by 'n'
        df = df.sort_values(by='n', ascending=False)
        
        # Write to xlsx
        df.to_excel('out/m63_opmerking_all.xlsx', index=False)
        
        return df
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    db_paths = ["data/raw/2023.db", "data/raw/2024.db"]  # List all your database paths
    table_name = "labosys"
    result_df = get_m63_info(db_paths, table_name) 