import pandas as pd
from tabulate import tabulate

def analyze_csv(file_path):
    """
    Analyze a CSV file and print a summary table.

    Args:
        file_path (str): Path to the CSV file.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Initialize summary table
    summary_table = []

    # Iterate over columns
    for column in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Calculate basic statistics
            mean = df[column].mean()
            median = df[column].median()
            min_val = df[column].min()
            max_val = df[column].max()

            # Append statistics to summary table
            summary_table.append([column, "Numeric", f"{mean:.2f}", f"{median:.2f}", f"{min_val:.2f}", f"{max_val:.2f}"])
        else:
            # List unique values for categorical columns
            unique_values = df[column].nunique()
            summary_table.append([column, "Categorical", f"{unique_values} unique values", "", "", ""])

    # Print summary table
    print(tabulate(summary_table, headers=["Column", "Type", "Mean", "Median", "Min", "Max"], tablefmt="grid"))

def main():
    file_path = input("Enter the CSV file path: ")
    analyze_csv(file_path)

if __name__ == "__main__":
    main()