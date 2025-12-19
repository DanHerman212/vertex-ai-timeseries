import pandas as pd

# Load the test data
df = pd.read_csv('local_test_data/test.csv')

print("--- General Statistics for 'y' (Minutes Between Trains) ---")
print(df['y'].describe())

print("\n--- Top 10 Highest Values ---")
print(df['y'].nlargest(10))

print("\n--- Top 10 Lowest Values ---")
print(df['y'].nsmallest(10))

# Check for potential data quality issues
print("\n--- Data Quality Checks ---")
print(f"Negative values: {(df['y'] < 0).sum()}")
print(f"Zero values: {(df['y'] == 0).sum()}")
print(f"NaN values: {df['y'].isna().sum()}")
