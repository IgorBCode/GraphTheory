import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("distances_matrix.csv", index_col=0)

# Initialize a flag to check if the matrix is symmetric
is_symmetric = True

# Check if the bottom half matches the upper half
for i in range(1, len(df)):  # Start from 1 to avoid diagonal
    for j in range(i):  # Only check the lower triangle (where i > j)
        if df.iat[i, j] != df.iat[j, i]:  # Compare with the mirrored cell
            row_name = df.index[i]
            col_name = df.columns[j]
            print(f"Mismatch at ({row_name}, {col_name}): {df.iat[i, j]} != {df.iat[j, i]}")
            is_symmetric = False

# Final result
if is_symmetric:
    print("The bottom half of the adjacency matrix matches the upper half (symmetric).")
else:
    print("The bottom half of the adjacency matrix does not match the upper half.")
