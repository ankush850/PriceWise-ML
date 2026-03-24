import pandas as pd
import os

# Automatically get the folder where this script is located
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to CSV files
test_csv = os.path.join(BASE_PATH, "test.csv")
out_csv = os.path.join(BASE_PATH, "test_out.csv")

# Debug print (optional) - shows exactly where it's looking
print(f"Loading: {test_csv}")
print(f"Loading: {out_csv}")

# Load CSVs
df_test = pd.read_csv(test_csv)
df_out = pd.read_csv(out_csv)

# Basic checks
assert list(df_out.columns) == ['sample_id', 'price'], f"Columns wrong: {df_out.columns}"
assert len(df_out) == len(df_test), f"Row count mismatch: out={len(df_out)} vs test={len(df_test)}"

missing = set(df_test['sample_id']) - set(df_out['sample_id'])
extra = set(df_out['sample_id']) - set(df_test['sample_id'])

assert len(missing) == 0, f"Missing sample_ids: {list(missing)[:10]}"
assert len(extra) == 0, f"Extra sample_ids: {list(extra)[:10]}"
assert df_out['price'].notnull().all(), "Null prices found"
assert (df_out['price'] > 0).all(), "Non-positive prices found"

print("All checks passed ✅")
print(df_out.head().to_string(index=False))
