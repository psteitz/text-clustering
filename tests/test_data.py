"""
Test data loading from csv to hf dataset.
"""
import context
from data import csv_to_hf
from datasets import load_from_disk

# Test loading csv to hf dataset
csv_to_hf.convert_csv_to_hf("test_load.csv", "Apple", "test_data")
ds = load_from_disk("test_data")
assert ds["text"][0] == "apple1"
assert ds.num_rows == 4
csv_to_hf.convert_csv_to_hf("test_load.csv", "Duck", "test_data")
ds = load_from_disk("test_data")
assert ds["text"][0] == "duck2"
assert ds["text"][1] == "duck3"
assert ds.num_rows == 3
assert ds.num_columns == 1
print("Test loading csv to hf dataset passed.")

# Test getting column names from csv
column_names = csv_to_hf.get_column_names("test_load.csv")
assert column_names == ["Apple", "Duck", "Dog", "Pig Pen"]
print("Test getting column names from csv passed.")
