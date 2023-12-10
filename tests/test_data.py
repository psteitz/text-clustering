import context
from data import csv_to_hf
from datasets import load_from_disk
csv_to_hf.convert_csv_to_hf("test_load.csv", "Apple", "test_data")
ds = load_from_disk("test_data")
assert ds["text"][0] == "apple1"
assert ds.num_rows == 4
csv_to_hf.convert_csv_to_hf("test_load.csv", "Duck", "test_data")
ds = load_from_disk("test_data")
assert ds["text"][0] is None
assert ds["text"][1] == "duck2"
assert ds.num_rows == 4
print(ds)
