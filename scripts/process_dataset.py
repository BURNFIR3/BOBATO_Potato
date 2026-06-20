"""scripts/process_dataset.py – Step 1: Process raw_data/ CSVs"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processor import process_raw_dataset

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Processing raw dataset from raw_data/")
    print("=" * 60)
    df = process_raw_dataset()
    if df is not None:
        print(f"\n[OK] Done! {len(df):,} records processed.")
    else:
        print("\n[FAIL] Processing failed.")
