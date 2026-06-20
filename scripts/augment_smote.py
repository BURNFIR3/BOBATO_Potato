"""scripts/augment_smote.py – Step 2: SMOTE augmentation"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smote_augmentation import augment_with_smote

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: SMOTE Augmentation")
    print("=" * 60)
    df = augment_with_smote()
    if df is not None:
        print(f"\n✅ Done! Augmented dataset: {len(df):,} records.")
    else:
        print("\n❌ Augmentation failed.")
