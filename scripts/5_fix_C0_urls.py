"""
Add URLs to C0 files for proper alignment with C1
"""

import pandas as pd

print("="*60)
print("ADDING URLs TO C0 FILES")
print("="*60)

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    
    # Load C0 (currently: label, C0)
    c0_df = pd.read_csv(f"outputs/stage0/{split}_with_C0.csv")
    
    # Load original CSV (has: url, label)
    orig_df = pd.read_csv(f"data/processed/{split}.csv")
    
    # Take first N URLs (where N = number of C0 samples)
    # ImageFolder processed them in order
    n = len(c0_df)
    urls = orig_df['url'].values[:n]
    
    # Add URL column at the beginning
    c0_df.insert(0, 'url', urls)
    
    # Save back
    c0_df.to_csv(f"outputs/stage0/{split}_with_C0.csv", index=False)
    
    print(f"  ✅ Added {len(urls):,} URLs")
    print(f"  Columns now: {c0_df.columns.tolist()}")

print("\n" + "="*60)
print("✅ DONE! C0 files now have URLs")
print("="*60)
