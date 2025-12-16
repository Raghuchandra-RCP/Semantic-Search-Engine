from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
import os
from typing import Optional, Callable

def download_and_save_dataset(progress_callback: Optional[Callable[[int, int], None]] = None):
    print("Downloading 20 Newsgroups dataset...")
    

    dataset = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        shuffle=False
    )
    

    output_dir = Path("data/docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    

    print(f"Saving {len(dataset.data)} documents to {output_dir}...")
    
    total = len(dataset.data)
    for i, (text, target) in enumerate(zip(dataset.data, dataset.target)):

        category = dataset.target_names[target]
        

        filename = f"doc_{i+1:03d}_{category.replace('.', '_')}.txt"
        filepath = output_dir / filename
        

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        

        if progress_callback and ((i + 1) % 100 == 0 or i == total - 1):
            progress_callback(i + 1, total)
    
    print(f"\n✓ Successfully saved {len(dataset.data)} documents to {output_dir}")
    print(f"  Categories: {len(dataset.target_names)}")
    print(f"  Sample categories: {', '.join(dataset.target_names[:5])}...")
    print("\nYou can now run: python main.py")
    
    return True

if __name__ == "__main__":
    download_and_save_dataset()

