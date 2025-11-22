"""Script to download and prepare the 20 Newsgroups dataset."""
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
import os


def download_and_save_dataset():
    """Download 20 Newsgroups dataset and save as .txt files."""
    print("Downloading 20 Newsgroups dataset...")
    
    # Fetch dataset
    dataset = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        shuffle=False
    )
    
    # Create output directory
    output_dir = Path("data/docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each document as a .txt file
    print(f"Saving {len(dataset.data)} documents to {output_dir}...")
    
    for i, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        # Get category name
        category = dataset.target_names[target]
        
        # Create filename: doc_001_category.txt
        filename = f"doc_{i+1:03d}_{category.replace('.', '_')}.txt"
        filepath = output_dir / filename
        
        # Save text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"\n✓ Successfully saved {len(dataset.data)} documents to {output_dir}")
    print(f"  Categories: {len(dataset.target_names)}")
    print(f"  Sample categories: {', '.join(dataset.target_names[:5])}...")
    print("\nYou can now run: python main.py")


if __name__ == "__main__":
    download_and_save_dataset()


