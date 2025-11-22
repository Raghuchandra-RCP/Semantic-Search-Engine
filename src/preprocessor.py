from pathlib import Path
from typing import List, Dict
from .utils import clean_text, compute_hash


class DocumentPreprocessor:
    
    def __init__(self, docs_folder: str):
        self.docs_folder = Path(docs_folder)
        if not self.docs_folder.exists():
            raise ValueError(f"Documents folder not found: {docs_folder}")
    
    def load_documents(self) -> List[Dict]:
        documents = []
        txt_files = list(self.docs_folder.glob("*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.docs_folder}")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                
                cleaned_text = clean_text(raw_text)
                
                if not cleaned_text or len(cleaned_text.strip()) < 50:
                    continue
                
                text_hash = compute_hash(cleaned_text)
                doc_id = txt_file.stem
                
                document = {
                    "doc_id": doc_id,
                    "filename": txt_file.name,
                    "text": cleaned_text,
                    "hash": text_hash,
                    "length": len(cleaned_text),
                    "filepath": str(txt_file)
                }
                
                documents.append(document)
                
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
                continue
        
        print(f"Loaded {len(documents)} documents from {self.docs_folder}")
        return documents
