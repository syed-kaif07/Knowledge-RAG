# index_llamaparse.py
import sys
import time
from src.markdown_ingestion import ingest_llamaparse_pipeline, get_new_llamaparse_files

def main():
    print("=== LlamaParse Markdown Indexing Pipeline ===")
    new_files = get_new_llamaparse_files()
    if not new_files:
        print("All LlamaParse markdown documents are already indexed.")
        return

    print(f"Discovered {len(new_files)} unindexed markdown document(s). Starting indexing...")
    start_time = time.time()
    
    vs, bm25, chunk_count = ingest_llamaparse_pipeline()
    
    elapsed = time.time() - start_time
    print(f"=== Indexing Completed Successfully in {elapsed:.2f}s ===")
    print(f"Total Chunks: {chunk_count}")

if __name__ == "__main__":
    main()
