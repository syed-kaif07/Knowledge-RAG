from pathlib import Path
from src.llamaparse_ingestion import parse_pdf


DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("parsed_docs/llamaparse")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDFs")


for i, pdf_path in enumerate(pdf_files, start=1):
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"

    if output_path.exists():
        print(f"[{i}/{len(pdf_files)}] SKIP: {pdf_path.name}")
        continue

    try:
        print(f"[{i}/{len(pdf_files)}] Parsing: {pdf_path.name}")

        markdown = parse_pdf(str(pdf_path))

        output_path.write_text(
            markdown,
            encoding="utf-8"
        )

        print(f"    Saved: {output_path}")

    except Exception as e:
        print(f"    FAILED: {pdf_path.name}")
        print(f"    Error: {e}")


print("\nBatch parsing complete.")
print(f"Output directory: {OUTPUT_DIR}")