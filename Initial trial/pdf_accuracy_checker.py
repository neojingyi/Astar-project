import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

# --- Utility Functions ---

def normalize_text(s: str) -> str:
    """
    Lowercase, remove non-alphanumeric, collapse whitespace.
    """
    return re.sub(r'\s+', ' ', re.sub(r'[^0-9a-zA-Z]+', ' ', s).lower()).strip()

def extract_sentences(ans: str) -> list[str]:
    """
    If there are quoted snippets, use those.
    Otherwise split on '.', '?', '!' and return non-empty pieces.
    """
    quotes = re.findall(r'"([^"]+)"', ans)
    if quotes:
        return [q.strip() for q in quotes if q.strip()]
    parts = re.split(r'(?<=[.?!])\s+', ans)
    return [p.strip() for p in parts if p.strip()]

# --- 1. Load Excel answers ---

EXCEL_PATH = '/Users/jingyi/Desktop/ACADS/ASTAR/Astar/Outputs/extracted_disclosure_answers_1.xlsx'
answers_df = pd.read_excel(EXCEL_PATH)
answers_df['Disclosure No.'] = answers_df['Disclosure No.'].astype(str).str.strip()
answers_df['Answer'] = answers_df['Answer'].fillna('')

# --- 2. Load PDF and extract per-page text ---

PDF_PATH = '/Users/jingyi/Desktop/ACADS/ASTAR/Astar/ESG reports/2023-Boeing-Sustainability-Report.pdf'
doc = fitz.open(PDF_PATH)
page_texts = [page.get_text("text") for page in doc]
# Also pre-normalize page texts
norm_pages = [normalize_text(t) for t in page_texts]

# --- 3. Verify each answer by searching snippets on each page ---

verified_pages_list = []
all_verified = []

for _, row in answers_df.iterrows():
    ans = row['Answer']
    snippets = extract_sentences(ans)
    pages_for_snippet = []
    snippet_found = []

    for snippet in snippets:
        norm_snip = normalize_text(snippet)
        found = []
        for pg_num, pg_text in enumerate(norm_pages, start=1):
            if norm_snip and norm_snip in pg_text:
                found.append(pg_num)
        pages_for_snippet.append(found)
        snippet_found.append(bool(found))

    # If no snippets at all, consider it unverified
    verified = all(snippet_found) and len(snippets) > 0
    # Record flattened unique pages
    flat_pages = sorted({p for sub in pages_for_snippet for p in sub})
    verified_pages_list.append(flat_pages)
    all_verified.append(verified)

answers_df['Pages Found'] = verified_pages_list
answers_df['Verified']    = all_verified

# --- 4. Summary and save ---

total = len(answers_df)
verified_count = sum(all_verified)
print(f"Verified {verified_count}/{total} ({verified_count/total*100:.1f}%)")

OUTPUT = 'extracted_answers_verification.xlsx'
answers_df.to_excel(OUTPUT, index=False)
print(f"Results written to {OUTPUT}")
