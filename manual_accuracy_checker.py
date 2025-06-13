import re
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Utility Functions ===

def clean_answer(ans: str) -> str:
    """Strip off 'SOURCE PAGES' suffix and normalize whitespace."""
    ans = re.sub(r'SOURCE PAGES?:.*$', '', ans, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', ans).strip()

def extract_sentences(ans: str) -> list[str]:
    """Return quoted snippets if present, else split into sentences."""
    quotes = re.findall(r'"([^"]+)"', ans)
    if quotes:
        return [q.strip() for q in quotes if q.strip()]
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', ans) if s.strip()]

def label_row(ans: str, manual_texts: list[str], fuzzy: int, semantic: float) -> str:
    """
    Label reliability:
      - If manual_texts is all empty and ans == "NOT FOUND IN CONTEXT", treat as Exact.
      - Otherwise do the verbatim snippet check, then thresholds.
    """
    cleaned = clean_answer(ans).strip().upper()

    # === SPECIAL CASE ===
    # No manual text at all, and answer says NOT FOUND IN CONTEXT
    if all(not txt.strip() for txt in manual_texts) and cleaned == "NOT FOUND IN CONTEXT":
        return 'Exact'

    # === ORIGINAL LOGIC ===
    cleaned_lower = cleaned.lower()
    for txt in manual_texts:
        norm_txt = re.sub(r'\s+', ' ', txt).strip().lower()
        for snippet in extract_sentences(cleaned_lower):
            if snippet and snippet in norm_txt:
                return 'Exact'
    if fuzzy >= 80 or semantic >= 0.75:
        return 'High'
    if fuzzy >= 60 or semantic >= 0.50:
        return 'Moderate'
    if fuzzy >= 30 or semantic >= 0.30:
        return 'Low'
    return 'Unreliable'



# === 1. Load & Normalize Disclosure Keys ===

EXTRACTED_PATH = '/Users/jingyi/Desktop/ACADS/ASTAR/Astar/Outputs/extracted_disclosure_answers_1.xlsx'
REFERENCE_PATH = '/Users/jingyi/Desktop/ACADS/ASTAR/Boeing-Accuracy-Validation-main 2/Boeing relevant GRI disclosures_06062024_merged_Ginny review.xlsx'

# Extracted answers
extracted_df = pd.read_excel(EXTRACTED_PATH)
extracted_df['Disclosure No.'] = extracted_df['Disclosure No.'].astype(str).str.strip()
extracted_df = extracted_df.drop_duplicates(subset=['Disclosure No.'], keep='first')

# Reference file
ref_df = pd.read_excel(REFERENCE_PATH)
# Rename if needed
if 'Disclosure' in ref_df.columns and 'Disclosure No.' not in ref_df.columns:
    ref_df = ref_df.rename(columns={'Disclosure': 'Disclosure No.'})
ref_df['Disclosure No.'] = ref_df['Disclosure No.'].astype(str).str.strip()

# === 2. Detect & Aggregate Manual-Extraction Columns ===

# Exclude metadata columns
exclude = {'Disclosure No.', 'Title', 'Requirement'}
# Find columns with any non-blank content beyond the key
manual_cols = [
    c for c in ref_df.columns
    if c not in exclude and ref_df[c].astype(str).str.strip().any()
]
if not manual_cols:
    raise ValueError(f"No manual columns found; saw: {ref_df.columns.tolist()}")

# Aggregate per Disclosure No.
ref_small = ref_df[['Disclosure No.'] + manual_cols].fillna('')
def concat_manual(series: pd.Series) -> str:
    return " ".join(str(s).strip() for s in series if str(s).strip())
ref_grouped = (
    ref_small
    .groupby('Disclosure No.')[manual_cols]
    .agg(concat_manual)
    .reset_index()
)
# Combine into one Reference Text
ref_grouped['Reference Text'] = (
    ref_grouped[manual_cols]
    .agg(' '.join, axis=1)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)
ref_grouped = ref_grouped[['Disclosure No.', 'Reference Text']]

# === 3. Merge Extracted + Reference ===

df = pd.merge(
    extracted_df,
    ref_grouped,
    on='Disclosure No.',
    how='left'
)
df['Answer'] = df['Answer'].fillna('')
df['Reference Text'] = df['Reference Text'].fillna('')

# === 4. Prepare TF-IDF Vectorizer ===

corpus = list(df['Answer']) + list(df['Reference Text'])
vectorizer = TfidfVectorizer().fit(corpus)
ans_vecs = vectorizer.transform(df['Answer'])

# === 5. Compute Similarity Scores ===

best_fuzzy = []
best_semantic = []
manual_texts_list = []

for i, row in df.iterrows():
    ans = row['Answer']
    ans_vec = ans_vecs[i]
    ref = row['Reference Text']

    # Fuzzy & semantic
    fz = fuzz.token_set_ratio(ans, ref)
    ref_vec = vectorizer.transform([ref])
    ss = cosine_similarity(ans_vec, ref_vec)[0][0]

    best_fuzzy.append(fz)
    best_semantic.append(ss)
    manual_texts_list.append([ref])

df['Best Fuzzy']    = best_fuzzy
df['Best Semantic'] = best_semantic
df['_manual_texts'] = manual_texts_list

# === 6. Label Reliability ===
df['Reliability'] = df.apply(
    lambda r: label_row(r['Answer'], r['_manual_texts'], r['Best Fuzzy'], r['Best Semantic']),
    axis=1
)

# === 6b. Clear out “NOT FOUND IN CONTEXT” entries ===
df.loc[df['Answer'].str.strip() == 'NOT FOUND IN CONTEXT', 'Reliability'] = ""

# === 7. Diagnostics for Unreliable Cases ===

print("\nSample Unreliable Cases:")
for _, row in df[df['Reliability']=='Unreliable'].head(5).iterrows():
    print(f"Disclosure {row['Disclosure No.']}")
    print(" Answer:    ", row['Answer'])
    print(" Reference: ", row['Reference Text'][:200], "…")
    print(" Scores:    ", f"Fuzzy={row['Best Fuzzy']}, Sem={row['Best Semantic']:.2f}")
    print("---")

# === 8. Summary & Save ===

counts = df['Reliability'].value_counts()
total = len(df)
accuracy = (counts.get('Exact',0) + counts.get('High',0)) / total * 100

print("\nReliability distribution:")
print(counts.to_string())
print(f"\nOverall Exact+High accuracy: {accuracy:.2f}%")

# Cleanup and save
df = df.drop(columns=['_manual_texts'])
df.to_excel('disclosure_accuracy_detailed.xlsx', index=False)
print("\nResults saved to 'disclosure_accuracy_detailed.xlsx'")
