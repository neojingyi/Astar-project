import os
import sys
import re
import time
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process as fuzzy_process

# Initialize spaCy sentencizer only
nlp = English()
nlp.add_pipe("sentencizer")


def extract_sentences_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from PDF and split into complete sentences using spaCy.
    Only return sentences ending with punctuation to avoid table fragments.
    Falls back to all spaCy sentences if too few proper sentences.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    text = "\n\n".join(pages).replace("\x0c", "\n")
    doc_sp = nlp(text)
    raw_sents = [s.text.strip() for s in doc_sp.sents]
    # Filter to sentences ending in .?!
    full_sents = [s for s in raw_sents if len(s) > 30 and re.search(r'[\.\?!]$', s)]
    # If enough full sentences, use them, else fallback
    if len(full_sents) >= max(100, len(raw_sents) // 2):
        return full_sents
    return [s for s in raw_sents if len(s) > 30]


def extract_full_sentence(fragment: str, query: str) -> str:
    """
    Given a fragment (usually a sub-sentence), find the nearest complete sentence within it.
    Ensures result ends with punctuation.
    """
    doc_sp = nlp(fragment)
    candidates = [s.text.strip() for s in doc_sp.sents if s.text.strip()]
    if not candidates:
        sent = fragment.strip()
    else:
        # Choose longest candidate as best approximation
        sent = max(candidates, key=len)
    # Ensure punctuation at end
    if not re.search(r'[\.\?!]$', sent):
        sent = sent.rstrip('.') + '.'
    return sent


def process_pdf_and_match(
    disclosures_file: str,
    pdf_path: str,
    emb_model_path: str = "./models/esg-finetuned",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
    top_m: int = 50,
    output_excel: str = "disclosure_matches.xlsx"
) -> pd.DataFrame:
    """
    Robust pipeline that now returns full sentences only, and if 'sub' column
    is present and non-empty, appends it to the requirement before matching.
    """
    start = time.perf_counter()

    # 1. Load disclosures
    if disclosures_file.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(disclosures_file)
    else:
        df = pd.read_csv(disclosures_file)
    # normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    id_col  = next((c for c in df.columns if 'disclosure' in c and ('no' in c or 'id' in c)), None)
    req_col = next((c for c in df.columns if 'requirement' in c), None)
    if not id_col or not req_col:
        raise ValueError("Required columns 'disclosure_id' and 'requirement' not found.")
    sub_col = 'sub' if 'sub' in df.columns else None

    # 2. Extract and filter sentences once
    sentences = extract_sentences_from_pdf(pdf_path)
    if not sentences:
        raise ValueError("No valid sentences extracted.")
    print(f"Using {len(sentences)} sentences for matching.")

    # 3. BM25 sparse retrieval setup
    tokenized = [
        [tok.text.lower() for tok in nlp(sent) if not tok.is_punct and not tok.is_space]
        for sent in sentences
    ]
    bm25 = BM25Okapi(tokenized)

    # 4. Dense SBERT semantic search setup
    bi_model    = SentenceTransformer(emb_model_path)
    emb_sent    = bi_model.encode(sentences,    convert_to_tensor=True, normalize_embeddings=True)
    # titles will be built per-row
    reranker    = CrossEncoder(reranker_model)

    # 5. TF-IDF fallback
    tfidf       = TfidfVectorizer(stop_words='english').fit(sentences)
    sent_tfidf  = tfidf.transform(sentences)

    results = []
    for _, row in df.iterrows():
        did = str(row[id_col]).strip()
        # build requirement, appending 'sub' if present and non-empty
        req = str(row[req_col]).strip()
        if sub_col and pd.notna(row[sub_col]) and str(row[sub_col]).strip():
            req = f"{req} {str(row[sub_col]).strip()}"

        # 1) BM25 retrieval
        q_tok = [tok.text.lower() for tok in nlp(req) if not tok.is_punct and not tok.is_space]
        bm25_scores = bm25.get_scores(q_tok)
        bm25_idx    = np.argsort(-bm25_scores)[:top_m]

        # 2) Dense SBERT retrieval
        emb_req    = bi_model.encode([req], convert_to_tensor=True, normalize_embeddings=True)
        dense_hits = util.semantic_search(emb_req, emb_sent, top_k=top_m)[0]
        dense_idx  = [hit['corpus_id'] for hit in dense_hits]

        # 3) Combine candidate indices
        cand_ids = list(dict.fromkeys(list(bm25_idx) + dense_idx))[:top_m]
        cands    = [sentences[j] for j in cand_ids]

        # 4) Cross-encoder rerank candidates
        rerank_scores = reranker.predict([(req, c) for c in cands], batch_size=16)
        ranked         = sorted(zip(cands, rerank_scores), key=lambda x: -x[1])[:top_k]

        # 5) TF-IDF fallback if not enough
        if len(ranked) < top_k:
            q_vec     = tfidf.transform([req])
            tf_scores = cosine_similarity(q_vec, sent_tfidf)[0]
            for j in np.argsort(-tf_scores):
                if len(ranked) >= top_k: break
                s = sentences[j]
                if s not in [r[0] for r in ranked]:
                    ranked.append((s, float(tf_scores[j])))

        # 6) Fuzzy fallback if still short
        if len(ranked) < top_k:
            for match, score in fuzzy_process.extract(req, sentences, limit=top_k):
                if len(ranked) >= top_k: break
                if match not in [r[0] for r in ranked]:
                    ranked.append((match, score/100.0))

        # 7) Ensure full sentences via extract_full_sentence
        row_res = {
            'Disclosure No.': did,
            'Requirement':    req
        }
        for idx, (frag, sc) in enumerate(ranked[:top_k], start=1):
            full_sent = extract_full_sentence(frag, req)
            row_res[f'match_{idx}'] = full_sent
            row_res[f'score_{idx}'] = sc

        results.append(row_res)
        time.sleep(1)

    out_df = pd.DataFrame(results)
    # Sanitize control chars
    illegal = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
    out_df = out_df.applymap(lambda v: illegal.sub('', v) if isinstance(v, str) else v)

    out_df.to_excel(output_excel, index=False)
    elapsed = time.perf_counter() - start
    print(f"Saved to '{output_excel}' in {elapsed:.2f}s")
    return out_df


if __name__ == '__main__':
    disclosures_file = input("Path to disclosures CSV/XLSX: ").strip().strip('"').strip("'")
    pdf_path         = input("Path to ESG report PDF:    ").strip().strip('"').strip("'")
    top_k            = int(input("Top-k matches [default 5]: ") or 5)
    output_excel     = input("Output Excel filename [default disclosure_matches.xlsx]: ").strip() or 'disclosure_matches.xlsx'
    process_pdf_and_match(
        disclosures_file,
        pdf_path,
        emb_model_path='./models/esg-finetuned',
        reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k=top_k,
        top_m=50,
        output_excel=output_excel
    )
