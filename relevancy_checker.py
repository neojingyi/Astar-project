import os
import time
import re
import requests
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Initialize spaCy sentencizer
nlp = English()
nlp.add_pipe("sentencizer")


def call_groq_llm(prompt: str,
                  model: str = "llama3-8b-8192",
                  temperature: float = 0.0,
                  max_retries: int = 5) -> str:
    """
    Call Groq's chat completion endpoint with retry on rate limiting (429 errors).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment or .env file.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[Retry {attempt}] Rate limited. Waiting {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(f"Groq API failed after {max_retries} retries: {e}")
            wait = 2 ** attempt
            print(f"[Retry {attempt}] Error: {e}. Retrying in {wait}s…")
            time.sleep(wait)

    raise RuntimeError("Groq API call failed after maximum retries.")


def chunk_pdf_with_pages(pdf_path: str, max_chars: int = 1200) -> list[dict]:
    """
    Split PDF into page-sized chunks of up to max_chars characters.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    for pi in range(len(doc)):
        page = doc.load_page(pi)
        text = page.get_text("text")
        start = 0
        while start < len(text):
            snippet = text[start:start + max_chars]
            chunks.append({"page": pi + 1, "text": snippet})
            start += max_chars
    return chunks


def extract_most_relevant_sentence(text_chunk: str, requirement: str) -> str:
    """
    Split a text chunk into sentences and return the one most relevant to the requirement via TF-IDF + cosine similarity.
    """
    doc = nlp(text_chunk)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        return text_chunk
    vect = TfidfVectorizer(stop_words="english")
    docs = sentences + [requirement]
    tfidf = vect.fit_transform(docs)
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    idx = sim.argmax()
    return sentences[idx]


def evaluate_matches(
    matches_excel: str,
    report_pdf: str,
    output_path: str = "match_evaluation.xlsx",
    fuzzy_threshold: int = 85
) -> pd.DataFrame:
    """
    Reads an Excel of matches (columns: disclosure_id, requirement, match_1, match_2, match_3),
    for each row:
      1) compute relevance on match_1;
      2) if relevance <=2, compute on match_2;
      3) if still <=2, compute on match_3;
      keep first with relevance>2 or fallback to match_1;
    then record which pages contain the chosen chunk via fuzzy matching,
    extract the single most relevant sentence from that chunk,
    and write results to Excel, including timing.
    """
    start_time = time.time()

    df = pd.read_excel(matches_excel)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ['disclosure_id', 'requirement', 'match_1', 'match_2', 'match_3']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Input must contain columns: {required}")

    chunks = chunk_pdf_with_pages(report_pdf)

    prompt_template = (
        "You are a strict ESG compliance evaluator.\n\n"
        "Rate relevance on a scale from 1 to 5:\n"
        "1 = Completely unrelated\n"
        "2 = Slightly related\n"
        "3 = Partially relevant\n"
        "4 = Mostly relevant\n"
        "5 = Perfectly relevant\n\n"
        "Respond ONLY with a single number (1–5).\n\n"
        "Disclosure ID: {id}\n"
        "Requirement: {requirement}\n"
        "Sentence: {sentence}\n\n"
        "Rating:"
    )

    results = []
    for _, row in df.iterrows():
        did = row['disclosure_id']
        req = row['requirement']

        # select best match_i by relevance >2
        best_relevance = None
        best_chunk = None
        used_match = 'match_1'
        for col in ['match_1', 'match_2', 'match_3']:
            raw = str(row[col]).strip()
            prompt = prompt_template.format(id=did, requirement=req, sentence=raw)
            try:
                rating_text = call_groq_llm(prompt)
                m = re.search(r"\d+", rating_text)
                relevance = int(m.group()) if m else None
            except Exception as e:
                print(f"Groq API error for {did} with {col}: {e}")
                relevance = None

            if best_relevance is None:
                best_relevance = relevance
                best_chunk = raw

            if relevance is not None and relevance > 2:
                best_relevance = relevance
                best_chunk = raw
                used_match = col
                break

        # pages found via fuzzy match
        chunk_lower = best_chunk.lower()
        pages = [
            c['page']
            for c in chunks
            if fuzz.partial_ratio(chunk_lower, c['text'].lower()) >= fuzzy_threshold
        ]
        pages_str = ",".join(map(str, sorted(set(pages)))) if pages else ""

        # extract most relevant sentence
        extracted_sentence = extract_most_relevant_sentence(best_chunk, req)

        results.append({
            'Disclosure No.': did,
            'Requirement': req,
            'Used Match': used_match,
            'Extracted Sentence': extracted_sentence,
            'Relevance (1-5)': best_relevance,
            'Pages Found': pages_str
        })

        time.sleep(1)

    df_out = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"Total evaluation time: {elapsed:.2f} seconds")
    df_out.to_excel(output_path, index=False)
    print(f"Evaluation saved to {output_path}")
    return df_out


if __name__ == '__main__':
    matches_excel = input("Enter path to matches Excel file: ").strip().strip('"').strip("'")
    report_pdf    = input("Enter path to ESG report PDF:   ").strip().strip('"').strip("'")
    output_xlsx   = input("Output Excel filename [default match_evaluation.xlsx]: ").strip() or 'match_evaluation.xlsx'
    evaluate_matches(matches_excel, report_pdf, output_xlsx)
