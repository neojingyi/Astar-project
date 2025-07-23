import os
import sys
import re
import time
import json
import fitz  # PyMuPDF
import pandas as pd
from pdfminer.high_level import extract_text
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Suppress HuggingFace tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === 1. Chunk PDF by page (with metadata) ===
def chunk_pdf_with_pages(pdf_path: str, max_chars_per_chunk: int = 1200) -> list[dict]:
    """
    Splits each page of the PDF into chunks of up to max_chars_per_chunk characters.
    Returns a list of dicts: {"page": page_number, "text": chunk_text}.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        full_text = page.get_text("text")
        start = 0
        while start < len(full_text):
            snippet = full_text[start : start + max_chars_per_chunk]
            chunks.append({"page": page_index + 1, "text": snippet})
            start += max_chars_per_chunk
    return chunks

# === 2. Load disclosure prompts ===
def load_disclosures(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.dropna(subset=["Prompt"])
    return df[["Disclosure No.", "Disclosure Title", "Requirement", "Prompt"]]

# === 3. Build FAISS index over page chunks ===
def build_faiss_index(chunks: list[dict], embedding_model_name: str) -> FAISS:
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store

# === 4. Safe invocation to handle Groq rate limits ===
def safe_invoke(runnable, inputs: dict):
    try:
        return runnable.invoke(inputs)
    except Exception as e:
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg:
            wait = 2
            print(f"[RATE LIMIT] Waiting {wait}s then retrying...")
            time.sleep(wait)
            return runnable.invoke(inputs)
        raise

# === 5. Extract a single disclosure with expanded context and permissive “quote-or-summarize” prompt ===
def extract_single_prompt(row: dict, vector_store: FAISS, llm) -> dict:
    disclosure_no = row["Disclosure No."]
    prompt_text   = row["Prompt"]

    try:
        # Retrieve top 5 similar page-chunks
        docs = vector_store.similarity_search(prompt_text, k=5)

        # Combine context from these chunks, track unique pages
        combined_context = ""
        pages = []
        for doc in docs:
            pg = doc.metadata["page"]
            if pg not in pages:
                pages.append(pg)
            combined_context += f"\n\n-- From Page {pg} --\n" + doc.page_content

        page_list_str = ", ".join(str(p) for p in sorted(pages))

        # Prompt asks: quote exact sentences if present, otherwise provide a concise paraphrase
        prompt_template = PromptTemplate(
            input_variables=["id", "prompt", "context", "pages"],
            template=(
                "You are a factual ESG‐report parsing assistant. Below is the exact text from pages {pages} of an ESG report. "
                "First try to FIND and RETURN the exact sentence(s) from this text that answer the question, enclosed in double quotes. "
                "If no exact sentence exists, PROVIDE a concise PARAPHRASED summary of the relevant information, again in double quotes, "
                "and indicate “PARAPHRASE” before the quoted text. "
                "Number each sentence accordingly to track number of sentences generated."
                "If the answer cannot be found or inferred, respond exactly “NOT FOUND IN CONTEXT.”\n\n"
                "Instruction ({id}): {prompt}\n\n"
                "Context (pages {pages}):\n"
                "{context}\n\n"
                "===\n"
                "Provide your answer below. Then, at the end, write “SOURCE PAGES: {pages}”."
                "An example of the right formatted output is:"
                "1. 'This report includes the operations of The Boeing Company and its subsidiaries.' SOURCE PAGES: 79"
                "2. 'This report includes the operations of The Boeing Company and its subsidiaries.' SOURCE PAGES: 80"
                ""
            )
        )

        # Invoke Groq
        runnable = prompt_template | llm
        result = safe_invoke(runnable, {
            "id": disclosure_no,
            "prompt": prompt_text,
            "context": combined_context,
            "pages": page_list_str
        })
        answer = result.content.strip() if hasattr(result, "content") else str(result).strip()

        # Handle “NOT FOUND IN CONTEXT”
        if "NOT FOUND IN CONTEXT" in answer:
            return {
                "Disclosure No.": disclosure_no,
                "Title": row["Disclosure Title"],
                "Requirement": row["Requirement"],
                "Answer": "NOT FOUND IN CONTEXT"
            }

        # Extract anything in double quotes (either exact or paraphrase)
        quoted_sentences = re.findall(r'"([^"]+)"', answer)
        if not quoted_sentences:
            # If no quotes found, return the raw answer as fallback
            return {
                "Disclosure No.": disclosure_no,
                "Title": row["Disclosure Title"],
                "Requirement": row["Requirement"],
                "Answer": answer
            }

        # Check each quoted snippet against the combined context (substring)
        norm_context = re.sub(r"\s+", " ", combined_context)
        verified = []
        for sent in quoted_sentences:
            norm_sent = re.sub(r"\s+", " ", sent.strip())
            if norm_sent in norm_context or answer.startswith("PARAPHRASE"):
                verified.append(sent)

        # Build final answer listing verified quotes (or paraphrase)
        if verified:
            final_answer = " ".join(f'"{q}"' for q in verified) + f" SOURCE PAGES: {page_list_str}"
        else:
            # If none verified, trust paraphrase or return answer without validation
            final_answer = answer.rstrip(f"SOURCE PAGES: {page_list_str}")

        return {
            "Disclosure No.": disclosure_no,
            "Title": row["Disclosure Title"],
            "Requirement": row["Requirement"],
            "Answer": final_answer
        }

    except Exception as e:
        print(f"[ERROR] Disclosure {disclosure_no} : {e}")
        return {
            "Disclosure No.": disclosure_no,
            "Title": row["Disclosure Title"],
            "Requirement": row["Requirement"],
            "Answer": ""
        }

# === 6. Sequential extraction for all disclosures ===
def extract_all_disclosures(df_prompts: pd.DataFrame, vector_store: FAISS, llm) -> pd.DataFrame:
    results = []
    for _, row in df_prompts.iterrows():
        result = extract_single_prompt(row.to_dict(), vector_store, llm)
        results.append(result)
        # Reduced pause to speed up (still avoid TPM spikes)
        time.sleep(0.1)
    return pd.DataFrame(results)

# === Main pipeline ===
def main():
    pdf_path = input("Enter ESG report path:").strip().strip("'").strip('"')
    tic = time.perf_counter()
    xlsx_path = '/Users/jingyi/Desktop/ACADS/ASTAR/Astar/Disclosures/GRI Disclosure List 2-promptOutput2.xlsx'
    output_path = "extracted_disclosure_answers_with_pages.xlsx"

    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

    # 1. Load disclosure prompts
    df_prompts = load_disclosures(xlsx_path)

    # 2. Chunk PDF by page and build FAISS index
    chunks = chunk_pdf_with_pages(pdf_path, max_chars_per_chunk=1200)
    vector_store = build_faiss_index(chunks, embedding_model_name)

    # 3. Initialize Groq LLM with temperature=0 (deterministic)
    load_dotenv()
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

    # 4. Extract all disclosures with expanded context & paraphrase fallback
    results_df = extract_all_disclosures(df_prompts, vector_store, llm)

    # 5. Save to Excel
    results_df.to_excel(output_path, index=False)
    toc = time.perf_counter()
    print(f"Extraction complete in {toc - tic} seconds. Results saved to {output_path}")

if __name__ == "__main__":
    main()
