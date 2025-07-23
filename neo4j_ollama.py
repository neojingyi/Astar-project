# successful version that responds to MAXEON questions with the correct information
# TO-FIX: does not respond to SPINDEX questions
"""
Vector-based GraphRAG Q&A using Neo4j and Ollama embeddings/LLM.

Requirements:
  pip install neo4j-graphrag[ollama]
  ollama pull nomic-embed-text
  ollama pull llama3.2:latest
  A running Neo4j (5.11+ Enterprise with vector plugin)
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# === Load env and connect ===
load_dotenv()
URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASS", "jingyi1212")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# === Initialize embeddings & retriever ===
# Ensure your Measure nodes have an 'embedding' property
embedder = OllamaEmbeddings(model="nomic-embed-text")
retriever = VectorRetriever(
    driver=driver,
    index_name="index-measure",
    embedder=embedder
)

# === Initialize LLM & RAG ===
llm = OllamaLLM(model_name="llama3.2:latest")
rag = GraphRAG(retriever=retriever, llm=llm)

# === Optional: (Re)embed all Measure nodes ===
def embed_measure_nodes():
    with driver.session() as session:
        result = session.run(
            "MATCH (m:Measure) WHERE m.embedding IS NULL RETURN id(m) AS id, m.`Extracted Sentence` AS sentence"
        )
        for row in result:
            vec = embedder.embed_query(row["sentence"])
            session.run(
                "MATCH (m) WHERE id(m)=$id SET m.embedding=$vec",
                {"id": row["id"], "vec": vec}
            )
    print("✅ All Measure nodes embedded")

# === Q&A Loop ===
def main():
    print("Neo4j GraphRAG (vector) Q&A — type 'exit' to quit")
    # embed_measure_nodes()  # uncomment if embeddings need refreshing
    while True:
        question = input("\nEnter question: ").strip()
        if not question or question.lower() in ('exit','quit'):
            break
        result = rag.search(query_text=question)
        print("\n=== Answer ===")
        print(result.answer)

if __name__ == "__main__":
    embed_measure_nodes() 
    main()
