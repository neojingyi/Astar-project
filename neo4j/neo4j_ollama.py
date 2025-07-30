
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

load_dotenv()
URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASS", "jingyi1212")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

embedder = OllamaEmbeddings(model="nomic-embed-text")
retriever = VectorRetriever(driver=driver, index_name="index-measure", embedder=embedder)
llm = OllamaLLM(model_name="llama3.2:latest")
rag = GraphRAG(retriever=retriever, llm=llm)

def embed_measure_nodes():
    with driver.session() as session:
        result = session.run(
            "MATCH (m:Measure) WHERE m.embedding IS NULL RETURN id(m) AS id, m.`Extracted Sentence` AS sentence"
        )
        count = 0
        for rec in result:
            vec = embedder.embed_query(rec["sentence"])
            session.run(
                "MATCH (m) WHERE id(m) = $id SET m.embedding = $vec",
                {"id": rec["id"], "vec": vec}
            )
            count += 1
    print(f" Embedded {count} Measure nodes.")

def graph_rag_qa(question: str, company: str = None, top_k: int = 5):
    """
    Semantic retrieval of top_k Measure sentences, filtered to the specified company,
    followed by grounded LLM answer.
    """
    base_text = question
    if company:
        base_text = f"{company}: {question}"

    result = rag.search(
        query_text=base_text,
        retriever_config={"top_k": top_k}
    )

    hits = []
    rr = getattr(result, 'retriever_result', None)
    if rr and getattr(rr, 'search_results', None):
        hits = rr.search_results
        if company:
            company_lower = company.lower()
            hits = [h for h in hits if h.node.get("Company", "").lower() == company_lower]

    if hits:
        print("\nRetrieved Context Sentences:")
        for hit in hits:
            sent = hit.node.get('Extracted Sentence', '').replace('\n', ' ')
            print(f"{hit.score:.4f} → {sent}")
    else:
        print("\n⚠️ No context retrieved for this company.")

    print("\n=== Answer ===")
    print(result.answer)

if __name__ == "__main__":
    
    print("Neo4j Company-scoped GraphRAG Q&A — type 'exit' to quit")
    while True:
        company = input("\nEnter company name (blank for all): ").strip() or None
        question = input("Enter your question: ").strip()
        if not question or question.lower() in ('exit', 'quit'):
            break
        graph_rag_qa(question, company=company)

    driver.close()
