
"""
Text2Cypher-based GraphRAG Q&A using Neo4j and Ollama LLM.

This script:
 1) Uses Ollama to translate natural‐language questions into Cypher queries.
 2) Executes the generated Cypher against Neo4j.
 3) Feeds the raw result records back into Ollama to produce a final answer.

Requirements:
  pip install neo4j-graphrag[ollama] python-dotenv
  ollama pull llama3.2:latest
  Neo4j Enterprise 5.11+ running locally (bolt://localhost:7687)
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG

load_dotenv()
NEO4J_URI    = os.getenv("NEO4J_URI",    "bolt://localhost:7687")
NEO4J_USER   = os.getenv("NEO4J_USER",   "neo4j")
NEO4J_PASS   = os.getenv("NEO4J_PASS",   "jingyi1212")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

llm = OllamaLLM(model_name=OLLAMA_MODEL)

schema_hint = (
    "(c:Company)-[:PROVIDED_BY]->(m:Measure), "
    "(d:Disclosure)-[:HAS_REQUIREMENT]->"
    "(r:Requirement)-[:ADDRESSED_BY]->(m:Measure)"
)

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=schema_hint
)

rag = GraphRAG(retriever=retriever, llm=llm)

def main():
    print("Neo4j Text2Cypher GraphRAG Q&A with Ollama — type 'exit' to quit")
    while True:
        question = input("\nEnter question: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        # 1) Generate & run Cypher, then get final answer
        result = rag.search(query_text=question)

        # 2) Print the generated Cypher query
        print("\n=== Generated Cypher Query ===")
        print(result.retriever_result.cypher or "<no query generated>")

        # 3) Print the raw records returned by Neo4j
        print("\n=== Raw Records ===")
        for record in result.retriever_result.raw_records:
            print(record)

        # 4) Print the final LLM answer
        print("\n=== Answer ===")
        print(result.answer)

    driver.close()

if __name__ == "__main__":
    main()
