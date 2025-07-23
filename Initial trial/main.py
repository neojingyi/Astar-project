import os
import sys
import shutil
import tempfile
import fitz
from docx import Document
from pdf_to_text import convert_pdfs_to_text
from text_to_index import create_faiss_index
from text_to_docx import text_to_docx
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
TAXONOMY_PATH = 'metrics_taxonomy.json'      # JSON file mapping canonical metrics

NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = 'neo4j'
NEO4J_PWD = 'password'

# Load FAISS index
def load_faiss_index(index_path, embedding_model):
    # Load the FAISS index using the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# Create the RAG system using FAISS and Ollama (Llama 3.1)
def create_analysis_rag_system(index_path, embedding_model='sentence-transformers/all-mpnet-base-v2', model_name="llama3.2"):
    # Load the FAISS index
    vector_store = load_faiss_index(index_path, embedding_model)

    # Initialize the Ollama model (Llama3.1)
    llm = OllamaLLM(model=model_name)

    # Create a more detailed prompt template
    question_prompt = PromptTemplate(
        input_variables=["context","question"],
        template = """
    Provide a numbered list of any sentences or phrases relating to process to determine material topics, 
    specifically to describe the process it has followed to determine its material topics, including,  
    how it has identified actual and potential, negative and positive impacts on the economy, environment, 
    and people, including impacts on their human rights, across its activities and business relationships. 
    Provide the excerpts with page number, if any.
    """
    )
    combine_prompt = PromptTemplate(
        input_variables=["summaries"],
        template = """
    Provide a numbered list of any sentences or phrases relating to process to determine material topics, 
    specifically to describe the process it has followed to determine its material topics, including,  
    how it has identified actual and potential, negative and positive impacts on the economy, environment, 
    and people, including impacts on their human rights, across its activities and business relationships. 
    Provide the excerpts with page number, if any.
"""
    )


    # Create a RetrievalQA chain that combines the vector store with the model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={
    "question_prompt": question_prompt,
    "combine_prompt": combine_prompt,
}

    )

    return qa_chain

def create_generation_rag_system(index_path, embedding_model='sentence-transformers/all-mpnet-base-v2', model_name="llama3.2"):
    vector_store = load_faiss_index(index_path, embedding_model)

    # Initialize the Ollama model (Llama3.1)
    llm = OllamaLLM(model=model_name)

    question_prompt = PromptTemplate(
        input_variables=["context","question"],
        template = """
    You are an ESG report writer. Use the retrieved context to:
    1. Analyse the format, styles and information of typical ESG reports
    2. Generate an ESG report from other company reports provided
    3. Make comparisons to data from past years to analyse the ESG performance of the company this year

    Make it as detailed as possible

    Context:
    {context}

    Question: {question}

    If the context does not contain enough information, clearly state that the information is not available in the context provided.
    If possible, provide a step-by-step explanation and highlight key details.
    """
    )

    combine_prompt = PromptTemplate(
        input_variables=["summaries"],
        template = """
You have a collection of reports from a company. Using these information, please write **a single ESG report of that company**,
with these sections:
1. Executive Summary
2. Methodology
3. Detailed Findings in these aspects:
    a. Environmental criteria
        ​​​Climate change
            Carbon emissions
            Product carbon footprint
            Financing environmental impact
            Climate change vulnerability
        Natural resources
            Water stress
            Biodiversity and land use
            Raw material sourcing
        Pollution and waste
            Toxic emissions and waste
            Packaging material and waste
            Electronic waste
        Environmental opportunity
            Opportunities in clean tech
            Opportunities in green building
            Opportunities in renewable energy
    b. Social criteria
        Human capital
            Labor management
            Health and safety
            Human capital development
            Supply chain labor standards
        Product liability
            Product safety and quality
            Chemical safety
            Financial product safety
            Privacy and data security
            Responsible investment
            Health risks
        Stakeholder opposition
            Controversial sourcing
        Social opportunity
            Access to communication
            Access to finance
            Access to health care
            Opportunities in nutrition and health
    c. Governance criteria
        ​​​Corporate governance
            Board diversity
            Executive compensation
            Ownership
            Accounting
        Corporate behavior
            Business ethics
            Anti-competitive practices
            Corruption and instability
            Financial system instability
            Tax transparency

4. Align the reporting style as similar to the past year ESG reports by that company
5. Provide detailed analysis of each aspect

Use full sentences, bullet points and tables where appropriate.

{summaries}
"""
    )
    # Create a RetrievalQA chain that combines the vector store with the model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={
    "question_prompt": question_prompt,
    "combine_prompt": combine_prompt,
}

    )

    return qa_chain


if __name__ == "__main__":
    initial = input ("Would you like to analyse/generate an ESG report (A/G): ")
    if initial == "A":
        pdf_path = input("Enter path to your ESG PDF report:").strip().strip('"').strip("'")
        if not (os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf")):
            print ("file not found or not a pdf")
            sys.exit(1)

        pdf_temp_dir = tempfile.mkdtemp(prefix="pdf_input_")
        shutil.copy(pdf_path,pdf_temp_dir)

        text_temp_dir = tempfile.mkdtemp(prefix = "pdf_txt_")
        convert_pdfs_to_text(pdf_temp_dir,text_temp_dir)

        index_temp_dir = tempfile.mkdtemp(prefix="faiss_idx_")
        create_faiss_index(text_temp_dir,index_temp_dir)


    # Initialize the RAG system
        rag = create_analysis_rag_system(index_path=index_temp_dir)

        default_question = (
        "Analyse the uploaded ESG report and return :\n"
        "1. A concise executive summary\n"
        "2. The key ESG factors and metrics\n"
        "3. An overall ESG rating for the company"
        )
        raw = rag.invoke(default_question)
        if isinstance(raw, dict):
            report_text = raw.get("result") or raw.get("output") or next(iter(raw.values()))
        else:
            report_text=raw

        nlp = load_ner_model(patterns=METRIC_PATTERNS)
        raw_entities = extract_entites(report_text,nlp)

        harmonizer = TaxonomyHarmonizer(TAXONOMY_PATH)
        harmonized = harmonizer.harmonize([text for text,_ in raw_entities])

        kg = GraphIngester(NEO4J_URI, NEO4J_USER, NEO4J_PWD)
        entities_only = [e[0] for e in raw_entities]
        kg.ingest(company, year, list(zip(entities_only)))


        print("ESG ANALYSIS:")
        print(report_text)
    
        base=os.path.basename(pdf_path)
        name,_=os.path.splitext(base)
        docx_out=f"{name}_ESG_Analysis.docx"
        text_to_docx(report_text,docx_out)

    elif initial == "G":
        resource_folder = input("Enter relevant documents you have:").strip().strip('"').strip("'")
        if not os.path.isdir(resource_folder):
            print("folder not found")
            sys.exit(1)

        pdf_temp_dir = tempfile.mkdtemp(prefix="pdf_input_")
        for filename in os.listdir(resource_folder):
            if filename.lower().endswith(".pdf"):
                shutil.copy(
                    os.path.join(resource_folder,filename),
                    pdf_temp_dir
                )
            

        text_temp_dir = tempfile.mkdtemp(prefix = "pdf_txt_")
        convert_pdfs_to_text(pdf_temp_dir,text_temp_dir)

        index_temp_dir = tempfile.mkdtemp(prefix="faiss_idx_")
        create_faiss_index(text_temp_dir,index_temp_dir)

        rag = create_generation_rag_system(index_path=index_temp_dir)

        default_question = (
            "Analyse the uploaded company reports from current and past years and return :\n"
            "1. An ESG report of the latest year\n"
            "2. The key ESG factors and metrics\n"
            "3. Detailed analysis of each factor"
        )
        raw = rag.invoke(default_question)
        if isinstance(raw, dict):
            report_text = raw.get("result") or raw.get("output") or next(iter(raw.values()))
        else:
            report_text=raw

        print("ESG REPORT:")
        print(report_text)
        folder_name=os.path.basename(os.path.normpath(resource_folder))
        docx_out=f"{folder_name}_ESG_Report.docx"
        text_to_docx(report_text,docx_out)

