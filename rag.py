import os
import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class ProRAGSystem:
    def __init__(self, file_path: str, persist_dir: str = "./rag_storage"):
        self.persist_dir = persist_dir
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        # 1. Models Setup
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 2. Storage & Parent-Child Splitters
        self.vectorstore = Chroma(
            collection_name="pro_rag", 
            embedding_function=self.embeddings,
            persist_directory=os.path.join(persist_dir, "chroma")
        )
        self.store = LocalFileStore(os.path.join(persist_dir, "parent_store"))
        
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

        # 3. Initialize Retrievers
        self._setup_retrievers(file_path)

    def _setup_retrievers(self, file_path):
        # Parent Retriever handles the logic of child-search -> parent-return
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        # Persistence Check: Only index if the store is empty
        stored_keys = list(self.store.yield_keys())
        if not stored_keys:
            print(" Indexing documents for the first time...")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(page_content=content)
            self.parent_retriever.add_documents([doc])
            stored_keys = list(self.store.yield_keys())

        # Pro Tip 2: Align BM25 with Parent Chunks
        # We retrieve the actual text of the parent chunks to feed BM25
        parent_texts = [self.store.mget([k])[0].decode("utf-8") for k in stored_keys]
        self.bm25_retriever = BM25Retriever.from_texts(parent_texts)
        self.bm25_retriever.k = 5

        # Hybrid Ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.parent_retriever],
            weights=[0.4, 0.6]
        )

    def generate_queries(self, question: str) -> List[str]:
        # Pro Tip 1: Clean Multi-Query Split using JSON
        prompt = ChatPromptTemplate.from_template("""
        Generate 3 different variations of this user question to improve retrieval.
        Output ONLY a JSON list of strings. 
        Example: ["var1", "var2", "var3"]
        Question: {question}""")
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw_output = chain.invoke({"question": question})
            # This ensures no "1., 2., 3." noise enters your search
            queries = json.loads(raw_output)
            return queries + [question]
        except:
            return [question]

    def ask(self, query: str):
        # Step 1: Multi-Query
        queries = self.generate_queries(query)
        
        # Step 2: Retrieve
        all_docs = []
        for q in queries:
            all_docs.extend(self.ensemble_retriever.get_relevant_documents(q))
        
        # Deduplicate
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()

        # Step 3: Rerank
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        top_context = "\n\n---\n\n".join([doc.page_content for doc, score in ranked[:5]])

        # Step 4: Final Answer
        final_prompt = f"Context:\n{top_context}\n\nQuestion: {query}\nAnswer:"
        return self.llm.predict(final_prompt)

# --- EXECUTION ---
if __name__ == "__main__":
    # Just point it to your file once
    rag = ProRAGSystem("morph.txt")
    
    while True:
        user_q = input("\nUser: ")
        if user_q.lower() == "exit": break
        print(f"\nAI: {rag.ask(user_q)}")
