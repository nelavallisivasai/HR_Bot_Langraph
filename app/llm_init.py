from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import types

# Load the sentence-transformers model
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Custom embed_query method
    def embed_query(self, texts):
        embeddings = self.encode(texts)
        return embeddings.tolist()

    model.embed_query = types.MethodType(embed_query, model)
    return model

def initialize_llm(api_key):
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=api_key,  
        model_name="llama3-groq-70b-8192-tool-use-preview",
        max_retries=2,
    )
    return llm

def initialize_vector_store(embedding_model):
    vector_store = Chroma(
        collection_name="policies",
        persist_directory="../vectorstore",
        embedding_function=embedding_model,
    )
    return vector_store
