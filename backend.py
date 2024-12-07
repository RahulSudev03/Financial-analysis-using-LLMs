from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os

#Loading environment variables from the .env file
load_dotenv()

index_name = "stocks"
name_space = "stock-descriptions"

#Access the pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key = pinecone_api_key)
pinecone_index = pc.Index(index_name)

#Access groq using groq api key
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key= groq_api_key
)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)


def query_pinecone(embedding, top_k=10, namespace=name_space):
    """
    Performs a similarity search on Pinecone.

    Args:
        embedding (np.ndarray): The query embedding.
        top_k (int): Number of top results to return.
        namespace (str): The namespace in Pinecone to search in.

    Returns:
        dict: The search results from Pinecone.
    """
    try:
        return pinecone_index.query(vector= embedding.tolist(), top_k= top_k, include_metadata=True, namespace=namespace)
    except Exception as e:
        raise RuntimeError(f"Error querying pinecone: {e}")
    
def generate_response(query, top_k = 10):
    """
    Generate a response to the user's query by fetching data from Pinecone and processing it with groq
    """
    
    try:
        query_embedding = get_huggingface_embeddings(query)
        top_matches = query_pinecone(query_embedding,top_k = top_k)
        contexts = [item['metadata']['text'] for item in top_matches['matches']]
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : top_k]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
        system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.
        """

        llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_query}
            ]
        )

        return llm_response.choices[0].message.content
    
    except Exception as e:
        return f"An error occured: {e}"
        
        
        
    

