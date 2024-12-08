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
    
def generate_response(query, top_k , filters):
    """
    Generate a response to the user's query by fetching data from Pinecone and processing it with groq
    """
    
    try:
        query_embedding = get_huggingface_embeddings(query)
        top_matches = query_pinecone(query_embedding,top_k)
        
        if filters:
            sector_filter = filters.get("sector")
            min_cap = filters.get("min_cap")
            max_cap = filters.get("max_cap")
            
            #Filter by sector
            if sector_filter and sector_filter != "All":
                top_matches["matches"] = [
                    match
                    for match in top_matches["matches"]
                    if match["metadata"].get("Sector", "Unknown") == sector_filter
                ]
                
            
                
            #Filter by market cap
            top_matches["matches"] = [
                match
            for match in top_matches["matches"]
                if match['metadata'].get('Market Cap')>= min_cap*1000000000 and match['metadata']['Market Cap']<=max_cap*1000000000
            ]
            
        
        
        
        contexts = [
    f"Name: {item['metadata'].get('Name', 'Unknown')}\n"
    f"Market Cap: {item['metadata'].get('Market Cap', 'Unknown')}\n"
    f"Details: {item['metadata'].get('text', 'No details available')}"
    for item in top_matches['matches']
]
        filter_details = (
            f"Filters Applied:\n"
            f"Sector: {filters.get('sector', 'All')}\n"
            f"Market Cap min: {filters.get('min_cap')}\n\n"
            f"Market Cap max: {filters.get('max_cap')}\n\n"
            if filters
            else ""
        )
        augmented_query = (
            "<CONTEXT>\n"
            + "\n\n-------\n\n".join(contexts)
            + "\n-------\n</CONTEXT>\n\n\n\n"
            + filter_details
            + "MY QUESTION:\n"
            + query
        )

        # Step 5: Query Groq LLM for the final synthesized response
        system_prompt = """
        You are an expert in financial analysis and stock market trends.
        Using the context provided, answer the user's question as clearly and concisely as possible.
        When you give a response, this is the format I want you to follow:
        CompanyName(ticker): 
        - Market Cap:
        - Description:
        """
        llm_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ],
        )
        final_answer = llm_response.choices[0].message.content

        return final_answer

    except Exception as e:
        return f"An error occurred: {e}"