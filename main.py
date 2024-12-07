import streamlit as st
from backend import get_huggingface_embeddings, query_pinecone, generate_response

st.set_page_config(page_title = "Financial Analysis with LLMs", layout = "centered")

st.title("📊 Financial Analysis with LLMs")

st.write("""Welcome to Financial Analysis app!
         Enter your query below to find companies and insights based on your needs!""")

query = st.text_input(
    label="💬 What would you like to search for?",
    placeholder="e.g., Find companies building data centers",
    help="Type your question about companies, sectors, or industries, and press 'Submit'."
)

status_placeholder = st.empty()
results_placeholder = st.empty()

if st.button("🔍 Search"):
    if query.strip():
        try:
            # Step 1: Generate embeddings for the query
            status_placeholder.write("Generating embeddings for your query...")
            query_embedding = get_huggingface_embeddings(query)

            # Step 2: Query Pinecone for top matches
            status_placeholder.write("Fetching top matches from the database...")
            results = query_pinecone(query_embedding)
            
            status_placeholder.empty()

            # Step 3: Display results
            results_placeholder.subheader("Answer: ")
            results_placeholder.write(generate_response(query))


        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a query!")