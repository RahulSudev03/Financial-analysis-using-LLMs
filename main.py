import streamlit as st
from backend import generate_response

st.set_page_config(page_title="Financial Analysis with LLMs", layout="centered")

# App title and description
st.title("📊 Stock Finder")
st.write("""welcome to the stock finder application!""")

# User input for query
query = st.text_input(
    label="💬 What would you like to search for?",
    placeholder="e.g., Find companies building data centers",
    help="Type your question about companies, sectors, or industries, and press 'Search'."
)

with st.expander("🔧 Filters (Optional)"):
    # Sector Filter
    sector = st.selectbox(
        label="Select Sector",
        options=["All", "Technology", "Healthcare", "Financials", "Industrials", "Consumer Goods"],
        index=0  # Default to "All"
    )

    # Market Cap Filter
    market_cap = st.slider(
        label="Market Capitalization (in billions)",
        min_value=0,
        max_value=1000,
        value=(0, 1000),  # Default range
        help="Adjust the range to filter companies based on their market cap in billions."
    )

    # Top K Matches Filter
    top_k = st.slider(
        label="Number of Matches (Top K)",
        min_value=1,
        max_value=20,
        value= 10,  # Default to 10 matches
        help="Set the number of top matches to return."
    )
    
    

# Placeholders for status and results
status_placeholder = st.empty()
results_placeholder = st.empty()

# Search button
if st.button("🔍 Search"):
    if query.strip():
        try:
            # Display status message
            status_placeholder.write("🔄 Processing your query...")

            # Filters for backend
            filters = {"sector": sector, "min_cap": market_cap[0], "max_cap": market_cap[1], "top_k": top_k}

            # Call the backend to generate the response
            result = generate_response(query, filters.get("top_k"), filters)

            # Clear the status message and display results
            status_placeholder.empty()
            results_placeholder.subheader("Answer:")
            results_placeholder.write(result)

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a query!")
