import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Setup Streamlit page
st.set_page_config(page_title="CSV RAG Q&A", layout="wide")
st.title("📄 RAG Q&A Chatbot with CSV Retrieval")

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedder()

# Upload training CSV
uploaded_file = st.file_uploader("📂 Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded and loaded successfully!")
        st.dataframe(df.head())

        # Combine row into text
        st.write("🔄 Processing rows...")
        rows = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()
        embeddings = model.encode(rows, convert_to_numpy=True)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # User input
        st.subheader("💬 Ask your question")
        user_question = st.text_input("Enter your question here:")

        if user_question:
            q_embedding = model.encode([user_question])
            k = 3
            distances, indices = index.search(q_embedding, k)
            st.subheader("📂 Top Retrieved Rows:")
            for i in indices[0]:
                st.write(f"• {rows[i]}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("📌 Please upload your Training Dataset CSV file to start.")
