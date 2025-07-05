import streamlit as st
import nbformat
from nbconvert import PythonExporter
import os

# Streamlit page setup
st.set_page_config(page_title="SVD Recommender", layout="wide")
st.title("🎬 Movie Recommendation System using SVD")

# Path to your Jupyter Notebook
NOTEBOOK_PATH = "Recommendation_System_SVD_Collaborative_Filtering.ipynb"

# Convert notebook to Python code
@st.cache_data
def load_notebook_code(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(nb)
    return source_code

# Load and execute notebook code
notebook_code = load_notebook_code(NOTEBOOK_PATH)
exec_globals = {}
exec(notebook_code, exec_globals)

# Access recommendation results from executed notebook
top_n = exec_globals["top_n"]

# Add user selection dropdown
st.subheader("👤 Choose a User ID to View Recommendations")
user_ids = sorted(top_n.keys())
selected_user = st.selectbox("Select User ID", user_ids)

# Show top 5 recommendations for selected user
if selected_user:
    st.subheader(f"Top 5 Recommendations for User `{selected_user}`:")
    for movie_id, predicted_rating in top_n[selected_user]:
        st.write(f"📽️ **Movie ID:** `{movie_id}` | ⭐ **Predicted Rating:** `{predicted_rating:.2f}`")
