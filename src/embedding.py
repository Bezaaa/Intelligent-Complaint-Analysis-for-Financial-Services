import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


data_path = Path("data/filtered_complaints.csv")
df = pd.read_csv(data_path)

# === 2. Text Chunking ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

docs = []
for idx, row in df.iterrows():
    chunks = text_splitter.split_text(row["cleaned_narrative"])
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "complaint_id": row["Complaint ID"] if "Complaint ID" in row else idx,
                    "product": row["Product"],
                }
            )
        )

print(f"Total chunks created: {len(docs)}")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_store = FAISS.from_documents(docs, embedding_model)


output_dir = Path("../vector_store")
output_dir.mkdir(exist_ok=True)
vector_store.save_local(str(output_dir))
print(f"Vector store saved to {output_dir}")
