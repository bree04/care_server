# rag.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

CSV_PATH = "medicines_preprocessed.csv"  # ← 전처리된 파일로 변경
CHROMA_PATH = "./chroma_db"

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("medicines")


def build_rag():
    if collection.count() > 0:
        print(f"✅ Chroma DB 이미 구축됨: {collection.count()}건")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.fillna("")

    documents = []
    ids = []
    metadatas = []

    for i, row in df.iterrows():
        text = str(row.get("RAG_TEXT", "")).strip()  # ← RAG_TEXT 컬럼 사용
        if not text:
            continue
        documents.append(text)
        ids.append(str(i))
        metadatas.append({
            "name": str(row.get("ITEM_NAME", "")),
            "company": str(row.get("ENTP_NAME", "")),
            "shape": str(row.get("DRUG_SHAPE", "")),
            "color1": str(row.get("COLOR_CLASS1", "")),
            "description": str(row.get("CHARTN", "")),
        })

    print(f"임베딩 중... {len(documents)}건")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Chroma DB 구축 완료: {collection.count()}건")


def search_rag(query: str, top_k: int = 3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["metadatas"][0] if results["metadatas"] else []


if __name__ == "__main__":
    build_rag()