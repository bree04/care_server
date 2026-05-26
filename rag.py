# rag.py  —  CareMate RAG 모듈 (업데이트 버전)
# 변경사항: drug_info 컬렉션(e약은요) 추가, 두 컬렉션 통합 검색

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
import os

CSV_PATH   = "medicines_preprocessed.csv"
CHROMA_PATH = "./chroma_db"
SQLITE_PATH = "caremate_server.db"

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

client = chromadb.PersistentClient(path=CHROMA_PATH)

# 기존 컬렉션 (낱알식별 CSV 기반)
collection = client.get_or_create_collection("medicines")

# 신규 컬렉션 (e약은요 — 효능/부작용/주의사항)
collection_drug = client.get_or_create_collection("drug_info")


# ─────────────────────────────────────────────
# 기존 build_rag (변경 없음)
# ─────────────────────────────────────────────
def build_rag():
    if collection.count() > 0:
        print(f"✅ Chroma DB(medicines) 이미 구축됨: {collection.count()}건")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.fillna("")

    documents, ids, metadatas = [], [], []
    for i, row in df.iterrows():
        text = str(row.get("RAG_TEXT", "")).strip()
        if not text:
            continue
        documents.append(text)
        ids.append(str(i))
        metadatas.append({
            "name":        str(row.get("ITEM_NAME", "")),
            "company":     str(row.get("ENTP_NAME", "")),
            "shape":       str(row.get("DRUG_SHAPE", "")),
            "color1":      str(row.get("COLOR_CLASS1", "")),
            "description": str(row.get("CHARTN", "")),
        })

    print(f"임베딩 중... {len(documents)}건")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"✅ Chroma DB(medicines) 구축 완료: {collection.count()}건")


# ─────────────────────────────────────────────
# 통합 RAG 검색 — medicines + drug_info 동시 검색
# ─────────────────────────────────────────────
def search_rag(query: str, top_k: int = 3) -> list:
    """
    두 컬렉션에서 검색 후 합산 반환.
    각 결과에 source 필드 포함.
    """
    query_embedding = model.encode([query]).tolist()
    results = []

    # 기존 medicines 컬렉션
    if collection.count() > 0:
        res = collection.query(query_embeddings=query_embedding, n_results=top_k)
        for meta in (res["metadatas"][0] if res["metadatas"] else []):
            meta["source"] = "medicines"
            results.append(meta)

    # 신규 drug_info 컬렉션 (e약은요)
    if collection_drug.count() > 0:
        res2 = collection_drug.query(query_embeddings=query_embedding, n_results=top_k)
        for meta in (res2["metadatas"][0] if res2["metadatas"] else []):
            meta["source"] = "drug_info"
            results.append(meta)

    return results


# ─────────────────────────────────────────────
# SQLite 헬퍼 함수 — DUR 조회
# ─────────────────────────────────────────────
def _get_con():
    return sqlite3.connect(SQLITE_PATH)


def check_elderly_warning(drug_name: str) -> dict:
    """약품명으로 노인주의 여부 확인"""
    con = _get_con()
    cur = con.cursor()
    cur.execute(
        "SELECT 품목명, 금기내용, 약효분류 FROM dur_elderly_warning WHERE 품목명 LIKE ?",
        (f"%{drug_name}%",)
    )
    rows = cur.fetchall()
    con.close()
    return {
        "found": len(rows) > 0,
        "items": [{"품목명": r[0], "금기내용": r[1], "약효분류": r[2]} for r in rows]
    }


def check_duration_warning(ingredient: str) -> dict:
    """
    성분명으로 최대 투여기간 확인
    예) 졸피뎀 → 28일, 케토롤락 → 7일
    """
    con = _get_con()
    cur = con.cursor()
    cur.execute(
        "SELECT DUR성분명, 최대투여기간, 제형, 비고 FROM dur_duration_warning WHERE DUR성분명 LIKE ?",
        (f"%{ingredient}%",)
    )
    rows = cur.fetchall()
    con.close()
    return {
        "found": len(rows) > 0,
        "items": [{"성분명": r[0], "최대투여기간": r[1], "제형": r[2], "비고": r[3]} for r in rows]
    }


def check_age_taboo(ingredient: str) -> dict:
    """성분명으로 특정 연령대 금기 확인 (예: 65세 이상)"""
    con = _get_con()
    cur = con.cursor()
    cur.execute(
        "SELECT DUR성분명, 연령기준, 금기내용 FROM dur_age_taboo WHERE DUR성분명 LIKE ?",
        (f"%{ingredient}%",)
    )
    rows = cur.fetchall()
    con.close()
    return {
        "found": len(rows) > 0,
        "items": [{"성분명": r[0], "연령기준": r[1], "금기내용": r[2]} for r in rows]
    }


def get_pill_image(drug_name: str) -> str | None:
    """약품명으로 낱알 이미지 URL 반환 (OCR 결과 매핑용)"""
    con = _get_con()
    cur = con.cursor()
    cur.execute(
        "SELECT 이미지URL FROM pill_identification WHERE 품목명 LIKE ? LIMIT 1",
        (f"%{drug_name}%",)
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def get_drug_info_from_db(drug_name: str) -> dict | None:
    """
    e약은요 기반 약품 상세 정보 직접 조회 (RAG 보완용)
    약품명이 정확히 일치할 때 빠르게 반환
    """
    con = _get_con()
    # drug_info 컬렉션은 ChromaDB에 있으므로 여기선 pill_identification으로 기본 정보만
    cur = con.cursor()
    cur.execute(
        "SELECT 품목명, 성상, 분류명, 이미지URL FROM pill_identification WHERE 품목명 LIKE ? LIMIT 1",
        (f"%{drug_name}%",)
    )
    row = cur.fetchone()
    con.close()
    if row:
        return {"품목명": row[0], "성상": row[1], "분류명": row[2], "이미지URL": row[3]}
    return None


if __name__ == "__main__":
    build_rag()