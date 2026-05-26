"""
CareMate 의약품 데이터 임포트 스크립트 (전처리 포함)
====================================================
실행 전 이 파일을 caremate_server 폴더에 넣고 실행하세요.

  python add_drug_data.py

필요 패키지:
  pip install chromadb pandas openpyxl sentence-transformers
"""

import os
import re
import sqlite3
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SQLITE_PATH = "caremate_server.db"
CHROMA_PATH = "./chroma_db"
DATA_DIR    = "."
MODEL_NAME  = "jhgan/ko-sroberta-multitask"
BATCH_SIZE  = 64   # CPU 환경에서 안정적인 크기

XLSX = {
    "easyDrug":     os.path.join(DATA_DIR, "1523_e약은요정보.xlsx"),
    "elderlyWarn":  os.path.join(DATA_DIR, "1489_DUR유형별 품목 현황_노인주의.xlsx"),
    "ageTaboo":     os.path.join(DATA_DIR, "1487_DUR유형별 성분 현황_특정연령대금기.xlsx"),
    "durationWarn": os.path.join(DATA_DIR, "1486_DUR유형별 성분 현황_투여기간주의.xlsx"),
    "additiveWarn": os.path.join(DATA_DIR, "1485_DUR유형별 성분 현황_첨가제주의.xlsx"),
    "pillId":       os.path.join(DATA_DIR, "1505_의약품 낱알식별.xlsx"),
}


# ─────────────────────────────────────────────
# 공통 전처리 유틸
# ─────────────────────────────────────────────
def clean_text(val) -> str:
    """HTML 태그 제거, 연속 공백 정리, 의미없는 값 제거"""
    if pd.isna(val):
        return ""
    text = str(val).strip()
    text = re.sub(r"<[^>]+>", " ", text)       # HTML 태그 제거
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # HTML 엔티티 제거
    text = re.sub(r"\s+", " ", text).strip()   # 연속 공백 정리
    if text.lower() in ("nan", "none", "-", "해당없음", "해당사항없음"):
        return ""
    return text


def preprocess_easy_drug(df: pd.DataFrame) -> pd.DataFrame:
    """e약은요 전처리 — HTML 제거, 중복 제거, RAG_TEXT 생성"""
    df.columns = [
        "품목일련번호", "제품명", "업체명",
        "효능", "사용법", "경고주의사항", "주의사항",
        "상호작용", "부작용", "보관법",
        "공개일자", "수정일자", "낱알이미지", "사업자번호"
    ]
    for col in ["제품명", "업체명", "효능", "사용법", "경고주의사항",
                "주의사항", "상호작용", "부작용", "보관법"]:
        df[col] = df[col].apply(clean_text)

    df["품목일련번호"] = df["품목일련번호"].astype(str).str.strip()
    df = df[df["제품명"] != ""]
    df = df.drop_duplicates(subset="품목일련번호", keep="first")

    def build_rag_text(row):
        parts = [f"약품명: {row['제품명']}"]
        if row["효능"]:         parts.append(f"효능: {row['효능']}")
        if row["사용법"]:       parts.append(f"사용법: {row['사용법']}")
        if row["경고주의사항"]: parts.append(f"경고: {row['경고주의사항']}")
        if row["주의사항"]:     parts.append(f"주의사항: {row['주의사항']}")
        if row["상호작용"]:     parts.append(f"상호작용: {row['상호작용']}")
        if row["부작용"]:       parts.append(f"부작용: {row['부작용']}")
        if row["보관법"]:       parts.append(f"보관법: {row['보관법']}")
        return " | ".join(parts)

    df["RAG_TEXT"] = df.apply(build_rag_text, axis=1)
    df = df[df["RAG_TEXT"].str.len() > 20]
    return df.reset_index(drop=True)


def preprocess_dur(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """DUR 공통 전처리 — 텍스트 정리, 키 없는 행 제거, 중복 제거"""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text)
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()
    if key_col in df.columns:
        df = df[df[key_col].astype(str).str.strip() != ""]
        df = df.drop_duplicates(subset=key_col, keep="first")
    return df.reset_index(drop=True)


def preprocess_pill(df: pd.DataFrame) -> pd.DataFrame:
    """낱알식별 전처리 — 텍스트 정리, 품목명 없는 행 제거, 중복 제거"""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text)
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()
    if "품목명" in df.columns:
        df = df[df["품목명"] != ""]
    if "품목일련번호" in df.columns:
        df["품목일련번호"] = df["품목일련번호"].astype(str).str.strip()
        df = df.drop_duplicates(subset="품목일련번호", keep="first")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 1. e약은요 → ChromaDB
# ─────────────────────────────────────────────
def import_easy_drug_to_chroma():
    print("\n[1/2] e약은요 → ChromaDB 임베딩 시작...")

    # 기존 컬렉션 확인 — 이미 완료됐으면 스킵
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("drug_info")
    if collection.count() > 0:
        print(f"  ✅ drug_info 이미 구축됨: {collection.count()}건 (스킵)")
        return

    print(f"  모델 로드 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("  엑셀 로드 및 전처리 중...")
    df = preprocess_easy_drug(pd.read_excel(XLSX["easyDrug"]))
    print(f"  전처리 완료: {len(df)}건")

    documents, ids, metadatas = [], [], []
    for _, row in df.iterrows():
        documents.append(row["RAG_TEXT"])
        ids.append(f"easy_{row['품목일련번호']}")
        metadatas.append({
            "name":      row["제품명"],
            "company":   row["업체명"],
            "image_url": str(row.get("낱알이미지", "")),
            "source":    "e약은요",
        })

    total = len(documents)
    print(f"  임베딩 시작 ({total}건, 배치 {BATCH_SIZE}개씩 — 완료까지 약 10~20분 소요)")
    print("  [중단하지 말고 기다려 주세요]")

    all_embeddings = []
    for i in range(0, total, BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(emb)
        done = min(i + BATCH_SIZE, total)
        pct  = done / total * 100
        bar  = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  [{bar}] {done}/{total} ({pct:.1f}%)", end="\r")

    print(f"\n  ChromaDB에 저장 중...")
    collection.add(
        documents=documents,
        embeddings=all_embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"  ✅ ChromaDB drug_info 구축 완료: {collection.count()}건")


# ─────────────────────────────────────────────
# STEP 2. DUR / 낱알식별 → SQLite
# ─────────────────────────────────────────────
def import_dur_to_sqlite():
    print("\n[2/2] DUR 데이터 → SQLite 시작...")
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()

    # ── 노인주의 품목 ─────────────────────────
    print("  노인주의 품목...")
    cur.execute("DROP TABLE IF EXISTS dur_elderly_warning")
    cur.execute("""
        CREATE TABLE dur_elderly_warning (
            품목일련번호 TEXT PRIMARY KEY,
            품목명      TEXT,
            업소명      TEXT,
            주성분      TEXT,
            약효분류    TEXT,
            고시일자    TEXT,
            금기내용    TEXT,
            DUR성분코드 TEXT,
            DUR성분명   TEXT
        )
    """)
    df = preprocess_dur(pd.read_excel(XLSX["elderlyWarn"]), key_col="품목일련번호")
    rows = [(
        str(r.get("품목일련번호","")), str(r.get("품목명","")),
        str(r.get("업소명","")),       str(r.get("주성분","")),
        str(r.get("약효분류","")),     str(r.get("고시일자","")),
        str(r.get("금기내용","")),     str(r.get("DUR성분코드","")),
        str(r.get("Dur성분명","")) or str(r.get("DUR성분명","")),
    ) for _, r in df.iterrows()]
    cur.executemany("INSERT OR REPLACE INTO dur_elderly_warning VALUES(?,?,?,?,?,?,?,?,?)", rows)
    print(f"    ✓ {len(rows)}건")

    # ── 특정연령대금기 성분 ────────────────────
    print("  특정연령대금기...")
    cur.execute("DROP TABLE IF EXISTS dur_age_taboo")
    cur.execute("""
        CREATE TABLE dur_age_taboo (
            DUR일련번호 TEXT PRIMARY KEY,
            DUR성분코드 TEXT,
            DUR성분명   TEXT,
            금기내용    TEXT,
            제형        TEXT,
            연령기준    TEXT,
            고시일자    TEXT
        )
    """)
    df = preprocess_dur(pd.read_excel(XLSX["ageTaboo"]), key_col="DUR일련번호")
    rows = [(
        str(r["DUR일련번호"]), str(r["DUR성분코드"]), str(r["DUR성분명"]),
        str(r["금기내용"]),    str(r["제형"]),        str(r["연령기준"]),
        str(r["고시일자"]),
    ) for _, r in df.iterrows()]
    cur.executemany("INSERT OR REPLACE INTO dur_age_taboo VALUES(?,?,?,?,?,?,?)", rows)
    print(f"    ✓ {len(rows)}건")

    # ── 투여기간주의 성분 ──────────────────────
    print("  투여기간주의...")
    cur.execute("DROP TABLE IF EXISTS dur_duration_warning")
    cur.execute("""
        CREATE TABLE dur_duration_warning (
            DUR일련번호  TEXT PRIMARY KEY,
            DUR성분코드  TEXT,
            DUR성분명    TEXT,
            제형         TEXT,
            최대투여기간 TEXT,
            고시일자     TEXT,
            비고         TEXT
        )
    """)
    df = preprocess_dur(pd.read_excel(XLSX["durationWarn"]), key_col="DUR일련번호")
    rows = [(
        str(r["DUR일련번호"]), str(r["DUR성분코드"]), str(r["DUR성분명"]),
        str(r["제형"]),        str(r["최대투여기간"]), str(r["고시일자"]),
        str(r["비고"]),
    ) for _, r in df.iterrows()]
    cur.executemany("INSERT OR REPLACE INTO dur_duration_warning VALUES(?,?,?,?,?,?,?)", rows)
    print(f"    ✓ {len(rows)}건")

    # ── 첨가제주의 성분 ────────────────────────
    print("  첨가제주의...")
    cur.execute("DROP TABLE IF EXISTS dur_additive_warning")
    cur.execute("""
        CREATE TABLE dur_additive_warning (
            DUR일련번호 TEXT PRIMARY KEY,
            DUR성분코드 TEXT,
            DUR성분명   TEXT,
            금기내용    TEXT,
            비고        TEXT
        )
    """)
    df = preprocess_dur(pd.read_excel(XLSX["additiveWarn"]), key_col="DUR일련번호")
    rows = [(
        str(r["DUR일련번호"]), str(r["DUR성분코드"]), str(r["DUR성분명"]),
        str(r["금기내용"]),    str(r["비고"]),
    ) for _, r in df.iterrows()]
    cur.executemany("INSERT OR REPLACE INTO dur_additive_warning VALUES(?,?,?,?,?)", rows)
    print(f"    ✓ {len(rows)}건")

    # ── 낱알식별 ──────────────────────────────
    print("  낱알식별 (27,000건)...")
    cur.execute("DROP TABLE IF EXISTS pill_identification")
    cur.execute("""
        CREATE TABLE pill_identification (
            품목일련번호 TEXT PRIMARY KEY,
            품목명      TEXT,
            업소명      TEXT,
            성상        TEXT,
            이미지URL   TEXT,
            표시앞      TEXT,
            표시뒤      TEXT,
            제형        TEXT,
            색상앞      TEXT,
            크기장축    TEXT,
            크기단축    TEXT,
            분류번호    TEXT,
            분류명      TEXT,
            전문일반    TEXT,
            보험코드    TEXT
        )
    """)
    df = preprocess_pill(pd.read_excel(XLSX["pillId"]))
    rows = [(
        str(r.get("품목일련번호","")), str(r.get("품목명","")),
        str(r.get("업소명","")),       str(r.get("성상","")),
        str(r.get("큰제품이미지","")), str(r.get("표시앞","")),
        str(r.get("표시뒤","")),       str(r.get("의약품제형","")),
        str(r.get("색상앞","")),       str(r.get("크기장축","")),
        str(r.get("크기단축","")),     str(r.get("분류번호","")),
        str(r.get("분류명","")),       str(r.get("전문일반구분","")),
        str(r.get("보험코드","")),
    ) for _, r in df.iterrows()]
    cur.executemany(
        "INSERT OR REPLACE INTO pill_identification VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pill_name     ON pill_identification(품목명)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_elderly_name  ON dur_elderly_warning(품목명)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_duration_comp ON dur_duration_warning(DUR성분명)")

    con.commit()
    con.close()
    print(f"    ✓ {len(rows)}건 + 인덱스 생성 완료")


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  CareMate 의약품 데이터 임포트 (전처리 포함)")
    print("=" * 52)

    # 파일 존재 확인
    missing = [k for k, v in XLSX.items() if not os.path.exists(v)]
    if missing:
        print(f"\n❌ 파일 없음: {missing}")
        exit(1)

    # 이전에 중단된 drug_info 컬렉션이 있으면 초기화
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection("drug_info")
        count = col.count()
        if 0 < count < 4000:  # 완성되지 않은 컬렉션
            print(f"\n⚠️  이전에 중단된 drug_info 감지 ({count}건) → 초기화 후 재시작")
            client.delete_collection("drug_info")
    except Exception:
        pass  # 컬렉션 없으면 그냥 진행

    import_easy_drug_to_chroma()
    import_dur_to_sqlite()

    print("\n" + "=" * 52)
    print("  ✅ 모든 임포트 완료!")
    print("  서버는 평소처럼 python main.py 로 실행하세요.")
    print("=" * 52)