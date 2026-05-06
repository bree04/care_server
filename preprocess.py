# preprocess.py
"""
의약품 낱알식별정보 전처리 스크립트
- 2020~2025년 데이터 합치기
- 결측값 처리
- RAG용 텍스트 문장 생성
- 중복 제거 (ITEM_SEQ 기준, 최신 연도 우선)
- 최종 CSV 저장: medicines_preprocessed.csv
"""

import pandas as pd
import os

# ── 파일 경로 설정 ──────────────────────────────
FILES = [
    ("2020", "의약품 낱알식별정보 데이터(2020년).xls", "xls"),
    ("2021", "의약품 낱알식별정보 데이터(2021년).csv", "csv"),
    ("2022", "의약품 낱알식별정보 데이터(2022년).csv", "csv"),
    ("2023", "의약품 낱알식별정보 데이터(2023년).csv", "csv"),
    ("2024", "의약품 낱알식별정보 데이터(2024년).csv", "csv"),
    ("2025", "의약품_낱알식별정보_데이터_2025년_.csv", "csv"),
]

KEEP_COLS = [
    "ITEM_SEQ",
    "ITEM_NAME",
    "ENTP_NAME",
    "CHARTN",
    "DRUG_SHAPE",
    "COLOR_CLASS1",
    "COLOR_CLASS2",
    "PRINT_FRONT",
    "PRINT_BACK",
    "ETC_OTC_CODE",
    "ITEM_ENG_NAME",
    "CLASS_NO",
    "LENG_LONG",
    "LENG_SHORT",
    "THICK",
]

OUTPUT_PATH = "medicines_preprocessed.csv"


def load_all() -> pd.DataFrame:
    """전 연도 파일 로드 후 합치기"""
    dfs = []
    for year, filename, fmt in FILES:
        if not os.path.exists(filename):
            print(f"⚠️  파일 없음 (건너뜀): {filename}")
            continue
        try:
            if fmt == "xls":
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename, encoding="cp949")

            df.columns = df.columns.str.upper().str.strip()
            df["SOURCE_YEAR"] = year
            dfs.append(df)
            print(f"✅ {year}년 로드: {len(df)}건")
        except Exception as e:
            print(f"❌ {year}년 로드 실패: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📦 전체 합계: {len(combined)}건")
    return combined


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """전처리 파이프라인"""

    # 1. 필요한 컬럼만 선택 (없는 컬럼은 빈값으로)
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[KEEP_COLS + ["SOURCE_YEAR"]].copy()

    # 2. 결측값 처리
    str_cols = [
        "ITEM_NAME", "ENTP_NAME", "CHARTN", "DRUG_SHAPE",
        "COLOR_CLASS1", "COLOR_CLASS2", "PRINT_FRONT", "PRINT_BACK",
        "ETC_OTC_CODE", "ITEM_ENG_NAME", "CLASS_NO"
    ]
    for col in str_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].replace("nan", "")

    num_cols = ["LENG_LONG", "LENG_SHORT", "THICK"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 3. ITEM_NAME 없는 행 제거
    df = df[df["ITEM_NAME"] != ""].reset_index(drop=True)

    # 4. 중복 제거: ITEM_SEQ 기준, 최신 연도 우선
    df["SOURCE_YEAR"] = df["SOURCE_YEAR"].astype(int)
    df = df.sort_values("SOURCE_YEAR", ascending=False)
    df = df.drop_duplicates(subset="ITEM_SEQ", keep="first")
    df = df.sort_values("ITEM_NAME").reset_index(drop=True)
    print(f"🔍 중복 제거 후: {len(df)}건")

    # 5. RAG용 텍스트 문장 생성
    def build_rag_text(row):
        parts = [row["ITEM_NAME"]]

        if row["ENTP_NAME"]:
            parts.append(f"제조사: {row['ENTP_NAME']}")
        if row["DRUG_SHAPE"]:
            parts.append(f"모양: {row['DRUG_SHAPE']}")

        colors = " ".join(filter(None, [row["COLOR_CLASS1"], row["COLOR_CLASS2"]]))
        if colors:
            parts.append(f"색상: {colors}")

        prints = " ".join(filter(None, [row["PRINT_FRONT"], row["PRINT_BACK"]]))
        if prints:
            parts.append(f"각인: {prints}")

        if row["ETC_OTC_CODE"]:
            parts.append(row["ETC_OTC_CODE"])
        if row["CHARTN"]:
            parts.append(row["CHARTN"])

        return " | ".join(parts)

    df["RAG_TEXT"] = df.apply(build_rag_text, axis=1)

    return df


def main():
    print("=" * 50)
    print("의약품 데이터 전처리 시작")
    print("=" * 50)

    df = load_all()
    df = preprocess(df)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료: {OUTPUT_PATH} ({len(df)}건)")

    print("\n=== RAG_TEXT 샘플 3건 ===")
    for text in df["RAG_TEXT"].head(3):
        print(f"  {text}\n")


if __name__ == "__main__":
    main()