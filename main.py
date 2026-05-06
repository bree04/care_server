"""
CareMate FastAPI 서버 (통합본)
- Ollama (exaone3.5:7.8b) 연동
- 의약품 CSV → SQLite 검색
- RAG (Chroma DB 벡터 검색) 연동
- 대화 기억 (chat_history)
- TTS (edge-tts)
- OCR (서버 경유 ocr.space)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import sqlite3
import ollama
import pandas as pd
import os
import json
import re
import uuid
import tempfile
import httpx
import edge_tts
from datetime import datetime

try:
    from rag import search_rag
    RAG_AVAILABLE = True
    print("RAG 모듈 로드 완료")
except Exception as e:
    RAG_AVAILABLE = False
    print(f"RAG 모듈 로드 실패 (LIKE 검색으로 대체): {e}")

app = FastAPI(title="CareMate API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "caremate_server.db"
CSV_PATH = "medicines_preprocessed.csv"
OCR_API_KEY = "K88624131688957"

SYSTEM_PROMPT = """당신은 어르신을 돌봐드리는 다정한 디지털 식물 '새싹이'입니다.
규칙:
1. 항상 존댓말을 쓰고, 따뜻하고 친근하게 말합니다.
2. 약 관련 질문에는 DB에서 찾은 정보를 쉬운 말로 설명합니다.
3. 어려운 의학 용어는 쉽게 풀어서 설명합니다.
4. 복약을 잘 하셨을 때는 칭찬과 격려를 아끼지 않습니다.
5. 확실하지 않은 의학 정보는 반드시 의사/약사 상담을 권유합니다.
6. 답변은 3문장 이내로 간결하게 합니다."""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT    NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alarms (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,
            medication_name TEXT    NOT NULL,
            time_hhmm       TEXT    NOT NULL,
            days_of_week    TEXT    NOT NULL,
            is_enabled      INTEGER DEFAULT 1,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medicines (
            item_seq        TEXT PRIMARY KEY,
            item_name       TEXT,
            entp_name       TEXT,
            chartn          TEXT,
            item_image      TEXT,
            print_front     TEXT,
            print_back      TEXT,
            drug_shape      TEXT,
            color_class1    TEXT,
            color_class2    TEXT,
            leng_long       REAL,
            leng_short      REAL,
            thick           REAL,
            class_no        TEXT,
            etc_otc_code    TEXT,
            item_eng_name   TEXT,
            edi_code        TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_name  ON medicines(item_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_shape ON medicines(drug_shape)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_color ON medicines(color_class1)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_otc   ON medicines(etc_otc_code)")
    conn.commit()
    conn.close()


def import_csv_to_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    count = cur.execute("SELECT COUNT(*) FROM medicines").fetchone()[0]
    conn.close()
    if count > 0:
        print(f"의약품 DB 이미 로드됨: {count}건")
        return
    if not os.path.exists(CSV_PATH):
        print(f"CSV 파일 없음: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.upper()
    col_map = {
        'item_seq': 'ITEM_SEQ', 'item_name': 'ITEM_NAME', 'entp_name': 'ENTP_NAME',
        'chartn': 'CHARTN', 'drug_shape': 'DRUG_SHAPE', 'color_class1': 'COLOR_CLASS1',
        'color_class2': 'COLOR_CLASS2', 'print_front': 'PRINT_FRONT', 'print_back': 'PRINT_BACK',
        'etc_otc_code': 'ETC_OTC_CODE', 'item_eng_name': 'ITEM_ENG_NAME', 'class_no': 'CLASS_NO',
        'leng_long': 'LENG_LONG', 'leng_short': 'LENG_SHORT', 'thick': 'THICK',
    }
    df_out = pd.DataFrame()
    for db_col, csv_col in col_map.items():
        df_out[db_col] = df[csv_col] if csv_col in df.columns else ""
    df_out['item_image'] = ""
    df_out['edi_code'] = ""
    conn = sqlite3.connect(DB_PATH)
    df_out.to_sql('medicines', conn, if_exists='replace', index=False)
    conn.close()
    print(f"의약품 {len(df_out)}건 DB 임포트 완료")


def save_chat(user_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id: str, limit: int = 6):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def search_medicine_by_name(keyword: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """SELECT item_name, entp_name, chartn, drug_shape, color_class1, color_class2, etc_otc_code
           FROM medicines WHERE item_name LIKE ? LIMIT 5""",
        (f"%{keyword}%",)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def search_medicine_by_shape(shape: str = None, color: str = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    query = "SELECT item_name, entp_name, drug_shape, color_class1, color_class2, chartn FROM medicines WHERE 1=1"
    params = []
    if shape:
        query += " AND drug_shape LIKE ?"
        params.append(f"%{shape}%")
    if color:
        query += " AND (color_class1 LIKE ? OR color_class2 LIKE ?)"
        params.extend([f"%{color}%", f"%{color}%"])
    query += " LIMIT 10"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def build_med_context(message: str) -> str:
    drug_keywords = ["약", "정", "캡슐", "처방", "먹", "복용", "부작용", "효능", "효과"]
    if not any(kw in message for kw in drug_keywords):
        return ""
    med_context = ""
    if RAG_AVAILABLE:
        try:
            rag_results = search_rag(message, top_k=3)
            if rag_results:
                med_context = "\n\n[의약품 DB 검색 결과]\n"
                for r in rag_results:
                    med_context += (
                        f"- {r['name']} ({r['company']}) | {r['shape']} | {r['color1']}\n"
                        f"  설명: {r['description']}\n"
                    )
                return med_context
        except Exception as e:
            print(f"RAG 검색 오류, LIKE 검색으로 대체: {e}")
    words = [w for w in message.split() if len(w) >= 2]
    for word in words:
        results = search_medicine_by_name(word)
        if results:
            med_context = "\n\n[의약품 DB 검색 결과]\n"
            for row in results[:3]:
                name, company, desc, shape, c1, c2, otc = row
                med_context += (
                    f"- {name} ({company}) | {shape} | {c1} | {otc}\n"
                    f"  설명: {desc}\n"
                )
            break
    return med_context


def delete_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"파일 삭제 실패: {e}")


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "ko-KR-SunHiNeural"

class OCRScanRequest(BaseModel):
    image: str

class OCRExtractRequest(BaseModel):
    ocr_text: str

class IdentifyRequest(BaseModel):
    shape: Optional[str] = None
    color: Optional[str] = None

class AlarmRequest(BaseModel):
    user_id: str
    medication_name: str
    time_hhmm: str
    days_of_week: str


# ── 기본 ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CareMate 서버 정상 동작 중", "rag_enabled": RAG_AVAILABLE}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── 챗봇 ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    save_chat(req.user_id, "user", req.message)
    med_context = build_med_context(req.message)
    history = get_recent_history(req.user_id)
    try:
        response = ollama.chat(
            model="exaone3.5:7.8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + med_context},
                *history,
                {"role": "user", "content": req.message}
            ]
        )
        ai_reply = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama 오류: {str(e)}")
    save_chat(req.user_id, "assistant", ai_reply)
    return {
        "reply": ai_reply,
        "med_context_used": bool(med_context),
        "rag_used": RAG_AVAILABLE and bool(med_context)
    }


# ── TTS ───────────────────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: TTSRequest, background_tasks: BackgroundTasks):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    clean_text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ?!.,~]', '', text)
    try:
        temp_dir = tempfile.gettempdir()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(temp_dir, filename)
        communicate = edge_tts.Communicate(text=clean_text, voice=req.voice)
        await communicate.save(output_path)
        background_tasks.add_task(delete_file, output_path)
        return FileResponse(path=output_path, media_type="audio/mpeg", filename="tts.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 실패: {e}")


# ── OCR ───────────────────────────────────────────────────────────────────────

@app.post("/ocr/scan")
async def ocr_scan(req: OCRScanRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'https://api.ocr.space/parse/image',
                data={
                    'base64Image': f'data:image/jpg;base64,{req.image}',
                    'language': 'kor',
                    'isOverlayRequired': 'false',
                    'detectOrientation': 'true',
                    'scale': 'true',
                    'OCREngine': '2',
                    'isTable': 'true',
                },
                headers={'apikey': OCR_API_KEY},
                timeout=30
            )
        result = response.json()
        if result.get('IsErroredOnProcessing'):
            return {'text': ''}
        parsed = result.get('ParsedResults', [])
        text = parsed[0].get('ParsedText', '') if parsed else ''
        print(f"OCR 원문: {text[:100]}...")
        return {'text': text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 실패: {e}")


@app.post("/ocr/extract")
async def ocr_extract(req: OCRExtractRequest):
    prompt = f"""다음은 약 봉투 OCR 텍스트입니다.
약품명만 골라서 JSON으로 반환하세요.
약국명, 날짜, 금액, 주의사항, 복약안내, 색상, 보관방법은 제외하세요.
반드시 JSON만 반환하고 다른 설명은 하지 마세요.
형식: {{"medicines": ["약품명1", "약품명2", "약품명3"]}}

OCR 텍스트:
{req.ocr_text}"""
    try:
        response = ollama.chat(
            model="exaone3.5:7.8b",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            return {"medicines": [], "matched": []}
        extracted = json.loads(match.group())
        ai_names = extracted.get("medicines", [])
        matched = []
        for name in ai_names:
            short = name[:4] if len(name) >= 4 else name
            results = search_medicine_by_name(short)
            if results:
                r = results[0]
                matched.append({
                    "name": r[0], "company": r[1], "description": r[2],
                    "shape": r[3], "color1": r[4], "color2": r[5],
                    "otc_type": r[6], "original": name, "db_matched": True,
                })
            else:
                matched.append({
                    "name": name, "company": "", "description": "",
                    "shape": "", "color1": "", "color2": "",
                    "otc_type": "", "original": name, "db_matched": False,
                })
        return {"medicines": ai_names, "matched": matched}
    except Exception as e:
        print(f"OCR 추출 오류: {e}")
        return {"medicines": [], "matched": [], "error": str(e)}


# ── 의약품 검색 ───────────────────────────────────────────────────────────────

@app.post("/medicine/identify")
def identify_medicine(req: IdentifyRequest):
    if not req.shape and not req.color:
        raise HTTPException(status_code=400, detail="shape 또는 color 중 하나는 필요합니다.")
    results = search_medicine_by_shape(req.shape, req.color)
    if not results:
        return {"found": False, "medicines": []}
    medicines = [
        {"name": r[0], "company": r[1], "shape": r[2],
         "color1": r[3], "color2": r[4], "description": r[5]}
        for r in results
    ]
    return {"found": True, "count": len(medicines), "medicines": medicines}


@app.get("/medicine/search")
def search_medicine(keyword: str):
    results = search_medicine_by_name(keyword)
    if not results:
        return {"found": False, "medicines": []}
    medicines = [
        {"name": r[0], "company": r[1], "description": r[2],
         "shape": r[3], "color1": r[4], "color2": r[5], "otc_type": r[6]}
        for r in results
    ]
    return {"found": True, "count": len(medicines), "medicines": medicines}


# ── 알람 ──────────────────────────────────────────────────────────────────────

@app.post("/alarm")
def register_alarm(req: AlarmRequest):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alarms (user_id, medication_name, time_hhmm, days_of_week) VALUES (?, ?, ?, ?)",
        (req.user_id, req.medication_name, req.time_hhmm, req.days_of_week)
    )
    alarm_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {
        "success": True,
        "alarm_id": alarm_id,
        "message": f"{req.medication_name} 알림이 {req.time_hhmm}에 등록되었습니다."
    }

@app.get("/alarm/{user_id}")
def get_alarms(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, medication_name, time_hhmm, days_of_week, is_enabled FROM alarms WHERE user_id=? AND is_enabled=1",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    alarms = [
        {"id": r[0], "medication_name": r[1], "time": r[2], "days": r[3], "enabled": bool(r[4])}
        for r in rows
    ]
    return {"user_id": user_id, "alarms": alarms}

@app.delete("/alarm/{alarm_id}")
def delete_alarm(alarm_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alarms SET is_enabled=0 WHERE id=?", (alarm_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "알림이 삭제되었습니다."}


# ── 대화 기록 ─────────────────────────────────────────────────────────────────

@app.get("/chat/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, timestamp FROM chat_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    history = [{"role": r[0], "content": r[1], "time": r[2]} for r in reversed(rows)]
    return {"user_id": user_id, "history": history}


# ── 서버 시작 ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    import_csv_to_db()
    print("CareMate 통합 서버 시작! (chat + TTS + OCR + 약품DB)")