"""
CareMate FastAPI 서버
- Ollama (llama3.2:1b) 연동
- 의약품 CSV → SQLite 검색
- 대화 기억 (chat_history)
- 약 모양/색상 식별 API
- 복약 알림 등록 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import ollama
import pandas as pd
import os
from datetime import datetime

app = FastAPI(title="CareMate API", version="1.0.0")

# CORS 허용 (Flutter 앱에서 호출 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "caremate_server.db"
CSV_PATH = "의약품_낱알식별정보_데이터_2025년_.csv"  # 같은 폴더에 CSV 파일 위치

# ──────────────────────────────────────────
# AI 페르소나 (새싹이)
# ──────────────────────────────────────────
SYSTEM_PROMPT = """당신은 어르신을 돌봐드리는 다정한 디지털 식물 '새싹이'입니다.
규칙:
1. 항상 존댓말을 쓰고, 따뜻하고 친근하게 말합니다.
2. 약 관련 질문에는 DB에서 찾은 정보를 쉬운 말로 설명합니다.
3. 어려운 의학 용어는 쉽게 풀어서 설명합니다.
4. 복약을 잘 하셨을 때는 칭찬과 격려를 아끼지 않습니다.
5. 확실하지 않은 의학 정보는 반드시 의사/약사 상담을 권유합니다.
6. 답변은 3문장 이내로 간결하게 합니다."""


# ──────────────────────────────────────────
# DB 초기화
# ──────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 대화 기록 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT    NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 복약 알림 테이블 (Android Room의 ALARM 테이블과 동기화용)
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

    # 의약품 정보 테이블 (CSV import)
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

    # 인덱스 (검색 성능 최적화)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_name     ON medicines(item_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_shape    ON medicines(drug_shape)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_color    ON medicines(color_class1)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_otc      ON medicines(etc_otc_code)")

    conn.commit()
    conn.close()


def import_csv_to_db():
    """CSV 파일을 medicines 테이블에 임포트 (최초 1회)"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    count = cur.execute("SELECT COUNT(*) FROM medicines").fetchone()[0]
    conn.close()

    if count > 0:
        print(f"✅ 의약품 DB 이미 로드됨: {count}건")
        return

    if not os.path.exists(CSV_PATH):
        print(f"⚠️  CSV 파일 없음: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, encoding='cp949')
    df.columns = df.columns.str.lower()

    conn = sqlite3.connect(DB_PATH)
    df_selected = df[[
        'item_seq','item_name','entp_name','chartn','item_image',
        'print_front','print_back','drug_shape','color_class1','color_class2',
        'leng_long','leng_short','thick','class_no','etc_otc_code',
        'item_eng_name','edi_code'
    ]]
    df_selected.to_sql('medicines', conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ 의약품 {len(df)}건 DB 임포트 완료")


# ──────────────────────────────────────────
# DB 헬퍼 함수
# ──────────────────────────────────────────
def save_chat(user_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id: str, limit: int = 6):
    """최근 대화 n개 반환 (Ollama messages 형식)"""
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
    """약 이름으로 검색"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT item_name, entp_name, chartn, drug_shape, color_class1, color_class2, etc_otc_code FROM medicines WHERE item_name LIKE ? LIMIT 5",
        (f"%{keyword}%",)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def search_medicine_by_shape(shape: str = None, color: str = None):
    """모양/색상으로 약 식별"""
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


# ──────────────────────────────────────────
# Request / Response 모델
# ──────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    message: str

class IdentifyRequest(BaseModel):
    shape: Optional[str] = None   # 예: "원형", "타원형", "장방형"
    color: Optional[str] = None   # 예: "하양", "노랑", "분홍"

class AlarmRequest(BaseModel):
    user_id: str
    medication_name: str
    time_hhmm: str        # 예: "08:30"
    days_of_week: str     # 예: "월,수,금" 또는 "매일"


# ──────────────────────────────────────────
# API 엔드포인트
# ──────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CareMate 서버 정상 동작 중 🌱"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    새싹이 AI 챗봇
    - 약 관련 키워드 감지 시 DB 검색 결과를 AI에 주입
    - 대화 기록 저장 및 불러오기
    """
    save_chat(req.user_id, "user", req.message)

    # 약 관련 키워드 감지 → DB 검색 후 AI에 주입
    med_context = ""
    drug_keywords = ["약", "정", "캡슐", "처방", "먹", "복용", "부작용", "효능", "효과"]
    if any(kw in req.message for kw in drug_keywords):
        # 메시지에서 약 이름 키워드 추출 (공백 기준 2글자 이상 단어)
        words = [w for w in req.message.split() if len(w) >= 2]
        for word in words:
            results = search_medicine_by_name(word)
            if results:
                med_context = "\n\n[의약품 DB 검색 결과]\n"
                for row in results[:3]:
                    name, company, desc, shape, c1, c2, otc = row
                    med_context += f"- {name} ({company}) | {shape} | {c1} | {otc}\n  설명: {desc}\n"
                break

    history = get_recent_history(req.user_id)

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + med_context},
                *history,
                {"role": "user", "content": req.message}
            ]
        )
        ai_reply = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama 오류: {str(e)}\nOllama가 실행 중인지 확인하세요.")

    save_chat(req.user_id, "assistant", ai_reply)
    return {"reply": ai_reply, "med_context_used": bool(med_context)}


@app.post("/medicine/identify")
def identify_medicine(req: IdentifyRequest):
    """
    약 모양/색상으로 식별
    예) shape="원형", color="하양" → 해당 약 목록 반환
    """
    if not req.shape and not req.color:
        raise HTTPException(status_code=400, detail="shape 또는 color 중 하나는 필요합니다.")

    results = search_medicine_by_shape(req.shape, req.color)
    if not results:
        return {"found": False, "medicines": []}

    medicines = [
        {
            "name": r[0],
            "company": r[1],
            "shape": r[2],
            "color1": r[3],
            "color2": r[4],
            "description": r[5]
        }
        for r in results
    ]
    return {"found": True, "count": len(medicines), "medicines": medicines}


@app.get("/medicine/search")
def search_medicine(keyword: str):
    """약 이름 키워드로 검색"""
    results = search_medicine_by_name(keyword)
    if not results:
        return {"found": False, "medicines": []}

    medicines = [
        {
            "name": r[0],
            "company": r[1],
            "description": r[2],
            "shape": r[3],
            "color1": r[4],
            "color2": r[5],
            "otc_type": r[6]
        }
        for r in results
    ]
    return {"found": True, "count": len(medicines), "medicines": medicines}


@app.post("/alarm")
def register_alarm(req: AlarmRequest):
    """복약 알림 등록 (Android Room과 동기화용)"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alarms (user_id, medication_name, time_hhmm, days_of_week) VALUES (?, ?, ?, ?)",
        (req.user_id, req.medication_name, req.time_hhmm, req.days_of_week)
    )
    alarm_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"success": True, "alarm_id": alarm_id, "message": f"{req.medication_name} 알림이 {req.time_hhmm}에 등록되었습니다."}


@app.get("/alarm/{user_id}")
def get_alarms(user_id: str):
    """사용자의 복약 알림 목록 조회"""
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
    """알림 삭제"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alarms SET is_enabled=0 WHERE id=?", (alarm_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "알림이 삭제되었습니다."}


@app.get("/chat/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    """대화 기록 조회"""
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


# ──────────────────────────────────────────
# 서버 시작 시 DB 초기화
# ──────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    import_csv_to_db()
    print("🌱 CareMate 서버 시작!")
