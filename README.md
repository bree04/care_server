# 🌱 CareMate 백엔드 서버

## 폴더 구조
```
caremate_server/
├── main.py                          # FastAPI 서버 (이 파일)
├── requirements.txt                 # 패키지 목록
├── 의약품_낱알식별정보_데이터_2025년_.csv  # ← CSV 파일 여기에 복사!
└── caremate_server.db               # 자동 생성됨
```

---

## 1단계: Ollama 설치 & 모델 다운로드

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: https://ollama.com 에서 설치파일 다운로드

# 모델 다운로드 (~800MB)
ollama pull llama3.2:1b

# Ollama 실행 확인
ollama run llama3.2:1b "안녕하세요"
```

---

## 2단계: Python 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 3단계: CSV 파일 배치

`의약품_낱알식별정보_데이터_2025년_.csv` 파일을
`main.py`와 **같은 폴더**에 복사합니다.

---

## 4단계: 서버 실행

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 확인: http://localhost:8000
API 문서 자동 생성: http://localhost:8000/docs

---

## API 사용 예시

### 💬 AI 챗봇 (새싹이)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "message": "오늘 혈압약 먹었어요"}'
```

### 💊 약 이름으로 검색
```bash
curl "http://localhost:8000/medicine/search?keyword=아토르바"
```

### 🔍 약 모양/색상으로 식별
```bash
curl -X POST "http://localhost:8000/medicine/identify" \
  -H "Content-Type: application/json" \
  -d '{"shape": "원형", "color": "하양"}'
```

### ⏰ 복약 알림 등록
```bash
curl -X POST "http://localhost:8000/alarm" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "medication_name": "혈압약", "time_hhmm": "08:30", "days_of_week": "매일"}'
```

---

## Android 앱 연동 시 주의사항

- 에뮬레이터에서 호출할 때: `http://10.0.2.2:8000`
- 실제 기기에서 호출할 때: `http://[PC의 IP주소]:8000`
  - PC IP 확인: `ipconfig` (Windows) 또는 `ifconfig` (Mac/Linux)
- 같은 WiFi에 연결되어 있어야 합니다.

---

## Android Room DB와의 역할 분담

| 역할 | Android Room | FastAPI 서버 |
|------|-------------|-------------|
| 복용 기록 저장 | ✅ 메인 | - |
| 나무 성장 상태 | ✅ 메인 | - |
| 비콘 연동 | ✅ 메인 | - |
| AI 챗봇 대화 | - | ✅ 메인 |
| 의약품 정보 검색 | - | ✅ 메인 |
| 복약 알림 서버 동기화 | 로컬 알람 | ✅ 백업/동기화 |
