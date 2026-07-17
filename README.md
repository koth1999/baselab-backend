# BASELAB Backend

KBO 선수 기록, 팀 순위, 경기 일정 및 경기 중계 데이터를 수집하는 FastAPI 백엔드입니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 문서는 실행 후 `http://localhost:8000/docs`에서 확인할 수 있습니다.
