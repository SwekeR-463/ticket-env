## Ticket Env

### How to run?

1. Install dependencies 
```bash
pip3 install -r requirements.txt
```

2. Run the fastapi server
```bash
uvicorn app:app --reload
```

3. In another terminal Run the playwright agent backed by `gpt-oss`
```bash
python3 agent_playwright.py
```