from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import random
import json
import os
from datetime import datetime

from pricing import ConcertPricing      
from reward_func import calculate_reward
from llm_client import decide_with_llm

# Create or reuse a results file
RESULTS_FILE = "results.json"
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Setup concerts
base_prices = {
    "Coldplay": 7000,
    "Arijit Singh": 5000,
    "Taylor Swift": 9000
}
total_tickets = {
    "Coldplay": 1000,
    "Arijit Singh": 1500,
    "Taylor Swift": 2000
}
concert_dates = {
    "Coldplay": datetime(2025, 9, 20),
    "Arijit Singh": datetime(2025, 9, 25),
    "Taylor Swift": datetime(2025, 9, 30)
}
pricing = ConcertPricing(base_prices, total_tickets, concert_dates)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Build a quick state snapshot for the UI (use latest price or base)
    ui_state = {}
    for c in base_prices:
        latest = pricing.price_history[c][-1] if pricing.price_history[c] else pricing.base_prices[c]
        remaining = pricing.total_tickets[c] - pricing.sold_tickets[c]
        ui_state[c] = {
            "latest_price": latest,
            "remaining": remaining,
            "preference": pricing.user_preferences[c]
        }
    return templates.TemplateResponse("index.html", {"request": request, "state": ui_state})

@app.get("/state")
def get_state():
    """Return JSON state that the agent/LLM can use"""
    prices = {}
    rem = {}
    pref = {}
    for c in base_prices:
        prices[c] = pricing.price_history[c][-1] if pricing.price_history[c] else pricing.base_prices[c]
        rem[c] = pricing.total_tickets[c] - pricing.sold_tickets[c]
        pref[c] = pricing.user_preferences[c]
    return {"prices": prices, "remaining": rem, "preference": pref, "date": str(pricing.current_date)}

@app.post("/buy")
def buy(concert: str = Form(...), user_prompt: str = Form(...)):
    """
    Called by the UI when clicking a buy button.
    We simulate one day (pricing.simulate_purchase) using the user_prompt and a generated web_traffic,
    then compute the reward for the chosen concert and log results.
    """
    # generate a web_traffic dict (you can replace with real traffic)
    web_traffic = {c: random.randint(20, 100) for c in base_prices}

    # Run simulation for the day (this updates price_history, sold tickets, etc.)
    day_state = pricing.simulate_purchase(user_prompt, web_traffic)

    # Now get purchased price for the selected concert from day_state
    if concert not in day_state:
        return JSONResponse({"error": "concert not found"}, status_code=400)

    purchased_price = day_state[concert]["Price"]
    price_history = pricing.price_history[concert]

    reward, bin_index, bins = calculate_reward(price_history, purchased_price, num_bins=5)

    # Log the result
    entry = {
        "date": str(pricing.current_date),
        "user_prompt": user_prompt,
        "concert": concert,
        "price": purchased_price,
        "reward": reward,
        "bins": bins,
        "bin_index": bin_index,
        "web_traffic": web_traffic
    }
    with open(RESULTS_FILE, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)

    return JSONResponse(entry)

@app.post("/llm_buy")
def llm_buy(user_prompt: str = Form(...)):
    """
    Endpoint for the agent to ask the LLM which concert to buy.
    The endpoint will:
      - fetch current state,
      - call the LLM (OpenRouter) to decide,
      - if decision != 'Wait' then simulate a day and compute reward for that decision,
      - return the outcome.
    """
    # fetch state snapshot as JSON string
    state = get_state()
    state_json = json.dumps(state)

    available_artists = list(base_prices.keys())
    decision = decide_with_llm(user_prompt, state_json, available_artists)

    # If LLM says Wait, do not advance day (or we could still advance per design). We'll simulate purchase only if buy.
    if decision == "Wait":
        return JSONResponse({"decision": "Wait", "message": "LLM chose to wait", "state": state})

    # Simulate day and get day_state
    web_traffic = {c: random.randint(33, 100) for c in base_prices}
    day_state = pricing.simulate_purchase(user_prompt, web_traffic)

    purchased_price = day_state[decision]["Price"]
    price_history = pricing.price_history[decision]

    reward, bin_index, bins = calculate_reward(price_history, purchased_price, num_bins=5)

    entry = {
        "date": str(pricing.current_date),
        "user_prompt": user_prompt,
        "decision": decision,
        "price": purchased_price,
        "reward": reward,
        "bins": bins,
        "bin_index": bin_index,
        "web_traffic": web_traffic
    }
    with open(RESULTS_FILE, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)

    return JSONResponse(entry)

@app.get("/results")
def results():
    with open(RESULTS_FILE, "r") as f:
        return JSONResponse(json.load(f))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)