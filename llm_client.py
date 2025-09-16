import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

def decide_with_llm(user_prompt: str, state_json: str, available_artists: list):
    """
    Ask OpenRouter to pick one artist (or 'Wait'). Returns artist name or 'Wait'.
    If OpenRouter fails or key missing, fallback to keyword match or random choice.
    """
    # Basic fallback: keyword match
    def fallback():
        low = user_prompt.lower()
        for a in available_artists:
            if a.lower() in low:
                return a
        # if "cheapest" in prompt -> choose artist with text "cheapest" not known here,
        # return the first as fallback
        if "cheapest" in low:
            # attempt parse state_json for prices: naive implementation
            try:
                import json
                s = json.loads(state_json)
                # state expected to contain mapping artist->price under "prices"
                if isinstance(s, dict) and "prices" in s:
                    prices = s["prices"]
                    return min(prices, key=prices.get)
            except Exception:
                pass
        import random
        return random.choice(available_artists)

    if not OPENROUTER_KEY:
        # no key, fallback
        return fallback()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that chooses exactly one concert artist from the available list. "
            "Output only the artist name or the single word 'Wait' (without quotes)."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"User prompt: {user_prompt}\n\n"
            f"Current state (JSON):\n{state_json}\n\n"
            f"Available artists: {', '.join(available_artists)}\n\n"
            "Rules:\n"
            "- If user explicitly mentions an artist, pick that artist.\n"
            "- If user says 'cheapest' choose the cheapest available today.\n"
            "- If user says 'costliest' choose the costliest available today.\n"
            "- If user says 'wait' or similar, respond 'Wait'.\n"
            "- Output EXACTLY one token: artist name or 'Wait'."
        )
    }
    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [system_msg, user_msg],
        "max_tokens": 512,
        "temperature": 0.5
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        # path: choices[0].message.content
        choice = data["choices"][0]["message"]["content"].strip()
        # if choice contains artist substring match, return that
        for a in available_artists:
            if a.lower() in choice.lower():
                return a
        if choice.strip().lower() == "wait":
            return "Wait"
        # fallback if OpenRouter returns something unexpected
        return fallback()
    except Exception as e:
        # print warning and fallback
        print("OpenRouter error or timeout:", e)
        return fallback()

