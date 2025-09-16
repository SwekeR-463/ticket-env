import asyncio, json
from playwright.async_api import async_playwright

# It opens the page to read the UI, takes a user prompt, calls POST /llm_buy, then prints result.

SERVER_BASE = "http://127.0.0.1:8000"

async def run_agent_rounds(rounds=5, user_prompts=None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=200)
        page = await browser.new_page()

        for i in range(rounds):
            print(f"\n=== Agent round {i+1} ===")
            # Optionally scrape the page
            await page.goto(SERVER_BASE)
            # get state as text (could also call /state)
            state_text = await page.inner_text("body")
            print("Page body snapshot length:", len(state_text))

            # choose prompt
            prompt = user_prompts[i] if user_prompts and i < len(user_prompts) else "Find me the cheapest ticket"

            # Call backend /llm_buy with Playwright's request API
            resp = await page.request.post(
                f"{SERVER_BASE}/llm_buy",
                form={"user_prompt": prompt}
            )
            data = await resp.json()
            print("LLM-buy response:", json.dumps(data, indent=2))

            # small pause to observe
            await asyncio.sleep(1)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_agent_rounds(rounds=50))
