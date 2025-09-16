from datetime import datetime, timedelta
import random
import statistics
import numpy as np
import os
import json
import requests


class ConcertPricing:
    def __init__(self, base_prices, total_tickets, concert_dates, log_file="reward_log.json"):
        self.base_prices = base_prices
        self.total_tickets = total_tickets
        self.sold_tickets = {c: 0 for c in total_tickets}
        self.concert_dates = concert_dates
        self.current_date = datetime.now()

        # Keep track of daily prices history for median floor
        self.price_history = {c: [] for c in base_prices}

        # Store user preferences in state (default neutral)
        self.user_preferences = {c: 50 for c in base_prices}

        # For logging
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    # --- Multipliers ---
    def _time_multiplier(self, concert_name: str) -> float:
        days_before = (self.concert_dates[concert_name] - self.current_date).days
        if days_before > 30:
            return 0.85
        elif days_before >= 7:
            return 1.0
        else:
            return 1.25

    def _inventory_multiplier(self, concert_name: str) -> float:
        total = self.total_tickets[concert_name]
        sold = self.sold_tickets[concert_name]
        remaining_ratio = (total - sold) / total
        if remaining_ratio > 0.5:
            return 1.0
        elif remaining_ratio > 0.2:
            return 1.15
        else:
            return 1.35

    def _traffic_multiplier(self, web_traffic: int) -> float:
        if web_traffic < 30:
            return 0.95
        elif web_traffic < 70:
            return 1.0
        else:
            return 1.2

    def _preference_multiplier(self, concert_name: str) -> float:
        preference = self.user_preferences.get(concert_name, 50)
        if preference < 30:
            return 0.9
        elif preference < 70:
            return 1.0
        else:
            return 1.15

    # --- Price calculation ---
    def get_price(self, concert_name: str, web_traffic: int) -> float:
        base = self.base_prices[concert_name]
        t_factor = self._time_multiplier(concert_name)
        i_factor = self._inventory_multiplier(concert_name)
        w_factor = self._traffic_multiplier(web_traffic)
        p_factor = self._preference_multiplier(concert_name)

        raw_price = round(base * t_factor * i_factor * w_factor * p_factor, 2)

        # Apply median floor
        if self.price_history[concert_name]:
            floor_price = statistics.median(self.price_history[concert_name])
            final_price = max(raw_price, floor_price)
        else:
            final_price = raw_price

        return final_price

    # --- Multi-turn simulation per user prompt ---
    def simulate_purchase(self, user_prompt: str, web_traffic_dict: dict, use_llm: bool = False):
        """
        user_prompt: string indicating concert the user wants
        web_traffic_dict: current traffic for each concert
        use_llm: if True, parse with OpenRouter LLM instead of keyword match
        """
        # Parse user prompt to set preferences
        if use_llm:
            self._parse_user_prompt_llm(user_prompt)
        else:
            self._parse_user_prompt(user_prompt)

        print(f"\n📅 Date: {self.current_date.strftime('%Y-%m-%d')}")
        day_state = {}

        for concert in self.base_prices:
            traffic = web_traffic_dict[concert]
            price = self.get_price(concert, traffic)
            self.price_history[concert].append(price)

            # Simulate random ticket sales based on traffic
            tickets_sold_today = random.randint(0, max(1, traffic // 10))
            self.sell_ticket(concert, tickets_sold_today)

            # Calculate reward based on purchased price
            reward = self.calculate_reward(concert, price, num_bins=5)

            day_state[concert] = {
                "Date": self.current_date.strftime("%Y-%m-%d"),
                "Price": price,
                "Traffic": traffic,
                "Preference": self.user_preferences[concert],
                "Sold Today": tickets_sold_today,
                "Total Sold": self.sold_tickets[concert],
                "Remaining": self.total_tickets[concert] - self.sold_tickets[concert],
                "Floor Price": statistics.median(self.price_history[concert]),
                "Reward": reward
            }

            print(f"{concert}: Price=₹{price}, Traffic={traffic}, Pref={self.user_preferences[concert]}, "
                  f"Sold Today={tickets_sold_today}, Total Sold={self.sold_tickets[concert]}, "
                  f"Remaining={day_state[concert]['Remaining']}, Floor Price=₹{day_state[concert]['Floor Price']}, "
                  f"Reward={reward}")

        # Save logs for this day
        self._log_rewards(day_state)

        # Advance to next day
        self.current_date += timedelta(days=1)
        return day_state

    def sell_ticket(self, concert_name: str, num: int = 1):
        remaining = self.total_tickets[concert_name] - self.sold_tickets[concert_name]
        num = min(num, remaining)
        self.sold_tickets[concert_name] += num

    # --- Prompt parser (keyword match) ---
    def _parse_user_prompt(self, prompt: str):
        prompt_lower = prompt.lower()
        selected_concert = None

        for concert in self.base_prices:
            if concert.lower() in prompt_lower:
                selected_concert = concert
                break

        for concert in self.base_prices:
            if concert == selected_concert:
                self.user_preferences[concert] = 95
            else:
                self.user_preferences[concert] = 20

    # --- Prompt parser with LLM (OpenRouter) ---
    def _parse_user_prompt_llm(self, prompt: str):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": f"You are an assistant. Choose exactly one artist from this list: {list(self.base_prices.keys())}. Output only the artist name."},
            {"role": "user", "content": prompt},
        ]
        payload = {"model": "openai/gpt-4o-mini", "messages": messages}
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            print("⚠️ OpenRouter error, falling back to keyword match")
            return self._parse_user_prompt(prompt)

        choice = resp.json()["choices"][0]["message"]["content"].strip()
        for concert in self.base_prices:
            if concert.lower() in choice.lower():
                selected_concert = concert
                break
        else:
            selected_concert = list(self.base_prices.keys())[0]

        for concert in self.base_prices:
            if concert == selected_concert:
                self.user_preferences[concert] = 95
            else:
                self.user_preferences[concert] = 20

    # --- Reward function ---
    def calculate_reward(self, concert_name: str, purchased_price: float, num_bins: int = 5):
        price_history = self.price_history[concert_name]
        if not price_history:
            return 0

        min_price = min(price_history)
        max_price = max(price_history)

        if min_price == max_price:
            return num_bins - 1

        purchased_price_clipped = np.clip(purchased_price, min_price, max_price)
        bins = np.linspace(min_price, max_price, num_bins + 1)
        bin_index = np.digitize(purchased_price_clipped, bins, right=True)

        reward = num_bins - bin_index
        reward = max(reward, 0)

        preference = self.user_preferences.get(concert_name, 50)
        if preference > 70:
            reward += 0.5
        elif preference < 30:
            reward -= 0.5

        return max(reward, 0)

    # --- Logging ---
    def _log_rewards(self, day_state):
        with open(self.log_file, "r+") as f:
            data = json.load(f)
            data.append(day_state)
            f.seek(0)
            json.dump(data, f, indent=2)


# --- Example usage ---
if __name__ == "__main__":
    base_prices = {"Coldplay": 5000, "Arijit Singh": 3000, "Taylor Swift": 8000}
    total_tickets = {"Coldplay": 1000, "Arijit Singh": 1500, "Taylor Swift": 2000}
    concert_dates = {
        "Coldplay": datetime(2025, 12, 20),
        "Arijit Singh": datetime(2025, 11, 25),
        "Taylor Swift": datetime(2025, 12, 30),
    }

    pricing = ConcertPricing(base_prices, total_tickets, concert_dates)

    for _ in range(5):
        traffic = {
            "Coldplay": random.randint(20, 90),
            "Arijit Singh": random.randint(10, 70),
            "Taylor Swift": random.randint(40, 100),
        }
        user_prompt = "I want the cheapest tickets"
        pricing.simulate_purchase(user_prompt, traffic, use_llm=True)
