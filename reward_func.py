import numpy as np

def calculate_reward(price_history, purchased_price, preference=50, num_bins=10, threshold=2.0):
    """
    Bin-based reward relative to historical prices (min..max).
    
    Args:
        price_history (list): historical prices
        purchased_price (float): price paid by agent
        preference (int): user preference for this concert (0-100)
        num_bins (int): number of bins to divide prices
        threshold (float): minimum acceptable reward

    Returns:
        reward (float): final adjusted reward
        bin_index (int): bin number (1..num_bins)
        bins (list): bin edges
    """
    if not price_history:
        return 0, None, []

    prices = np.array(price_history, dtype=float)
    min_p, max_p = float(prices.min()), float(prices.max())

    if min_p == max_p:
        # All prices equal → max reward
        reward = num_bins - 1
        return reward, 1, [min_p] * (num_bins + 1)

    # Clip purchase price into [min, max]
    purchased = float(np.clip(purchased_price, min_p, max_p))

    bins = np.linspace(min_p, max_p, num_bins + 1).tolist()
    bin_index = int(np.digitize(purchased, bins, right=True))
    bin_index = max(1, min(num_bins, bin_index))  # ensure within range

    # Base reward: closer to min → higher reward
    reward = num_bins - bin_index
    reward = max(reward, 0)

    # --- Preference adjustment ---
    if preference > 50:
        reward += 0.25
    elif preference < 30:
        reward -= 1.5

    # --- Threshold adjustment ---
    if reward <= threshold:
        if preference > 50:
            reward += 0.75  # soften penalty for favorite concert
        else:
            reward -= 1.0  # harsher penalty if not preferred

    return reward, bin_index, bins