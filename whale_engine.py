import requests

WHALE_THRESHOLD = 20000


def detect_whale(inst):

    url = f"https://www.okx.com/api/v5/market/trades?instId={inst}&limit=100"

    r = requests.get(url)

    data = r.json()["data"]

    whale_volume = 0

    for trade in data:

        size = float(trade["sz"])
        price = float(trade["px"])

        usdt_value = size * price

        if usdt_value >= WHALE_THRESHOLD:
            whale_volume += usdt_value

    return whale_volume
