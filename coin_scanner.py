import requests


def scan_hot_coins():

    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

    r = requests.get(url)

    data = r.json()["data"]

    coins = []

    for coin in data:

        change = float(coin["sodUtc8"])

        if change > 0.05:
            coins.append(coin["instId"])

    coins = coins[:3]

    return coins
