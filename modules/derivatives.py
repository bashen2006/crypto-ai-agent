import requests
import time

BASE_URL = "https://www.okx.com/api/v5"


# 限速控制
last_call = 0
MIN_INTERVAL = 1.2


def rate_limit():

    global last_call

    now = time.time()

    diff = now - last_call

    if diff < MIN_INTERVAL:

        time.sleep(MIN_INTERVAL - diff)

    last_call = time.time()


# =========================
# Funding Rate 资金费率
# =========================
def get_funding_rate(symbol):

    rate_limit()

    try:

        url = f"{BASE_URL}/public/funding-rate?instId={symbol}"

        r = requests.get(url, timeout=10)

        data = r.json()

        return float(data["data"][0]["fundingRate"])

    except:

        return None


# =========================
# Open Interest 未平仓合约
# =========================
def get_open_interest(symbol):

    rate_limit()

    try:

        url = f"{BASE_URL}/public/open-interest?instId={symbol}"

        r = requests.get(url, timeout=10)

        data = r.json()

        return float(data["data"][0]["oi"])

    except:

        return None


# =========================
# Long Short Ratio 多空比
# =========================
def get_long_short_ratio(symbol):

    rate_limit()

    try:

        url = f"{BASE_URL}/rubik/stat/contracts/long-short-account-ratio?instId={symbol}"

        r = requests.get(url, timeout=10)

        data = r.json()

        ratio = data["data"][0]["ratio"]

        return float(ratio)

    except:

        return None
