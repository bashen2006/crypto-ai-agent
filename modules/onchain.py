import requests
import time


# Whale Alert API
BASE_URL = "https://api.whale-alert.io/v1/transactions"

# 你的API key（先留空，后面我们再申请）
API_KEY = ""


# 限速控制
last_call = 0
MIN_INTERVAL = 2


def rate_limit():

    global last_call

    now = time.time()

    diff = now - last_call

    if diff < MIN_INTERVAL:

        time.sleep(MIN_INTERVAL - diff)

    last_call = time.time()


# =========================
# 获取巨鲸交易
# =========================
def get_whale_transactions(symbol="btc", min_value=500000):

    rate_limit()

    try:

        params = {

            "api_key": API_KEY,
            "currency": symbol,
            "min_value": min_value

        }

        r = requests.get(BASE_URL, params=params, timeout=10)

        data = r.json()

        if "transactions" not in data:

            return []

        return data["transactions"]

    except:

        return []


# =========================
# 判断是否有交易所流入
# =========================
def detect_exchange_inflow(transactions):

    inflow = []

    for tx in transactions:

        if "exchange" in str(tx):

            inflow.append(tx)

    return inflow


# =========================
# 判断是否有交易所流出
# =========================
def detect_exchange_outflow(transactions):

    outflow = []

    for tx in transactions:

        if "exchange" in str(tx):

            outflow.append(tx)

    return outflow
