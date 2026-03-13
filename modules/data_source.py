import requests
import time
import pandas as pd

BASE_URL = "https://www.okx.com/api/v5"

# API限速控制
last_call = 0
MIN_INTERVAL = 1.2


def rate_limit():

    global last_call

    now = time.time()

    diff = now - last_call

    if diff < MIN_INTERVAL:

        time.sleep(MIN_INTERVAL - diff)

    last_call = time.time()


# 获取最新价格
def get_price(symbol):

    rate_limit()

    url = f"{BASE_URL}/market/ticker?instId={symbol}"

    r = requests.get(url, timeout=10)

    data = r.json()

    return float(data["data"][0]["last"])


# 获取K线数据
def get_kline(symbol, bar="5m", limit=200):

    rate_limit()

    url = f"{BASE_URL}/market/candles?instId={symbol}&bar={bar}&limit={limit}"

    r = requests.get(url, timeout=10)

    data = r.json()

    if "data" not in data:

        return None

    df = pd.DataFrame(data["data"])

    df = df.iloc[::-1]

    df = df.astype(float)

    df.columns = [

        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_ccy",
        "vol_ccy_quote",
        "confirm"
    ]

    return df


# 获取成交量
def get_volume(symbol):

    df = get_kline(symbol)

    if df is None:

        return None

    return float(df["volume"].iloc[-1])


# 获取最近N个收盘价
def get_close_series(symbol):

    df = get_kline(symbol)

    if df is None:

        return None

    return df["close"]


# 获取市场趋势（简单判断）
def market_trend(symbol):

    closes = get_close_series(symbol)

    if closes is None:

        return "unknown"

    ma50 = closes.rolling(50).mean().iloc[-1]

    ma200 = closes.rolling(200).mean().iloc[-1]

    price = closes.iloc[-1]

    if price > ma50 > ma200:

        return "bull"

    if price < ma50 < ma200:

        return "bear"

    return "sideways"
