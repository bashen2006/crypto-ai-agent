import requests导入requests
import time导入time
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

    return返回 df


# 获取成交量
def get_volume(symbol):

    df = get_kline(symbol)

    if df is None:

        return None返回 无

    return float(df["volume"].iloc[-1])


# 获取最近N个收盘价
def get_close_series(symbol):

    df = get_kline(symbol)

    if df is None:

        return None返回 无返回无无返回 无返回无无返回 无返回无无返回 无返回无

    return df["close"]返回df["close"]


# 获取市场趋势（简单判断）
def market_trend(symbol):

    closes = get_close_series(symbol)

    if closes is None:如果收盘价为 None:如果收盘价为None:如果收盘价为

        return "unknown"返回 “未知”

    ma50 = closes.rolling(50).mean().iloc[-1]ma50 = 收盘价.rolling(50).mean().iloc[-1]

    ma200 = closes.rolling(200).mean().iloc[-1]ma200 = 收盘价.rolling(200).mean().iloc[-1]

    price = closes.iloc[-1]价格 = 收盘价.iloc[-1]

    if price > ma50 > ma200:如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：如果价格 > 50日均线 > 200日均线：

        return "bull"返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”return “bull”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”返回“牛”

    if price < ma50 < ma200:如果价格 < ma50 < ma200：如果价格 < ma50 < ma200:如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：如果价格 < ma50 < ma200：

        return "bear"返回“熊”返回“熊”

    return "sideways"返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”return “sideways”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”返回“横向”
