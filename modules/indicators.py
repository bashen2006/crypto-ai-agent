import pandas as pd


# =========================
# 移动平均线 MA
# =========================
def MA(series, period):

    return series.rolling(period).mean()


# =========================
# 指数移动平均 EMA
# =========================
def EMA(series, period):

    return series.ewm(span=period, adjust=False).mean()


# =========================
# RSI 指标
# =========================
def RSI(series, period=14):

    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# MACD 指标
# =========================
def MACD(series):

    ema12 = EMA(series, 12)
    ema26 = EMA(series, 26)

    macd = ema12 - ema26
    signal = EMA(macd, 9)

    histogram = macd - signal

    return macd, signal, histogram


# =========================
# Bollinger 布林带
# =========================
def Bollinger(series, period=20):

    ma = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return upper, ma, lower


# =========================
# ATR 波动率
# =========================
def ATR(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    return atr
