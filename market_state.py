import pandas as pd


def get_market_state(df):

    df["ma30"] = df["close"].rolling(30).mean()
    df["ma90"] = df["close"].rolling(90).mean()

    ma30 = df["ma30"].iloc[-1]
    ma90 = df["ma90"].iloc[-1]

    if ma30 > ma90:
        trend = "牛市"
    elif ma30 < ma90:
        trend = "熊市"
    else:
        trend = "震荡"

    volatility = (df["high"] - df["low"]).mean()

    if volatility > df["close"].mean() * 0.02:
        risk = "高波动"
    else:
        risk = "正常波动"

    return trend, risk
