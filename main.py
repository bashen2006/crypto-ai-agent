import time
import json
import requests
import pandas as pd
from datetime import datetime

CONFIG_FILE = "config.json"
LOG_FILE = "prediction_log.json"
AI_MEMORY_FILE = "ai_memory.json"

WHALE_THRESHOLD = 20000


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def load_ai_memory():
    try:
        with open(AI_MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "trend_weight": 0.3,
            "volume_weight": 0.2,
            "momentum_weight": 0.2,
            "volatility_weight": 0.1
        }


# 获取K线
def get_okx_kline(inst):

    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit=50"

    r = requests.get(url)

    data = r.json()["data"]

    df = pd.DataFrame(data)

    df = df.iloc[:, :6]

    df.columns = ["ts", "open", "high", "low", "close", "volume"]

    df = df.astype(float)

    return df


# 巨鲸检测
def detect_whale(inst):

    try:

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

    except:
        return 0


# AI行情评级（升级版）
def get_market_state(df):

    df["ma30"] = df["close"].rolling(30).mean()
    df["ma90"] = df["close"].rolling(90).mean()

    ma30 = df["ma30"].iloc[-1]
    ma90 = df["ma90"].iloc[-1]

    momentum = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

    if ma30 > ma90 and momentum > 0.02:
        return "牛市"

    elif ma30 < ma90 and momentum < -0.02:
        return "熊市"

    else:
        return "震荡"


# AI评分
def calculate_score(df, memory, whale_volume):

    score = 50

    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()

    ma7 = df["ma7"].iloc[-1]
    ma30 = df["ma30"].iloc[-1]

    trend_score = 20 if ma7 > ma30 else -20

    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()

    volume_score = 10 if volume > avg_volume * 1.5 else 0

    momentum = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

    momentum_score = 10 if momentum > 0 else -10

    volatility = (df["high"] - df["low"]).mean()

    volatility_score = 5 if volatility > df["close"].mean() * 0.01 else 0

    score += trend_score * memory["trend_weight"]
    score += volume_score * memory["volume_weight"]
    score += momentum_score * memory["momentum_weight"]
    score += volatility_score * memory["volatility_weight"]

    if whale_volume > 0:
        score += 5

    score = max(0, min(100, int(score)))

    return score


# 信号强度
def get_signal_strength(score):

    if score >= 75:
        return "强信号"

    elif score >= 60:
        return "中等信号"

    elif score >= 45:
        return "观察"

    else:
        return "弱信号"


# Telegram
def send_telegram(message, token, chat_id):

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=payload)


# 扫描暴涨币
def scan_hot_coins():

    try:

        url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

        r = requests.get(url)

        data = r.json()["data"]

        coins = []

        for coin in data:

            change = float(coin["sodUtc8"])

            if change > 0.05:
                coins.append(coin["instId"])

        return coins[:3]

    except:
        return []


# 保存预测
def save_prediction(coin, signal, price, score, df):

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    record = {
        "time": str(datetime.now()),
        "coin": coin,
        "signal": signal,
        "price": price,
        "score": score
    }

    logs.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


# AI复盘
def review_predictions():

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        return

    if len(logs) == 0:
        return

    correct = 0
    total = 0

    for record in logs:

        coin = record["coin"]
        price_then = record["price"]
        signal = record["signal"]

        try:

            df = get_okx_kline(coin)

            price_now = df["close"].iloc[-1]

            if signal == "买入" and price_now > price_then:
                correct += 1

            elif signal == "卖出" and price_now < price_then:
                correct += 1

            total += 1

        except:
            continue

    if total == 0:
        return

    accuracy = correct / total

    print("AI复盘完成")
    print("预测次数:", total)
    print("正确次数:", correct)
    print("准确率:", accuracy)


# 主程序
def main():

    config = load_config()

    print("AI交易系统启动")

    last_review = time.time()
    last_scan = time.time()

    while True:

        memory = load_ai_memory()

        print("\n开始市场扫描:", datetime.now())

        if time.time() - last_scan > 3600:

            hot_coins = scan_hot_coins()

            for c in hot_coins:
                if c not in config["coins"]:
                    config["coins"].append(c)

            last_scan = time.time()

        for coin in config["coins"]:

            try:

                df = get_okx_kline(coin)

                whale_volume = detect_whale(coin)

                trend = get_market_state(df)

                score = calculate_score(df, memory, whale_volume)

                price = df["close"].iloc[-1]

                strength = get_signal_strength(score)

                if score >= config["buy_threshold"]:
                    signal = "买入"
                elif score <= config["sell_threshold"]:
                    signal = "卖出"
                else:
                    signal = "中性"

                print(
                    f"[{datetime.now()}] {coin} | 评分:{score} | 信号:{signal} | 强度:{strength} | 市场:{trend} | 巨鲸:{int(whale_volume)}"
                )

                save_prediction(coin, signal, price, score, df)

                if signal != "中性":

                    msg = f"""
AI交易信号

币种：{coin}

当前价格：{price}

交易信号：{signal}
信号强度：{strength}

AI评分：{score}

行情评级：{trend}

巨鲸资金：{int(whale_volume)} USDT
"""

                    send_telegram(
                        msg,
                        config["telegram_bot_token"],
                        config["telegram_chat_id"]
                    )

            except Exception as e:

                print("错误:", e)

        if time.time() - last_review > 21600:

            print("开始AI复盘")

            review_predictions()

            last_review = time.time()

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
