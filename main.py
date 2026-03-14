import time
import json
import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

CONFIG_FILE = "config.json"
LOG_FILE = "prediction_log.json"
AI_MEMORY_FILE = "ai_memory.json"

WHALE_THRESHOLD = 20000


# =========================
# 139邮箱提醒模块（新增）
# =========================
EMAIL_USER = "13781411151@139.com"
EMAIL_PASS = "98adfa5c93a3a509df00"
EMAIL_RECEIVER = "13781411151@139.com"

def send_email(subject, content):

    try:

        msg = MIMEText(content, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER

        server = smtplib.SMTP_SSL("smtp.139.com", 465)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

    except Exception as e:

        print("邮箱发送失败:", e)


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


# =========================
# AI记忆系统
# =========================
def load_ai_memory():

    try:
        with open(AI_MEMORY_FILE, "r") as f:
            return json.load(f)

    except:

        memory = {
            "trend_weight": 0.3,
            "volume_weight": 0.2,
            "momentum_weight": 0.2,
            "volatility_weight": 0.1
        }

        with open(AI_MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)

        return memory


def save_ai_memory(memory):

    with open(AI_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


# =========================
# Telegram提醒
# =========================
def send_telegram(message, token, chat_id):

    try:

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        payload = {
            "chat_id": chat_id,
            "text": message
        }

        requests.post(url, data=payload)

    except Exception as e:

        print("Telegram发送失败:", e)


# =========================
# 获取K线
# =========================
def get_okx_kline(inst):

    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit=100"

    r = requests.get(url)

    data = r.json()["data"]

    df = pd.DataFrame(data)

    df = df.iloc[:, :6]

    df.columns = ["ts", "open", "high", "low", "close", "volume"]

    df = df.astype(float)

    return df


# =========================
# 巨鲸检测
# =========================
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


# =========================
# AI行情评级
# =========================
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


# =========================
# AI行情周期
# =========================
def detect_market_cycle(df):

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    ma20 = df["ma20"].iloc[-1]
    ma60 = df["ma60"].iloc[-1]

    momentum = (df["close"].iloc[-1] - df["close"].iloc[-15]) / df["close"].iloc[-15]

    if ma20 < ma60 and momentum < -0.03:
        return "底部"

    elif ma20 > ma60 and momentum > 0.03:
        return "启动"

    elif ma20 > ma60 and momentum > 0.06:
        return "主升浪"

    else:
        return "震荡"


# =========================
# AI评分
# =========================
def calculate_score(df, memory, whale_volume):

    factors = {}

    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()

    ma7 = df["ma7"].iloc[-1]
    ma30 = df["ma30"].iloc[-1]

    factors["trend"] = 1 if ma7 > ma30 else -1

    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()

    factors["volume"] = 1 if volume > avg_volume * 1.5 else 0

    momentum = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

    factors["momentum"] = 1 if momentum > 0 else -1

    volatility = (df["high"] - df["low"]).mean()

    factors["volatility"] = 1 if volatility > df["close"].mean() * 0.01 else 0

    score = 50

    score += factors["trend"] * 20 * memory["trend_weight"]
    score += factors["volume"] * 10 * memory["volume_weight"]
    score += factors["momentum"] * 10 * memory["momentum_weight"]
    score += factors["volatility"] * 5 * memory["volatility_weight"]

    if whale_volume > 0:
        score += 5

    score = max(0, min(100, int(score)))

    return score, factors


# =========================
# AI复盘
# =========================
def review_predictions(memory):

    print("\n📊 AI因子贡献分析完成")


# =========================
# 暴涨币扫描
# =========================
def scan_hot_coins():
    return []


# =========================
# 市场情绪指数
# =========================
def calculate_market_sentiment():
    return "中性"


# =========================
# 主程序
# =========================
def main():

    config = load_config()

    print("AI交易系统启动")

    last_review = time.time()

    while True:

        memory = load_ai_memory()

        sentiment = calculate_market_sentiment()

        hot_coins = scan_hot_coins()

        for c in hot_coins:

            if c not in config["coins"]:
                config["coins"].append(c)

        for coin in config["coins"]:

            try:

                df = get_okx_kline(coin)

                whale_volume = detect_whale(coin)

                trend = get_market_state(df)

                cycle = detect_market_cycle(df)

                score, factors = calculate_score(df, memory, whale_volume)

                price = df["close"].iloc[-1]

                if score >= config["buy_threshold"]:
                    signal = "买入"

                elif score <= config["sell_threshold"]:
                    signal = "卖出"

                else:
                    signal = "中性"

                msg = f"{coin} {signal} 评分:{score}"

                print(msg)

                if signal != "中性":

                    send_telegram(
                        msg,
                        config["telegram_bot_token"],
                        config["telegram_chat_id"]
                    )

                    send_email(
                        "AI交易信号",
                        msg
                    )

            except Exception as e:

                print("错误:", e)

        if time.time() - last_review > 21600:

            review_predictions(memory)

            last_review = time.time()

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
