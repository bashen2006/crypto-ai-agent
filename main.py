import time
import json
import requests
import pandas as pd
from datetime import datetime

CONFIG_FILE = "config.json"
LOG_FILE = "prediction_log.json"
AI_MEMORY_FILE = "ai_memory.json"

WHALE_THRESHOLD = 20000

# ===== 微信机器人配置 =====
WECHAT_BOT_ID = "aibRY2AELN5hw8GS18BGLm3Zw-AebN9sIsC"
WECHAT_SECRET = "IEsHdy4PpqjpvIu1G0JEwMW7ZTJ3MTU2whghxOcFwtv"


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


# =========================
# 微信机器人发送
# =========================
def send_wechat(message):

    try:

        url = "https://api.coze.cn/v1/bot/message"

        headers = {
            "Authorization": f"Bearer {WECHAT_SECRET}",
            "Content-Type": "application/json"
        }

        data = {
            "bot_id": WECHAT_BOT_ID,
            "content": message
        }

        requests.post(url, headers=headers, json=data)

    except Exception as e:
        print("微信发送失败:", e)


# =========================
# Telegram发送
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
# 双通道发送
# =========================
def send_notification(message, config):

    send_telegram(
        message,
        config["telegram_bot_token"],
        config["telegram_chat_id"]
    )

    send_wechat(message)


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

    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()

    if ma20 < ma60 and momentum < -0.03:
        return "底部"

    elif ma20 > ma60 and momentum > 0.03 and volume > avg_volume:
        return "启动"

    elif ma20 > ma60 and momentum > 0.06:
        return "主升浪"

    elif ma20 > ma60 and momentum < 0:
        return "顶部"

    else:
        return "震荡"


# =========================
# AI评分
# =========================
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


# =========================
# 信号强度
# =========================
def get_signal_strength(score):

    if score >= 75:
        return "强信号"
    elif score >= 60:
        return "中等信号"
    elif score >= 45:
        return "观察"
    else:
        return "弱信号"


# =========================
# 超级信号
# =========================
def check_super_signal(score, cycle, whale_volume):

    if score >= 75 and cycle == "主升浪" and whale_volume >= 50000:
        return True

    return False


# =========================
# 主程序
# =========================
def main():

    config = load_config()

    print("AI交易系统启动")

    while True:

        memory = load_ai_memory()

        print("\n开始市场扫描:", datetime.now())

        for coin in config["coins"]:

            try:

                df = get_okx_kline(coin)

                whale_volume = detect_whale(coin)

                trend = get_market_state(df)

                cycle = detect_market_cycle(df)

                score = calculate_score(df, memory, whale_volume)

                price = df["close"].iloc[-1]

                strength = get_signal_strength(score)

                super_signal = check_super_signal(score, cycle, whale_volume)

                if score >= config["buy_threshold"]:
                    signal = "买入"
                elif score <= config["sell_threshold"]:
                    signal = "卖出"
                else:
                    signal = "中性"

                print(
                    f"[{datetime.now()}] {coin} | 评分:{score} | 信号:{signal} | 周期:{cycle}"
                )

                if super_signal:

                    msg = f"""
🔥 AI超级信号

币种：{coin}

价格：{price}

行情阶段：{cycle}
行情评级：{trend}

AI评分：{score}

巨鲸资金：{int(whale_volume)} USDT

建议：强买入
"""

                    send_notification(msg, config)

                elif signal != "中性":

                    msg = f"""
AI交易信号

币种：{coin}

价格：{price}

行情阶段：{cycle}
行情评级：{trend}

信号：{signal}
强度：{strength}

评分：{score}

巨鲸资金：{int(whale_volume)}
"""

                    send_notification(msg, config)

            except Exception as e:

                print("错误:", e)

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
