import requests
import time
import json
import statistics
from datetime import datetime

# =========================
# 读取配置
# =========================

with open("config.json") as f:
    config = json.load(f)

coins = config["coins"]
check_interval = config["check_interval"]

telegram_token = config["telegram_bot_token"]
telegram_chat_id = config["telegram_chat_id"]

whale_trade_usdt = config["whale_trade_usdt"]
volume_spike = config["volume_spike"]

# =========================
# Telegram
# =========================

def send_message(msg):

    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"

    data = {
        "chat_id": telegram_chat_id,
        "text": msg
    }

    try:
        requests.post(url, json=data, timeout=10)
    except:
        print("Telegram发送失败")


# =========================
# OKX K线
# =========================

def get_candles(symbol):

    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=5m&limit=100"

    r = requests.get(url, timeout=10)

    data = r.json()["data"]

    closes = []
    volumes = []

    for c in data:

        closes.append(float(c[4]))
        volumes.append(float(c[5]))

    closes.reverse()
    volumes.reverse()

    return closes, volumes


# =========================
# MA
# =========================

def MA(data, period):

    if len(data) < period:
        return None

    return sum(data[-period:]) / period


# =========================
# RSI
# =========================

def RSI(data, period=14):

    gains = []
    losses = []

    for i in range(-period, 0):

        change = data[i] - data[i - 1]

        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# =========================
# MACD
# =========================

def MACD(data):

    ema12 = statistics.mean(data[-12:])
    ema26 = statistics.mean(data[-26:])

    return ema12 - ema26


# =========================
# 巨鲸监控
# =========================

def check_whale(symbol):

    url = f"https://www.okx.com/api/v5/market/trades?instId={symbol}&limit=50"

    r = requests.get(url)

    trades = r.json()["data"]

    whales = []

    for t in trades:

        size = float(t["sz"])
        price = float(t["px"])

        usdt = size * price

        if usdt > whale_trade_usdt:

            side = "买入" if t["side"] == "buy" else "卖出"

            whales.append((side, int(usdt)))

    return whales


# =========================
# AI评分
# =========================

def calculate_score(ma7, ma30, rsi, macd, volume, avg_volume, whales):

    score = 50

    if ma7 > ma30:
        score += 15

    else:
        score -= 10

    if rsi < 30:
        score += 10

    if rsi > 70:
        score -= 10

    if macd > 0:
        score += 10

    if volume > avg_volume * volume_spike:
        score += 10

    if whales:
        score += 5

    return score


# =========================
# 记录交易信号
# =========================

def log_signal(coin, price, score, signal):

    try:

        with open("trade_log.json") as f:
            log = json.load(f)

    except:
        log = []

    log.append({

        "coin": coin,
        "time": str(datetime.now()),
        "price": price,
        "score": score,
        "signal": signal

    })

    with open("trade_log.json", "w") as f:
        json.dump(log, f, indent=2)


# =========================
# 自动复盘
# =========================

def review_signals():

    try:

        with open("trade_log.json") as f:
            log = json.load(f)

    except:
        return

    if len(log) < 5:
        return

    success = 0

    for item in log[-10:]:

        coin = item["coin"]

        price_old = item["price"]

        closes,_ = get_candles(coin)

        price_now = closes[-1]

        if item["signal"] == "buy":

            if price_now > price_old:
                success += 1

        if item["signal"] == "sell":

            if price_now < price_old:
                success += 1

    winrate = success / 10

    print("策略胜率:", winrate)

    if winrate < 0.4:

        send_message(f"⚠️AI策略近期胜率较低：{winrate}")


# =========================
# 主循环
# =========================

while True:

    print("AI量化系统运行中...")

    for coin in coins:

        try:

            closes, volumes = get_candles(coin)

            price = closes[-1]

            ma7 = MA(closes, 7)
            ma30 = MA(closes, 30)

            rsi = RSI(closes)

            macd = MACD(closes)

            volume = volumes[-1]

            avg_volume = statistics.mean(volumes[-20:])

            whales = check_whale(coin)

            score = calculate_score(
                ma7, ma30, rsi, macd, volume, avg_volume, whales
            )

            if score >= config["buy_threshold"]:

                signal = "buy"
                advice = "买入信号"

            elif score <= config["sell_threshold"]:

                signal = "sell"
                advice = "卖出信号"

            else:

                signal = "hold"
                advice = "观望"

            msg = f"""
{coin}

价格：{price}

MA7：{round(ma7,2)}
MA30：{round(ma30,2)}

RSI：{round(rsi,2)}
MACD：{round(macd,2)}

AI评分：{score}

建议：{advice}

时间：{datetime.now()}
"""

            print(msg)

            if signal != "hold":

                send_message(msg)

                log_signal(coin, price, score, signal)

            time.sleep(2)

        except Exception as e:

            print("错误:", e)

    review_signals()

    print("等待下一轮检测")

    time.sleep(check_interval)
