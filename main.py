import time
import json
import requests
import pandas as pd
from datetime import datetime

CONFIG_FILE = "config.json"
LOG_FILE = "prediction_log.json"
LEARNING_FILE = "learning_log.json"
AI_MEMORY_FILE = "ai_memory.json"


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
            "volatility_weight": 0.1,
            "sentiment_weight": 0.2
        }


# 获取OKX K线
def get_okx_kline(inst):

    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit=50"

    r = requests.get(url)

    data = r.json()["data"]

    df = pd.DataFrame(data)

    df = df.iloc[:, :6]

    df.columns = ["ts", "open", "high", "low", "close", "volume"]

    df = df.astype(float)

    return df


# AI评分（带权重）
def calculate_score(df, memory):

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

    score = max(0, min(100, int(score)))

    return score


# 发送Telegram
def send_telegram(message, token, chat_id):

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=payload)


# 保存预测
def save_prediction(coin, signal, price, score):

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


# AI策略进化
def evolve_strategy(accuracy):

    memory = load_ai_memory()

    if accuracy > 0.65:
        memory["trend_weight"] += 0.02

    if accuracy < 0.5:
        memory["trend_weight"] -= 0.02

    memory["trend_weight"] = max(0.1, min(0.5, memory["trend_weight"]))

    with open(AI_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

    print("AI策略已更新:", memory)


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

            if signal == "BUY" and price_now > price_then:
                correct += 1

            if signal == "SELL" and price_now < price_then:
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

    record = {
        "time": str(datetime.now()),
        "accuracy": accuracy,
        "checked_predictions": total
    }

    try:
        with open(LEARNING_FILE, "r") as f:
            learning_logs = json.load(f)
    except:
        learning_logs = []

    learning_logs.append(record)

    with open(LEARNING_FILE, "w") as f:
        json.dump(learning_logs, f, indent=4)

    evolve_strategy(accuracy)


# 分析币种
def analyze_coin(inst, config, memory):

    df = get_okx_kline(inst)

    score = calculate_score(df, memory)

    price = df["close"].iloc[-1]

    if score >= config["buy_threshold"]:
        signal = "BUY"
    elif score <= config["sell_threshold"]:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return signal, score, price


# 主程序
def main():

    config = load_config()

    print("===================================")
    print("AI交易系统已启动")
    print("开始监控OKX市场数据")
    print("监控币种:", config["coins"])
    print("扫描周期:", config["check_interval"], "秒")
    print("===================================")

    last_review = time.time()

    while True:

        memory = load_ai_memory()

        print("")
        print("开始新一轮市场扫描:", datetime.now())

        for coin in config["coins"]:

            try:

                signal, score, price = analyze_coin(coin, config, memory)

                print(
                    f"[{datetime.now()}] 币种:{coin} | 评分:{score} | 信号:{signal} | 当前价:{price}"
                )

               save_prediction(coin, signal, price, score)

if signal != "NEUTRAL":

    msg = f"""
【AI交易信号】

币种：{coin}

评分：{score}

当前价格：{price}

交易信号：{signal}

数据来源：OKX

时间：{datetime.now()}
"""

                    send_telegram(
                        msg,
                        config["telegram_bot_token"],
                        config["telegram_chat_id"]
                    )

            except Exception as e:

                print(f"[{datetime.now()}] 发生错误:", e)

        if time.time() - last_review > 21600:

            print("开始AI复盘...")

            review_predictions()

            last_review = time.time()

        print("本轮扫描完成，等待下一次扫描...")

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
