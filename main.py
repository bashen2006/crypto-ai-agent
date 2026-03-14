import time
import json
import requests
import pandas as pd
from datetime import datetime

CONFIG_FILE = "config.json"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


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


# 计算评分
def calculate_score(df):

    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()

    ma7 = df["ma7"].iloc[-1]
    ma30 = df["ma30"].iloc[-1]

    score = 50

    # 趋势评分
    if ma7 > ma30:
        score += 20
    else:
        score -= 20

    # 成交量评分
    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()

    if volume > avg_volume * 1.5:
        score += 10

    return int(score)


# 发送Telegram
def send_telegram(message, token, chat_id):

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=payload)


# 分析单个币
def analyze_coin(inst, config):

    df = get_okx_kline(inst)

    score = calculate_score(df)

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

    while True:

        print("")
        print("开始新一轮市场扫描:", datetime.now())

        for coin in config["coins"]:

            try:

                signal, score, price = analyze_coin(coin, config)

                print(
                    f"[{datetime.now()}] 币种:{coin} | 评分:{score} | 信号:{signal} | 当前价:{price}"
                )

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

        print("本轮扫描完成，等待下一次扫描...")

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
