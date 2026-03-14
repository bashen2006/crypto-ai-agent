import time
import json
import requests
import pandas as pd

CONFIG_FILE = "config.json"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def get_okx_kline(inst):
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit=50"
    r = requests.get(url)
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df


def calculate_score(df):
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()

    ma7 = df["ma7"].iloc[-1]
    ma30 = df["ma30"].iloc[-1]

    score = 50

    if ma7 > ma30:
        score += 20
    else:
        score -= 20

    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()

    if volume > avg_volume * 1.5:
        score += 10

    return score


def send_telegram(message, token, chat_id):

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=payload)


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


def main():

    config = load_config()

    print("AI Agent Started")

    while True:

        for coin in config["coins"]:

            try:

                signal, score, price = analyze_coin(coin, config)

                print(coin, signal, score)

                if signal != "NEUTRAL":

                    msg = f"""
AI交易信号

币种: {coin}

评分: {score}

价格: {price}

信号: {signal}
"""

                    send_telegram(
                        msg,
                        config["telegram_bot_token"],
                        config["telegram_chat_id"]
                    )

            except Exception as e:

                print("error:", e)

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
