import requests
import json


# =========================
# 读取配置
# =========================
def load_config():

    with open("config.json", "r") as f:

        return json.load(f)


# =========================
# 发送Telegram消息
# =========================
def send_message(message):

    config = load_config()

    token = config["telegram_bot_token"]

    chat_id = config["telegram_chat_id"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    data = {

        "chat_id": chat_id,

        "text": message

    }

    try:

        requests.post(url, data=data, timeout=10)

    except Exception as e:

        print("Telegram发送失败:", e)


# =========================
# 发送AI交易报告
# =========================
def send_trade_report(report):

    message = f"""

🚨 AI交易提醒 🚨

{report}

"""

    send_message(message)


# =========================
# 发送复盘报告
# =========================
def send_review_report(report):

    message = f"""

📊 AI策略复盘 📊

{report}

"""

    send_message(message)
