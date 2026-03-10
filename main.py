import requests
import time
import json
from telegram import Bot

# ==============================
# 读取配置
# ==============================
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

coins = config.get("coins", ["bitcoin","ethereum","solana","okb"])
buy_threshold = config.get("buy_threshold", 70)
sell_threshold = config.get("sell_threshold", 30)
risk_check = config.get("risk_check", True)
check_interval = config.get("check_interval", 300)
notify_method = config.get("notify_method", "telegram")
telegram_bot_token = config.get("telegram_bot_token")
telegram_chat_id = config.get("telegram_chat_id")

# ==============================
# 安全初始化 Telegram
# ==============================
bot = None
if notify_method == "telegram":
    if telegram_bot_token and telegram_chat_id:
        try:
            bot = Bot(token=telegram_bot_token)
        except Exception as e:
            print(f"初始化 Telegram 失败: {e}")
            bot = None
    else:
        print("Telegram Token 或 Chat ID 为空，消息将打印在控制台")

# ==============================
# 获取价格函数（安全）
# ==============================
def get_price(coin):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        r = requests.get(url, timeout=10)
        data = r.json()
        if coin in data and "usd" in data[coin]:
            return data[coin]["usd"]
        else:
            return None
    except Exception as e:
        print(f"{coin} 获取价格失败: {e}")
        return None

# ==============================
# 多因子决策逻辑（模拟示例）
# ==============================
def analyze_coin(price):
    import random
    score = random.randint(0, 100)
    action = "HOLD"
    if score >= buy_threshold:
        action = "BUY"
    elif score <= sell_threshold:
        action = "SELL"
    return action, score

# ==============================
# 中文通知函数（保证发送到 Telegram 是中文）
# ==============================
def send_message(msg):
    if notify_method == "telegram" and bot:
        try:
            bot.send_message(chat_id=telegram_chat_id, text=str(msg))
        except Exception as e:
            print(f"发送 Telegram 消息失败: {e}")
    else:
        print(f"[通知] {msg}")

# ==============================
# 主循环
# ==============================
while True:
    print("------AI Crypto Monitor------")
    for coin in coins:
        price = get_price(coin)
        if price is None:
            print(f"{coin}：获取价格失败")
            continue
        action, score = analyze_coin(price)
        # 中文化显示
        action_text = {"BUY":"买入", "SELL":"卖出", "HOLD":"观望"}
        msg = f"{coin.upper()} 当前价格：${price:.2f} | 综合评分：{score or 0} | 建议操作：{action_text.get(action, '观望')}"
        print(msg)
        send_message(msg)
    print(f"下次检测在 {check_interval} 秒后\n")
    time.sleep(check_interval)
