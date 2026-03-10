import requests
import time
import json
from telegram import Bot

# ==============================
# 读取配置
# ==============================
with open("config.json", "r") as f:
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
# 初始化 Telegram
# ==============================
bot = None
if notify_method == "telegram" and telegram_bot_token and telegram_chat_id:
    bot = Bot(token=telegram_bot_token)

# ==============================
# 获取价格函数
# ==============================
def get_price(coin):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        r = requests.get(url, timeout=10)
        data = r.json()
        return data[coin]["usd"]
    except:
        return None

# ==============================
# 多因子决策逻辑（模拟示例）
# ==============================
def analyze_coin(price):
    """
    这里是高级模型占位：
    - 技术指标：简单用价格变化模拟
    - 市场情绪：模拟情绪指数
    - 链上巨鲸监控：随机生成
    """
    # 模拟因子评分 0-100
    import random
    score = random.randint(0, 100)
    action = "HOLD"
    if score >= buy_threshold:
        action = "BUY"
    elif score <= sell_threshold:
        action = "SELL"
    return action, score

# ==============================
# 通知函数
# ==============================
def send_message(msg):
    if notify_method == "telegram" and bot:
        bot.send_message(chat_id=telegram_chat_id, text=msg)
    else:
        print(f"[通知] {msg}")  # 邮件或微信可扩展

# ==============================
# 主循环
# ==============================
while True:
    print("------AI Crypto Monitor------")
    for coin in coins:
        price = get_price(coin)
        if price is None:
            print(f"{coin}: 获取价格失败")
            continue
        action, score = analyze_coin(price)
        msg = f"{coin.upper()} | Price: ${price:.2f} | Score: {score} | Action: {action}"
        print(msg)
        send_message(msg)
    print(f"下次检测在 {check_interval} 秒后\n")
    time.sleep(check_interval)
