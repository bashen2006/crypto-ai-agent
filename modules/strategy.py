import json
import os
from datetime import datetime


LOG_FILE = "data/trade_log.json"


# =========================
# 初始化日志文件
# =========================

def init_log():

    if not os.path.exists("data"):

        os.makedirs("data")

    if not os.path.exists(LOG_FILE):

        with open(LOG_FILE, "w") as f:

            json.dump([], f)


# =========================
# 记录信号
# =========================

def record_signal(coin, price, score, signal):

    init_log()

    with open(LOG_FILE, "r") as f:

        data = json.load(f)


    entry = {

        "coin": coin,
        "time": str(datetime.utcnow()),
        "price": price,
        "score": score,
        "signal": signal,
        "result": None

    }


    data.append(entry)


    with open(LOG_FILE, "w") as f:

        json.dump(data, f, indent=2)


# =========================
# 更新信号结果
# =========================

def update_results(current_price):

    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    updated = False

    for entry in data:

        # 已经判断过的不再改变
        if entry["result"] is not None:
            continue

        old_price = entry["price"]

        if entry["signal"] in ["强买入", "建议买入"]:

            if current_price > old_price:
                entry["result"] = "win"
            else:
                entry["result"] = "lose"

            updated = True

        elif entry["signal"] in ["强卖出", "建议卖出"]:

            if current_price < old_price:
                entry["result"] = "win"
            else:
                entry["result"] = "lose"

            updated = True

    if updated:
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)

        json.dump(data, f, indent=2)
