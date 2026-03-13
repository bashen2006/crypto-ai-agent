import json
import os
from datetime import datetime

LOG_FILE = "data/trade_log.json"


# =========================
# 加载日志
# =========================
def load_logs():

    if not os.path.exists(LOG_FILE):

        return []

    with open(LOG_FILE, "r") as f:

        return json.load(f)


# =========================
# 计算整体胜率
# =========================
def calculate_win_rate():

    data = load_logs()

    if len(data) == 0:

        return 0, 0, 0

    wins = 0
    loses = 0

    for entry in data:

        if entry["result"] == "win":

            wins += 1

        elif entry["result"] == "lose":

            loses += 1

    total = wins + loses

    if total == 0:

        return 0, wins, loses

    win_rate = wins / total

    return win_rate, wins, loses


# =========================
# 最近N次信号胜率
# =========================
def recent_performance(n=20):

    data = load_logs()

    if len(data) == 0:

        return 0

    recent = data[-n:]

    wins = 0
    total = 0

    for entry in recent:

        if entry["result"] is not None:

            total += 1

            if entry["result"] == "win":

                wins += 1

    if total == 0:

        return 0

    return wins / total


# =========================
# 生成复盘报告
# =========================
def generate_report():

    win_rate, wins, loses = calculate_win_rate()

    recent_rate = recent_performance()

    report = {

        "time": str(datetime.utcnow()),
        "total_wins": wins,
        "total_loses": loses,
        "overall_win_rate": win_rate,
        "recent_win_rate": recent_rate

    }

    return report


# =========================
# 打印复盘结果
# =========================
def print_report():

    report = generate_report()

    print("====== AI策略复盘 ======")

    print("时间:", report["time"])

    print("总胜率:", round(report["overall_win_rate"] * 100, 2), "%")

    print("近期胜率:", round(report["recent_win_rate"] * 100, 2), "%")

    print("胜利次数:", report["total_wins"])

    print("失败次数:", report["total_loses"])
