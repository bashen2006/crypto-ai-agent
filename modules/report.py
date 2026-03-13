from datetime import datetime


# =========================
# 市场状态判断
# =========================
def market_status(score):

    if score >= 75:

        return "强牛趋势"

    elif score >= 60:

        return "牛市趋势"

    elif score <= 25:

        return "强熊趋势"

    elif score <= 40:

        return "熊市趋势"

    else:

        return "震荡市场"


# =========================
# 生成交易报告
# =========================
def generate_trade_report(coin, price, score, signal):

    report = f"""
============================

AI交易报告

币种: {coin}

时间: {datetime.utcnow()}

当前价格: {price}

AI评分: {score}

AI建议: {signal}

市场状态: {market_status(score)}

============================
"""

    return report


# =========================
# 生成复盘报告
# =========================
def generate_review_report(win_rate, wins, loses):

    report = f"""
============================

AI策略复盘报告

胜利次数: {wins}

失败次数: {loses}

策略胜率: {round(win_rate * 100,2)} %

============================
"""

    return report
