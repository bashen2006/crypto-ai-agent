import time
import json

from modules.data_source import get_price, get_kline
from modules.indicators import MA, RSI, MACD, Bollinger
from modules.derivatives import get_funding_rate, get_open_interest, get_long_short_ratio
from modules.onchain import get_whale_transactions
from modules.ai_engine import calculate_score, generate_signal
from modules.strategy import record_signal, update_results
from modules.learning import calculate_win_rate
from modules.report import generate_trade_report, generate_review_report
from modules.telegram_bot import send_trade_report, send_review_report


# =========================
# 读取配置
# =========================

def load_config():

    with open("config.json", "r") as f:

        return json.load(f)


config = load_config()

coins = config["coins"]

buy_threshold = config["buy_threshold"]

sell_threshold = config["sell_threshold"]

interval = config["check_interval"]


print("A12 AI量化系统启动...")

# =========================
# 主循环
# =========================

while True:

    try:

        for coin in coins:

            print("监控币种:", coin)

            # 获取价格
            price = get_price(coin)

            # 获取K线
            df = get_kline(coin)

            if df is None:

                continue

            close = df["close"]


            # =========================
            # 技术指标
            # =========================

            ma7 = MA(close, 7).iloc[-1]

            ma30 = MA(close, 30).iloc[-1]

            rsi = RSI(close).iloc[-1]

            macd, signal, hist = MACD(close)

            macd_val = macd.iloc[-1]

            macd_signal = signal.iloc[-1]

            upper, mid, lower = Bollinger(close)

            boll_upper = upper.iloc[-1]

            boll_lower = lower.iloc[-1]


            indicators = {

                "price": price,

                "ma7": ma7,

                "ma30": ma30,

                "rsi": rsi,

                "macd": macd_val,

                "macd_signal": macd_signal,

                "boll_upper": boll_upper,

                "boll_lower": boll_lower

            }


            # =========================
            # 衍生品数据
            # =========================

            funding_rate = get_funding_rate(coin)

            open_interest = get_open_interest(coin)

            long_short_ratio = get_long_short_ratio(coin)


            derivatives = {

                "funding_rate": funding_rate,

                "open_interest": open_interest,

                "long_short_ratio": long_short_ratio

            }


            # =========================
            # 巨鲸监控
            # =========================

            whales = get_whale_transactions()


            # =========================
            # AI评分
            # =========================

            score = calculate_score(indicators, derivatives, whales)

            signal_text = generate_signal(score)


            print("AI评分:", score)

            print("AI建议:", signal_text)


            # =========================
            # 记录信号
            # =========================

            record_signal(coin, price, score, signal_text)


            # =========================
            # 更新策略结果
            # =========================

            update_results(price)


            # =========================
            # 生成交易报告
            # =========================

            report = generate_trade_report(coin, price, score, signal_text)

            print(report)


            # =========================
            # 发送Telegram
            # =========================

            send_trade_report(report)


            # =========================
            # 策略复盘
            # =========================

            win_rate, wins, loses = calculate_win_rate()

            review = generate_review_report(win_rate, wins, loses)

            send_review_report(review)


        print("等待下一轮监控...")

        time.sleep(interval)

    except Exception as e:

        print("系统错误:", e)

        time.sleep(60)
