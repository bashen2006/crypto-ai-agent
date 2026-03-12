import requests
import pandas as pd
import time
import json
from datetime import datetime

print("AI量化系统 V10 启动")

with open("config.json") as f:
    config=json.load(f)

coins=config["coins"]

BOT=config["telegram_bot_token"]
CHAT=config["telegram_chat_id"]

def send(msg):

    url=f"https://api.telegram.org/bot{BOT}/sendMessage"

    try:
        requests.post(url,data={"chat_id":CHAT,"text":msg})
    except:
        print("telegram error")


def get_kline(symbol):

    url=f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=5m&limit=200"

    r=requests.get(url)

    data=r.json()["data"]

    df=pd.DataFrame(data)

    df=df.iloc[::-1]

    df=df.astype(float)

    return df


def MA(series,n):

    return series.rolling(n).mean()


def RSI(series,period=14):

    delta=series.diff()

    gain=(delta.where(delta>0,0)).rolling(period).mean()

    loss=(-delta.where(delta<0,0)).rolling(period).mean()

    rs=gain/loss

    rsi=100-(100/(1+rs))

    return rsi


def MACD(series):

    exp1=series.ewm(span=12).mean()

    exp2=series.ewm(span=26).mean()

    macd=exp1-exp2

    signal=macd.ewm(span=9).mean()

    return macd,signal


def bollinger(series):

    ma=series.rolling(20).mean()

    std=series.rolling(20).std()

    upper=ma+2*std

    lower=ma-2*std

    return upper,lower


def bull_bear(ma50,ma100,ma200):

    if ma50>ma100 and ma100>ma200:

        return "牛市中 🟢"

    if ma50<ma100 and ma100<ma200:

        return "熊市中 🔴"

    return "牛熊转换中 🟡"


def score(price,ma7,ma30,ma50,ma100,ma200,rsi,macd,signal,upper,lower):

    s=50

    if ma7>ma30:
        s+=10

    if ma50>ma100:
        s+=10

    if price>ma200:
        s+=10

    if rsi<35:
        s+=10

    if rsi>70:
        s-=10

    if macd>signal:
        s+=10

    if price<lower:
        s+=10

    if price>upper:
        s-=10

    return max(0,min(100,s))


last_review=time.time()
last_day=time.time()

while True:

    print("开始新一轮检测")

    for coin in coins:

        try:

            df=get_kline(coin)

            close=df[4]

            price=float(close.iloc[-1])

            ma7=MA(close,7).iloc[-1]
            ma30=MA(close,30).iloc[-1]
            ma50=MA(close,50).iloc[-1]
            ma100=MA(close,100).iloc[-1]
            ma200=MA(close,200).iloc[-1]

            rsi=RSI(close).iloc[-1]

            macd,signal=MACD(close)

            macd=macd.iloc[-1]
            signal=signal.iloc[-1]

            upper,lower=bollinger(close)

            upper=upper.iloc[-1]
            lower=lower.iloc[-1]

            market=bull_bear(ma50,ma100,ma200)

            ai=score(price,ma7,ma30,ma50,ma100,ma200,rsi,macd,signal,upper,lower)

            if ai>=config["strong_buy"]:

                action="强买入"

            elif ai>=config["buy_threshold"]:

                action="建议买入"

            elif ai<=config["strong_sell"]:

                action="强卖出"

            elif ai<=config["sell_threshold"]:

                action="建议卖出"

            else:

                action="观望"

            msg=f"""
{coin}

价格: {round(price,2)}

市场状态: {market}

RSI: {round(rsi,2)}

AI评分: {ai}

AI建议:
{action}
"""

            print(msg)

            send(msg)

        except Exception as e:

            send(f"{coin} 数据异常")

            print(e)

        time.sleep(1)

    if time.time()-last_review>21600:

        send("AI策略6小时复盘：系统运行正常")

        last_review=time.time()

    if time.time()-last_day>86400:

        send("🔥AI策略24小时报告🔥")

        last_day=time.time()

    time.sleep(config["check_interval"])
