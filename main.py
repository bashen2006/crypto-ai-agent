import time
import json
import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import os

CONFIG_FILE="config.json"
MEMORY_FILE="ai_memory.json"
LOG_FILE="prediction_log.json"

WHALE_THRESHOLD=20000
SIGNAL_COOLDOWN=1800

STATUS_PUSH_INTERVAL=300

last_signal_time={}
last_status_push=0

MAX_LOG_SIZE=2000
MAX_DYNAMIC_COINS=3


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_memory():

    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)

    except:

        memory={
            "trend_weight":0.3,
            "momentum_weight":0.25,
            "volume_weight":0.2
        }

        save_memory(memory)

        return memory


def save_memory(memory):

    with open(MEMORY_FILE,"w") as f:
        json.dump(memory,f,indent=4)


# =========================
# 新增：安全请求
# =========================

def safe_request(url):

    for i in range(3):

        try:

            r=requests.get(url,timeout=10)

            if r.status_code==200:

                return r.json()

        except:

            time.sleep(2)

    return None


def get_kline(inst):

    url=f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit=100"

    data=safe_request(url)

    if not data:
        raise Exception("行情请求失败")

    data=data["data"]

    df=pd.DataFrame(data)

    df=df.iloc[:,:6]

    df.columns=["ts","open","high","low","close","volume"]

    df=df.astype(float)

    return df


def detect_whale(inst):

    try:

        url=f"https://www.okx.com/api/v5/market/trades?instId={inst}&limit=100"

        data=safe_request(url)

        if not data:
            return 0

        data=data["data"]

        whale=0

        for trade in data:

            size=float(trade["sz"])
            price=float(trade["px"])

            value=size*price

            if value>WHALE_THRESHOLD:

                whale+=value

        return whale

    except:

        return 0


def calculate_score(df,memory,whale):

    df["ma20"]=df["close"].rolling(20).mean()
    df["ma60"]=df["close"].rolling(60).mean()

    ma20=df["ma20"].iloc[-1]
    ma60=df["ma60"].iloc[-1]

    momentum=(df["close"].iloc[-1]-df["close"].iloc[-10])/df["close"].iloc[-10]

    score=50
    factors={}

    if ma20>ma60:

        score+=20*memory["trend_weight"]
        factors["trend"]=1

    else:

        factors["trend"]=-1

    if momentum>0.02:

        score+=15*memory["momentum_weight"]
        factors["momentum"]=1

    else:

        factors["momentum"]=-1

    volume=df["volume"].iloc[-1]
    avg=df["volume"].mean()

    if volume>avg*1.5:

        score+=10*memory["volume_weight"]
        factors["volume"]=1

    else:

        factors["volume"]=0

    if whale>0:

        score+=5
        factors["whale"]=1

    score=max(0,min(100,int(score)))

    return score,factors


def market_risk_index():

    try:

        df=get_kline("BTC-USDT")

        df["ma200"]=df["close"].rolling(200).mean()

        price=df["close"].iloc[-1]
        ma200=df["ma200"].iloc[-1]

        if price<ma200:

            return "高风险"

        else:

            return "正常"

    except:

        return "未知"


def market_sentiment():

    try:

        url="https://www.okx.com/api/v5/market/tickers?instType=SPOT"

        data=safe_request(url)

        if not data:
            return "未知"

        data=data["data"]

        up=0
        down=0

        for c in data:

            try:

                if "chgUtc" in c:

                    change=float(c["chgUtc"])

                else:

                    last=float(c["last"])
                    open24=float(c["open24h"])

                    change=(last-open24)/open24

                if change>0:

                    up+=1
                else:
                    down+=1

            except:
                pass

        total=up+down

        if total==0:
            return "未知"

        ratio=up/total

        if ratio>0.7:
            return "极度贪婪"
        elif ratio>0.55:
            return "贪婪"
        elif ratio>0.45:
            return "中性"
        else:
            return "恐慌"

    except:

        return "未知"


def scan_hot_coins():

    coins=[]

    try:

        url="https://www.okx.com/api/v5/market/tickers?instType=SPOT"

        data=safe_request(url)

        if not data:
            return []

        data=data["data"]

        for c in data:

            try:

                if "chgUtc" in c:

                    change=float(c["chgUtc"])

                else:

                    last=float(c["last"])
                    open24=float(c["open24h"])

                    change=(last-open24)/open24

                if change>0.05:

                    coins.append(c["instId"])

            except:

                pass

        return coins[:MAX_DYNAMIC_COINS]

    except:

        return []


def hot_coin_filter(coin):

    try:

        df=get_kline(coin)

        change=(df["close"].iloc[-1]-df["close"].iloc[-30])/df["close"].iloc[-30]

        if change>0:

            return True

        else:

            return False

    except:

        return False


def send_telegram(msg,token,chat):

    url=f"https://api.telegram.org/bot{token}/sendMessage"

    data={"chat_id":chat,"text":msg}

    requests.post(url,data=data)


def send_email(subject,content,user,password,receiver):

    msg=MIMEText(content,"plain","utf-8")

    msg["Subject"]=subject
    msg["From"]=user
    msg["To"]=receiver

    server=smtplib.SMTP_SSL("smtp.139.com",465)

    server.login(user,password)

    server.sendmail(user,receiver,msg.as_string())

    server.quit()


# =========================
# 新增：5分钟状态推送
# =========================

def send_status_report(config,risk,sentiment,coin_count):

    now_str=time.strftime("%Y-%m-%d %H:%M:%S")

    msg=f"""
AI系统状态报告

监控币种数量: {coin_count}
市场风险: {risk}
市场情绪: {sentiment}

系统运行状态: 正常
时间: {now_str}
"""

    send_telegram(msg,
                  config["telegram_bot_token"],
                  config["telegram_chat_id"])

    send_email("AI系统状态报告",
               msg,
               config["email_user"],
               config["email_pass"],
               config["email_receiver"])


def main():

    global last_status_push

    config=load_config()

    print("AI交易系统启动")

    while True:

        try:

            risk=market_risk_index()
            sentiment=market_sentiment()

            hot=scan_hot_coins()

            coins=config["coins"].copy()

            for h in hot:

                if hot_coin_filter(h):

                    if h not in coins:

                        coins.append(h)

            memory=load_memory()

            for coin in coins:

                df=get_kline(coin)

                whale=detect_whale(coin)

                score,factors=calculate_score(df,memory,whale)

                price=df["close"].iloc[-1]

                if score>=config["buy_threshold"]:

                    signal="买入"

                elif score<=config["sell_threshold"]:

                    signal="卖出"

                else:

                    signal="中性"

                msg=f"""
币种: {coin}
信号: {signal}
AI评分: {score}
市场风险: {risk}
市场情绪: {sentiment}
"""

                print(msg)

            # 新增：5分钟状态推送
            now=time.time()

            if now-last_status_push>STATUS_PUSH_INTERVAL:

                send_status_report(config,risk,sentiment,len(coins))

                last_status_push=now

        except Exception as e:

            print("系统异常，自动恢复:",e)

        time.sleep(config["check_interval"])


if __name__=="__main__":

    main()
