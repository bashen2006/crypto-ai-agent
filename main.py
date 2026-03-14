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
DAILY_REPORT_INTERVAL=86400
BACKTEST_INTERVAL=86400
ADAPTIVE_OPTIMIZATION_INTERVAL=86400

last_signal_time={}
last_status_push=0
last_daily_report=0
last_backtest_time=0
last_adaptive_time=0

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
# 安全请求
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


# =========================
# 新增：AI策略自适应优化
# =========================

def adaptive_strategy_optimization():

    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE) as f:
        logs=json.load(f)

    correct=0
    wrong=0

    for r in logs:

        if r.get("result")=="correct":
            correct+=1

        elif r.get("result")=="wrong":
            wrong+=1

    total=correct+wrong

    if total<10:
        return

    winrate=correct/total

    memory=load_memory()

    if winrate>0.6:

        memory["trend_weight"]=min(memory["trend_weight"]+0.02,1)
        memory["momentum_weight"]=min(memory["momentum_weight"]+0.02,1)
        memory["volume_weight"]=min(memory["volume_weight"]+0.02,1)

    elif winrate<0.4:

        memory["trend_weight"]=max(memory["trend_weight"]-0.02,0)
        memory["momentum_weight"]=max(memory["momentum_weight"]-0.02,0)
        memory["volume_weight"]=max(memory["volume_weight"]-0.02,0)

    save_memory(memory)

    print("AI策略自适应优化完成 当前权重:",memory)


# =========================
# AI自动回测
# =========================

def calculate_backtest_performance():

    if not os.path.exists(LOG_FILE):
        return 0,0

    with open(LOG_FILE) as f:

        logs=json.load(f)

    profit=0
    trades=0

    for r in logs:

        if r["result"]=="correct":
            profit+=1

        elif r["result"]=="wrong":
            profit-=1

        if r["result"]!=None:
            trades+=1

    return profit,trades


def send_backtest_report(config):

    profit,trades=calculate_backtest_performance()

    msg=f"""
AI策略自动回测报告

历史交易次数: {trades}
策略收益评分: {profit}

时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

    send_telegram(msg,
                  config["telegram_bot_token"],
                  config["telegram_chat_id"])

    send_email("AI策略回测报告",
               msg,
               config["email_user"],
               config["email_pass"],
               config["email_receiver"])


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
# 主程序
# =========================

def main():

    global last_backtest_time
    global last_adaptive_time

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

                if score>=config["buy_threshold"]:
                    signal="买入"
                elif score<=config["sell_threshold"]:
                    signal="卖出"
                else:
                    signal="中性"

                print(coin,signal,score)

            now=time.time()

            if now-last_backtest_time>BACKTEST_INTERVAL:

                send_backtest_report(config)

                last_backtest_time=now

            if now-last_adaptive_time>ADAPTIVE_OPTIMIZATION_INTERVAL:

                adaptive_strategy_optimization()

                last_adaptive_time=now

        except Exception as e:

            print("系统异常，自动恢复:",e)

        time.sleep(config["check_interval"])


if __name__=="__main__":

    main()
