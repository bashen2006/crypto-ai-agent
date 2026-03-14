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

last_signal_time={}

MAX_LOG_SIZE=2000
MAX_DYNAMIC_COINS=3


# =========================
# 配置加载
# =========================

def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


# =========================
# AI权重
# =========================

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
# 新增：安全请求模块
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


# =========================
# OKX行情
# =========================

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


# =========================
# 巨鲸监控
# =========================

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


# =========================
# AI评分
# =========================

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
# 市场风险指数
# =========================

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


# =========================
# 市场情绪指数
# =========================

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


# =========================
# 强势币扫描
# =========================

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


# =========================
# 强势币过滤
# =========================

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


# =========================
# 日志优化（新增）
# =========================

def trim_logs():

    try:

        if not os.path.exists(LOG_FILE):
            return

        with open(LOG_FILE) as f:

            logs=json.load(f)

        if len(logs)>MAX_LOG_SIZE:

            logs=logs[-MAX_LOG_SIZE:]

            with open(LOG_FILE,"w") as f:

                json.dump(logs,f)

    except:

        pass


# =========================
# 记录预测
# =========================

def log_prediction(coin,price,signal,factors):

    try:
        with open(LOG_FILE) as f:
            logs=json.load(f)
    except:
        logs=[]

    logs.append({

        "coin":coin,
        "time":time.time(),
        "price":price,
        "signal":signal,
        "factors":factors,
        "result":None
    })

    with open(LOG_FILE,"w") as f:
        json.dump(logs,f)

    trim_logs()


# =========================
# 验证预测
# =========================

def evaluate_predictions():

    try:

        with open(LOG_FILE) as f:

            logs=json.load(f)

    except:

        return

    changed=False

    for r in logs:

        if r["result"]!=None:
            continue

        if time.time()-r["time"]<7200:
            continue

        df=get_kline(r["coin"])

        price=df["close"].iloc[-1]

        if price>r["price"]:
            r["result"]="correct"
        else:
            r["result"]="wrong"

        changed=True

    if changed:

        with open(LOG_FILE,"w") as f:
            json.dump(logs,f)


# =========================
# AI权重优化
# =========================

def optimize_weights():

    try:

        with open(LOG_FILE) as f:
            logs=json.load(f)

    except:
        return

    memory=load_memory()

    stats={"trend":[0,0],"momentum":[0,0],"volume":[0,0]}

    for r in logs:

        if r["result"]==None:
            continue

        for f in stats:

            if f in r["factors"]:

                stats[f][1]+=1

                if r["result"]=="correct":

                    stats[f][0]+=1

    for f in stats:

        if stats[f][1]>5:

            rate=stats[f][0]/stats[f][1]

            if rate>0.6:
                memory[f+"_weight"]+=0.01
            else:
                memory[f+"_weight"]-=0.01

    save_memory(memory)


# =========================
# 系统统计
# =========================

def system_stats():

    try:

        with open(LOG_FILE) as f:

            logs=json.load(f)

    except:

        return

    total=0
    correct=0

    for r in logs:

        if r["result"]!=None:

            total+=1

            if r["result"]=="correct":

                correct+=1

    if total>0:

        winrate=round(correct/total*100,2)

        print(f"系统统计 | 总预测:{total} | 胜率:{winrate}%")


# =========================
# 主程序
# =========================

def main():

    config=load_config()

    print("AI交易系统启动")

    while True:

        try:

            evaluate_predictions()
            optimize_weights()
            system_stats()

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

                now=time.time()

                if signal!="中性":

                    if coin in last_signal_time:

                        if now-last_signal_time[coin]<SIGNAL_COOLDOWN:

                            continue

                    send_telegram(msg,
                                  config["telegram_bot_token"],
                                  config["telegram_chat_id"])

                    send_email("AI交易信号",
                               msg,
                               config["email_user"],
                               config["email_pass"],
                               config["email_receiver"])

                    log_prediction(coin,price,signal,factors)

                    last_signal_time[coin]=now

        except Exception as e:

            print("系统异常，自动恢复:",e)

        time.sleep(config["check_interval"])


if __name__=="__main__":

    main()
