import time
import json
import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import shutil
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from functools import wraps
import threading

# =========================
# Railway Volume 挂载等待
# 解决：代码启动比Volume挂载快，导致读不到持久化目录
# =========================
def wait_for_volume(path, max_wait=60):
    """
    等待Railway Volume挂载完成并验证可读写。
    Railway挂载Volume需要几秒，代码启动太快会读不到。
    max_wait=60：最多等60秒，足够覆盖任何正常挂载延迟。
    """
    print(f"⏳ 等待存储挂载: {path}")
    for i in range(max_wait):
        if os.path.exists(path):
            # 不只检查目录存在，还要验证真正可写
            test_file = os.path.join(path, ".mount_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("ok")
                os.remove(test_file)
                print(f"✅ 存储挂载成功（等待了 {i} 秒）: {path}")
                return True
            except Exception as e:
                print(f"  第{i+1}秒: 目录存在但不可写: {e}")
        else:
            if i % 10 == 0:
                print(f"  第{i+1}秒: 挂载等待中...")
        time.sleep(1)
    print(f"❌ 存储挂载超时（{max_wait}秒），使用本地临时目录")
    return False

# =========================
# 确定数据目录
# =========================
_RAILWAY_VOLUME = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/app/data")
_volume_ok      = wait_for_volume(_RAILWAY_VOLUME, max_wait=60)

if _volume_ok:
    DATA_DIR = _RAILWAY_VOLUME
else:
    # Volume不可用时使用本地目录，但明确告警
    DATA_DIR = "./data"
    print("⚠️ 警告：使用本地临时目录，重启后数据将全部丢失！")
    print("⚠️ 请检查云服务器存储卷是否正确挂载到 /app/data")

os.makedirs(DATA_DIR, exist_ok=True)
print(f"📁 数据存储目录: {DATA_DIR}")

# =========================
# 机器学习库导入
# =========================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    BAYES_OPT_AVAILABLE = True
except ImportError:
    BAYES_OPT_AVAILABLE = False
    print("提示: 未安装 scikit-optimize，贝叶斯超参优化功能已禁用。")

# =========================
# 配置常量
# =========================
CONFIG_FILE        = "config.json"
MEMORY_FILE        = "ai_memory.json"
LOG_FILE           = "prediction_log.json"
MODEL_FILE         = "ai_model.pkl"
SCALER_FILE        = "scaler.pkl"
FEATURES_FILE      = "feature_config.json"
SIGNAL_CONFIRM_FILE= "signal_confirm.json"
# 修改二：新增定时状态持久化文件，防止重启重复发送
TIMING_FILE        = "timing_state.json"
SIGNAL_TIME_FILE   = "signal_time.json"   # 信号冷却时间持久化
FEATURES_LOG_FILE  = "features_log.json"  # 特征数据独立存储（防止主日志过大导致OOM）

MEMORY_PATH        = os.path.join(DATA_DIR, MEMORY_FILE)
LOG_PATH           = os.path.join(DATA_DIR, LOG_FILE)
MODEL_PATH         = os.path.join(DATA_DIR, MODEL_FILE)
SCALER_PATH        = os.path.join(DATA_DIR, SCALER_FILE)
FEATURES_PATH      = os.path.join(DATA_DIR, FEATURES_FILE)
SIGNAL_CONFIRM_PATH= os.path.join(DATA_DIR, SIGNAL_CONFIRM_FILE)
TIMING_PATH        = os.path.join(DATA_DIR, TIMING_FILE)
SIGNAL_TIME_PATH   = os.path.join(DATA_DIR, SIGNAL_TIME_FILE)
FEATURES_LOG_PATH  = os.path.join(DATA_DIR, FEATURES_LOG_FILE)

# 时间间隔
STATUS_PUSH_INTERVAL          = 300    # 5分钟检查一次（有变化才发）
FORCE_STATUS_INTERVAL         = 1800   # 30分钟强制发一次（无论有无变化）
DAILY_REPORT_INTERVAL         = 86400
BACKTEST_INTERVAL             = 86400
ADAPTIVE_OPTIMIZATION_INTERVAL= 3600
MARKET_CYCLE_INTERVAL         = 3600

# 修改三：按市场周期动态冷却时间（秒）
# 升级为15m+1h+4h三周期系统后，信号质量提高，冷却时间同步放大
# 牛市趋势明确，使用180分钟（趋势冷却）
# 震荡行情，使用90分钟（震荡冷却）
# 熊市保守，使用180分钟（避免频繁抄底）
COOLDOWN_BY_CYCLE = {
    "牛市": 10800,  # 180分钟（趋势市）
    "震荡": 5400,   # 90分钟（震荡市）
    "熊市": 10800,  # 180分钟（熊市保守）
    "未知": 5400    # 默认90分钟
}

# 数据限制
MAX_LOG_SIZE          = 5000
MAX_DYNAMIC_COINS     = 3
MIN_TRAIN_SAMPLES     = 80
SCORE_CHANGE_THRESHOLD= 10

# Gate.io
GATEIO_BASE_URL = "https://api.gateio.ws/api/v4"

# 过滤阈值
VOLUME_RATIO_MIN  = 2.0
VOLATILITY_MIN    = 0.01
PROFIT_THRESHOLD  = 0.005  # 降低到0.5%（原1.5%在熊市太难达到，导致全部错误无法训练）

# 多档信号阈值
SIGNAL_STRONG_BUY  = 75
SIGNAL_BUY         = 60
SIGNAL_STRONG_SELL = 25
SIGNAL_SELL        = 40

# 信号连续确认次数
SIGNAL_CONFIRM_COUNT = 2

# 修改一：主流币集合（与BTC高度联动）
MAJOR_COINS = {"BTC", "ETH", "OKB", "BNB", "SOL", "XRP", "ADA", "DOGE"}

# =========================
# 修复一：有限容量缓存，防止OOM
# =========================
def cache(ttl_seconds=60, max_size=256):
    def decorator(func):
        cache_store = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            with lock:
                entry = cache_store.get(key)
                if entry and now - entry['timestamp'] < ttl_seconds:
                    return entry['value']
            result = func(*args, **kwargs)
            with lock:
                cache_store[key] = {'value': result, 'timestamp': now}
                if len(cache_store) > max_size:
                    oldest = min(cache_store, key=lambda k: cache_store[k]['timestamp'])
                    del cache_store[oldest]
            return result
        return wrapper
    return decorator

# =========================
# 全局状态
# =========================
last_signal_time    = {}
last_status_push    = 0
last_daily_report   = 0
last_backtest_time  = 0
last_adaptive_time  = 0
last_cycle_check    = 0
current_market_cycle= "未知"
last_scores         = {}
btc_features_cache  = {'timestamp': 0, 'features': None}
start_time          = time.time()
# 历史评分缓存：用于百分位数动态阈值计算
# 每轮循环把所有监控币种的评分存入，最多保留500条
_score_history      = []   # 格式：[score, score, ...]
_SCORE_HISTORY_MAX  = 500  # 最多保留500条

# =========================
# 百分位数动态阈值
# 核心思路：无论牛熊震荡，始终取最强20%信号作为买入阈值
#           最弱20%信号作为卖出阈值，完全自适应市场
# =========================
def calc_dynamic_threshold(score_history, default_buy=62, default_sell=38):
    """
    用最近500条历史评分的百分位数动态计算阈值。
    数据不足20条时使用默认值，避免样本太少导致阈值失真。

    牛市：评分整体高 → 阈值自动升高 → 只选最强信号
    熊市：评分整体低 → 阈值自动降低 → 熊市反弹也能捕捉
    震荡：评分均匀分布 → 阈值适中   → 频率稳定
    """
    if len(score_history) < 20:
        return default_buy, default_sell

    scores = np.array(score_history[-500:])  # 最近500条

    # 第80百分位作为买入阈值（最强20%才触发）
    # 第20百分位作为卖出阈值（最弱20%才触发）
    buy_thr  = float(np.percentile(scores, 80))
    sell_thr = float(np.percentile(scores, 20))

    # 安全边界：防止极端行情下阈值过高或过低
    buy_thr  = max(min(round(buy_thr),  70), 55)  # 买入：55-70之间
    sell_thr = max(min(round(sell_thr), 45), 30)  # 卖出：30-45之间

    # 保证买卖之间至少10分间距
    if buy_thr - sell_thr < 10:
        mid      = (buy_thr + sell_thr) / 2
        buy_thr  = int(mid + 5)
        sell_thr = int(mid - 5)

    return buy_thr, sell_thr

# =========================
# 修改二：定时状态持久化
# 重启后从文件恢复上次发送时间，避免重复推送
# =========================
def load_timing_state():
    """从文件加载定时状态，重启后不会归零"""
    try:
        with open(TIMING_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {
            'last_backtest_time': 0,
            'last_daily_report':  0,
            'last_status_push':   0
        }

def save_timing_state(state):
    """原子保存定时状态"""
    tmp = TIMING_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        shutil.move(tmp, TIMING_PATH)
    except Exception as e:
        print(f"⚠️ 定时状态保存失败: {e}")

# =========================
# last_signal_time 持久化
# 解决：重启后新旧容器并发，各自内存独立导致同一信号发两次
# =========================
def load_signal_time():
    """从文件恢复信号冷却时间，重启后不丢失"""
    try:
        with open(SIGNAL_TIME_PATH, encoding='utf-8') as f:
            data = json.load(f)
        # 过滤掉过期的记录（超过最长冷却时间3600秒），避免加载太旧的数据
        now = time.time()
        max_cooldown = max(COOLDOWN_BY_CYCLE.values())
        return {k: v for k, v in data.items() if now - v < max_cooldown}
    except Exception:
        return {}

def save_signal_time(signal_times):
    """原子保存信号冷却时间"""
    tmp = SIGNAL_TIME_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(signal_times, f, indent=2)
        shutil.move(tmp, SIGNAL_TIME_PATH)
    except Exception as e:
        print(f"⚠️ 信号冷却时间保存失败: {e}")

# =========================
# 工具函数
# =========================
def format_gateio_symbol(inst):
    return inst.replace("-", "_").upper()

@cache(ttl_seconds=2, max_size=64)
def safe_request(url, params=None, max_retries=3):
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            print(f"网络请求失败，状态码 {r.status_code}: {url}")
        except Exception as e:
            print(f"网络请求异常: {e}")
            time.sleep(2)
    return None

@cache(ttl_seconds=5, max_size=32)
def get_ticker(inst):
    symbol = format_gateio_symbol(inst)
    url    = f"{GATEIO_BASE_URL}/spot/tickers"
    data   = safe_request(url, params={"currency_pair": symbol})
    if data and isinstance(data, list) and len(data) > 0:
        return float(data[0]['last'])
    return None

@cache(ttl_seconds=60, max_size=64)
def get_kline(inst, interval="15m", limit=200):
    symbol = format_gateio_symbol(inst)
    url    = f"{GATEIO_BASE_URL}/spot/candlesticks"
    data   = safe_request(url, params={"currency_pair": symbol,
                                        "interval": interval, "limit": limit})
    if not isinstance(data, list):
        raise Exception(f"获取{inst} K线失败: {data}")
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "volume", "close", "high", "low", "open"]
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
    df["ts"] = df["ts"] * 1000
    return df

@cache(ttl_seconds=60, max_size=8)
def scan_hot_coins(limit=20):
    url  = f"{GATEIO_BASE_URL}/spot/tickers"
    data = safe_request(url)
    if not isinstance(data, list):
        return []
    pairs = [d for d in data if d["currency_pair"].endswith("_USDT")]
    pairs.sort(key=lambda x: float(x["change_percentage"]), reverse=True)
    return [(p["currency_pair"].replace("_", "-"), float(p["change_percentage"]))
            for p in pairs[:limit]]

def hot_coin_filter(coin_name):
    return coin_name not in ["BTC-USDT", "ETH-USDT", "OKB-USDT"]

def detect_whale(inst):
    return 0

@cache(ttl_seconds=300, max_size=4)
def get_market_sentiment():
    url  = f"{GATEIO_BASE_URL}/spot/tickers"
    data = safe_request(url)
    if not isinstance(data, list):
        return None
    pairs = [d for d in data if d["currency_pair"].endswith("_USDT")]
    up    = sum(1 for d in pairs if float(d["change_percentage"]) > 0)
    down  = len(pairs) - up
    top5  = sorted(pairs, key=lambda x: float(x["quote_volume"]), reverse=True)[:5]
    avg_change = sum(float(d["change_percentage"]) for d in top5) / 5 if top5 else 0
    return {
        'market_up_count':        up,
        'market_down_count':      down,
        'market_up_down_ratio':   up / down if down > 0 else up,
        'market_top_vol_avg_change': avg_change
    }

@cache(ttl_seconds=3600, max_size=16)
def get_funding_rate(inst):
    WHITELIST = {"BTC", "ETH", "OKB", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "LINK"}
    base = inst.split("-")[0].upper()
    if base not in WHITELIST:
        return None
    data = safe_request(f"{GATEIO_BASE_URL}/futures/usdt/contracts/{format_gateio_symbol(inst)}")
    if data and isinstance(data, dict) and 'funding_rate' in data:
        return float(data['funding_rate'])
    return None

# =========================
# ATR计算（用于止损止盈）
# =========================
def calculate_atr(df, period=14):
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    tr_list = []
    for i in range(1, len(df)):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0
    return np.mean(tr_list[-period:])

# =========================
# 修改一：BTC主导度判断
# 用于区分"BTC独立行情"和"普跌行情"
# =========================
@cache(ttl_seconds=60, max_size=4)
def get_btc_dominance_trend():
    """
    判断BTC当前走势类型：
    - btc_independent: BTC涨幅明显高于大盘均值（资金虹吸效应，山寨币可能被抽血）
    - btc_falling:     BTC整体处于下跌趋势（均线死叉）
    - btc_change:      BTC近1小时涨跌幅
    - market_avg:      大盘平均涨跌幅
    """
    try:
        df_btc     = get_kline("BTC-USDT", interval="15m", limit=60)
        btc_change = (df_btc['close'].iloc[-1] - df_btc['close'].iloc[-4]) \
                     / df_btc['close'].iloc[-4]   # 近1小时涨跌幅（4根15分钟K线）

        # 大盘平均涨跌幅
        sentiment  = get_market_sentiment()
        market_avg = sentiment['market_top_vol_avg_change'] / 100 if sentiment else 0

        # BTC涨幅比大盘高1.5%以上 → BTC独立行情，资金虹吸，山寨币暂缓
        btc_independent = (btc_change - market_avg) > 0.015

        # BTC均线死叉 → 整体下跌趋势
        ma20 = df_btc['close'].rolling(20).mean().iloc[-1]
        ma60 = df_btc['close'].rolling(60).mean().iloc[-1]
        btc_falling = ma20 < ma60

        return {
            'btc_change':     btc_change,
            'market_avg':     market_avg,
            'btc_independent':btc_independent,
            'btc_falling':    btc_falling
        }
    except Exception as e:
        print(f"BTC市场主导度判断失败: {e}")
        return {
            'btc_change':     0,
            'market_avg':     0,
            'btc_independent':False,
            'btc_falling':    False
        }

# =========================
# 信号连续确认状态管理
# =========================
def load_signal_confirm():
    try:
        with open(SIGNAL_CONFIRM_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_signal_confirm(data):
    tmp = SIGNAL_CONFIRM_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, SIGNAL_CONFIRM_PATH)
    except Exception as e:
        print(f"⚠️ 信号确认状态保存失败: {e}")

# 内存缓存：signal_confirm 不再每次读写文件
# 进程内共享，重启时从文件恢复一次
_signal_confirm_cache = None

def _get_confirm_cache():
    global _signal_confirm_cache
    if _signal_confirm_cache is None:
        _signal_confirm_cache = load_signal_confirm()
    return _signal_confirm_cache

def check_signal_confirm(coin, signal_type, score):
    confirms = _get_confirm_cache()
    entry    = confirms.get(coin, {'signal': None, 'count': 0, 'scores': []})
    if entry['signal'] != signal_type:
        entry = {'signal': signal_type, 'count': 1, 'scores': [score]}
    else:
        entry['count'] += 1
        entry['scores'].append(score)
        entry['scores'] = entry['scores'][-SIGNAL_CONFIRM_COUNT:]
    confirms[coin] = entry
    # 每5次写一次文件，减少I/O（重启后最多丢失5次确认记录，影响极小）
    if sum(e.get('count', 0) for e in confirms.values()) % 5 == 0:
        save_signal_confirm(confirms)
    if entry['count'] >= SIGNAL_CONFIRM_COUNT:
        return True, float(np.mean(entry['scores']))
    return False, score

def reset_signal_confirm(coin):
    confirms = _get_confirm_cache()
    if coin in confirms:
        del confirms[coin]
        save_signal_confirm(confirms)

# =========================
# AI 模型类（含过拟合防护）
# =========================
class AITradingModel:
    def __init__(self):
        self.model              = None
        self.scaler             = StandardScaler()
        self.feature_importance = {}
        self.training_history   = []
        self.is_trained         = False
        self.feature_names      = []
        self.cv_accuracy        = 0.0

    def extract_features(self, df, whale, market_cycle, coin_name):
        features = {}
        if len(df) < 100:
            return None

        close  = df['close'].values
        volume = df['volume'].values

        for period in [5, 10, 20, 30, 50, 60, 100]:
            if len(df) >= period:
                ma = df['close'].rolling(period).mean().iloc[-1]
                features[f'ma_{period}']            = ma
                features[f'price_ma_{period}_ratio']= close[-1] / ma if ma != 0 else 1

        for period in [5, 10, 20, 30]:
            if len(df) >= period:
                features[f'momentum_{period}'] = (close[-1] - close[-period]) / close[-period]

        for period in [10, 20, 30]:
            if len(df) >= period:
                returns = np.diff(close[-period:]) / close[-period:-1]
                features[f'volatility_{period}'] = np.std(returns)

        vol_ma = pd.Series(volume).rolling(20).mean().iloc[-1]
        features['volume_ratio'] = volume[-1] / vol_ma if vol_ma != 0 else 1
        features['volume_trend'] = (volume[-1] - volume[-5]) / volume[-5] if volume[-5] != 0 else 0

        h20 = np.max(close[-20:])
        l20 = np.min(close[-20:])
        features['price_position_20'] = (close[-1] - l20) / (h20 - l20) if h20 > l20 else 0.5

        delta    = np.diff(close)
        gain     = np.where(delta > 0, delta, 0)
        loss     = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1] if len(gain) >= 14 else 0
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1] if len(loss) >= 14 else 0
        features['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 50

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig  = macd.ewm(span=9, adjust=False).mean()
        features['macd']           = macd.iloc[-1]
        features['macd_signal']    = sig.iloc[-1]
        features['macd_histogram'] = macd.iloc[-1] - sig.iloc[-1]

        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        bbu = (sma + std * 2).iloc[-1]
        bbl = (sma - std * 2).iloc[-1]
        features['bb_position'] = (close[-1] - bbl) / (bbu - bbl) if bbu > bbl else 0.5

        features['atr_ratio']    = calculate_atr(df) / close[-1] if close[-1] != 0 else 0
        features['whale_value']  = whale / 10000
        features['market_cycle'] = 2 if market_cycle == "牛市" else (1 if market_cycle == "震荡" else 0)
        # coin_id 已删除（md5哈希对模型无实际参考价值）

        try:
            df_h1 = get_kline(coin_name, interval="1h", limit=100)
            if df_h1 is not None and len(df_h1) >= 60:
                ma20h1 = df_h1['close'].rolling(20).mean().iloc[-1]
                features['h1_ma20']            = ma20h1
                features['h1_price_ma20_ratio']= close[-1] / ma20h1 if ma20h1 != 0 else 1
                features['h1_momentum_10']     = (df_h1['close'].iloc[-1] - df_h1['close'].iloc[-10]) \
                                                  / df_h1['close'].iloc[-10]
        except Exception:
            pass

        global btc_features_cache
        now = time.time()
        if now - btc_features_cache['timestamp'] > 60:
            try:
                df_btc = get_kline("BTC-USDT", interval="15m", limit=100)
                if df_btc is not None and len(df_btc) >= 60:
                    ma20b = df_btc['close'].rolling(20).mean().iloc[-1]
                    btc_features_cache['features'] = {
                        'btc_ma20':             ma20b,
                        'btc_price_ma20_ratio': df_btc['close'].iloc[-1] / ma20b if ma20b != 0 else 1,
                        'btc_momentum_10':      (df_btc['close'].iloc[-1] - df_btc['close'].iloc[-10])
                                                / df_btc['close'].iloc[-10]
                    }
                    btc_features_cache['timestamp'] = now
            except Exception:
                pass
        if btc_features_cache['features']:
            features.update(btc_features_cache['features'])

        # 注意：已删除 hour/weekday/is_weekend/coin_id 四个无效特征
        # 原因：加密货币24小时全年无休，时间特征意义不大
        #       coin_id 用md5哈希，对模型没有实际参考价值
        # 效果：特征从47个降到43个，减少过拟合风险

        sentiment = get_market_sentiment()
        if sentiment:
            features.update(sentiment)

        funding = get_funding_rate(coin_name)
        if funding is not None:
            features['funding_rate'] = funding

        return features

    # 需要从特征中过滤掉的无效特征
    # 这里统一定义，方便后续维护
    DROPPED_FEATURES = {'hour', 'weekday', 'is_weekend', 'coin_id'}

    def prepare_training_data(self, logs):
        X_list, y_list = [], []
        # 加载features独立文件，用于还原特征数据
        features_data  = load_features_log()
        for log in logs:
            if not log.get('verified') or log.get('result') not in ['correct', 'wrong']:
                continue
            # 优先从独立文件取features（新格式），兼容旧格式（直接含features字段）
            features = None
            if log.get('feature_id') and log['feature_id'] in features_data:
                features = features_data[log['feature_id']]
            elif log.get('features'):
                features = log['features']  # 兼容旧格式
            if not features:
                continue
            # 过滤掉无效特征（兼容旧数据，新数据已不包含这些特征）
            features = {k: v for k, v in features.items()
                       if k not in self.DROPPED_FEATURES}
            X_list.append(features)
            y_list.append(1 if log['result'] == 'correct' else 0)
        if len(X_list) < MIN_TRAIN_SAMPLES:
            return None, None
        X_df = pd.DataFrame(X_list)
        # ⚠️ 不在此处更新 feature_names！
        # feature_names 只能在 train() 成功完成后才更新。
        return X_df, np.array(y_list)

    def train(self, X, y):
        if X is None or len(X) < MIN_TRAIN_SAMPLES:
            print(f"训练样本不足，当前: {len(X) if X is not None else 0} 条，需要: {MIN_TRAIN_SAMPLES} 条")
            return False

        if isinstance(X, pd.DataFrame):
            orig   = len(X)
            X      = X.dropna()
            # dropna后index不连续，需先对齐y再重置index
            y      = np.array(y)[X.index.tolist()]
            X      = X.reset_index(drop=True)
            if len(X) < MIN_TRAIN_SAMPLES:
                print(f"清洗NaN后样本不足 {MIN_TRAIN_SAMPLES} 条，跳过本次训练")
                return False
            if len(X) < orig:
                print(f"已清除 {orig - len(X)} 条含NaN的样本")

        # 检查标签多样性：必须同时有correct和wrong样本
        # 如果全部是一种标签，sklearn会报ValueError崩溃
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            label_name = "全部正确" if unique_labels[0] == 1 else "全部错误"
            print(f"⚠️ 训练标签只有一种类型（{label_name}），无法训练，跳过本次")
            print(f"   原因：胜率为0%或100%时模型无法学习，需等待更多样化的信号")
            return False

        # ✅ 到这里才真正开始训练，此时同步更新 feature_names
        # scaler 也会在下面 fit_transform 中重新拟合，保证两者始终匹配
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        print(f"开始训练模型，特征数量: {len(self.feature_names)}, 正确率: {np.mean(y):.1%}")

        X_scaled = self.scaler.fit_transform(X)

        base_models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, subsample=0.7,
                min_samples_leaf=5, random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=4,
                min_samples_leaf=5, max_features='sqrt',
                random_state=42
            ))
        ]

        model = VotingClassifier(estimators=base_models, voting='soft')
        model.fit(X_scaled, y)

        cv        = StratifiedKFold(n_splits=min(5, len(X) // 10), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_acc    = float(np.mean(cv_scores))
        train_acc = accuracy_score(y, model.predict(X_scaled))

        overfit_gap = train_acc - cv_acc
        if overfit_gap > 0.2:
            print(f"⚠️ 过拟合警告: 训练准确率={train_acc:.3f}，交叉验证准确率={cv_acc:.3f}，差距={overfit_gap:.3f}")

        self.model       = model
        self.cv_accuracy = cv_acc

        if hasattr(model, 'estimators_'):
            imps = [est.feature_importances_
                    for _, est in base_models
                    if hasattr(est, 'feature_importances_')]
            if imps:
                self.feature_importance = dict(zip(self.feature_names, np.mean(imps, axis=0)))

        self.is_trained = True
        self.training_history.append({
            'timestamp':   time.time(),
            'samples':     len(X),
            'train_acc':   train_acc,
            'cv_accuracy': cv_acc,
            'overfit_gap': overfit_gap,
            'features':    len(self.feature_names)
        })

        print(f"✅ 模型训练完成: 样本数={len(X)}，训练准确率={train_acc:.4f}，交叉验证准确率={cv_acc:.4f}")
        if self.feature_importance:
            top5 = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"前5重要特征: {top5}")
        return True

    def predict(self, features):
        if not self.is_trained or self.model is None:
            return 50, {}
        fv    = [features.get(name, 0) for name in self.feature_names]
        fs    = self.scaler.transform([fv])
        proba = self.model.predict_proba(fs)[0]
        score = proba[1] * 100 if len(proba) == 2 else 50
        confidence = float(np.max(proba))
        contrib    = {}
        if hasattr(self.model, 'estimators_'):
            est = self.model.estimators_[0]
            if hasattr(est, 'feature_importances_'):
                for i, name in enumerate(self.feature_names):
                    contrib[name] = est.feature_importances_[i] * fs[0][i]
        return score, {'confidence': confidence, 'contribution': contrib}

    # 修复二：原子保存
    def save(self):
        if not self.model:
            return False
        tmp_m = MODEL_PATH    + ".tmp"
        tmp_s = SCALER_PATH   + ".tmp"
        tmp_f = FEATURES_PATH + ".tmp"
        try:
            joblib.dump(self.model,  tmp_m)
            joblib.dump(self.scaler, tmp_s)
            with open(tmp_f, 'w', encoding='utf-8') as f:
                json.dump({
                    'feature_names':      self.feature_names,
                    'training_history':   self.training_history,
                    'feature_importance': self.feature_importance,
                    'cv_accuracy':        self.cv_accuracy
                }, f, indent=4)
            shutil.move(tmp_m, MODEL_PATH)
            shutil.move(tmp_s, SCALER_PATH)
            shutil.move(tmp_f, FEATURES_PATH)
            print("✅ 模型文件保存成功")
            return True
        except Exception as e:
            print(f"❌ 模型文件保存失败: {e}")
            for tmp in [tmp_m, tmp_s, tmp_f]:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass
            return False

    # 修复三：分步加载，详细报错
    def load(self):
        missing = [p for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
                   if not os.path.exists(p)]
        if missing:
            print(f"⚠️ 模型文件不存在: {missing}")
            return False
        try:
            self.model = joblib.load(MODEL_PATH)
            print("  ✔ 模型主文件加载成功")
        except Exception as e:
            print(f"  ✘ 模型主文件加载失败: {e}"); return False
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("  ✔ 归一化器加载成功")
        except Exception as e:
            print(f"  ✘ 归一化器加载失败: {e}"); return False
        try:
            with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.feature_names      = data.get('feature_names', [])
            self.training_history   = data.get('training_history', [])
            self.feature_importance = data.get('feature_importance', {})
            self.cv_accuracy        = data.get('cv_accuracy', 0.0)
            print(f"  ✔ 特征配置加载成功，共 {len(self.feature_names)} 个特征")
        except Exception as e:
            print(f"  ✘ 特征配置加载失败: {e}"); return False
        self.is_trained = True
        last = self.training_history[-1] if self.training_history else {}
        print(f"✅ 模型加载成功 | 交叉验证准确率={self.cv_accuracy:.4f} | 样本数={last.get('samples','?')}") 
        return True


ai_model = AITradingModel()
ai_model.load()

# =========================
# 配置 & 内存
# =========================
def load_config():
    default = {
        "coins": ["BTC-USDT", "ETH-USDT", "OKB-USDT"],
        "buy_threshold": 70, "sell_threshold": 35,
        "check_interval": 900,
        "telegram_bot_token": "", "telegram_chat_id": "",
        "email_user": "", "email_pass": "", "email_receiver": "",
        "use_ml_model": True, "ml_weight": 0.4
    }
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in default.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    except FileNotFoundError:
        return default

def load_memory():
    default = {
        "trend_weight": 0.3, "momentum_weight": 0.25, "volume_weight": 0.2,
        "volatility_weight": 0.1, "sentiment_weight": 0.2, "ml_weight": 0.4,
        "buy_threshold": 70, "sell_threshold": 35, "feature_importance": {},
        "last_training_time": 0, "best_accuracy": 0.0,
        "model_version": "1.0.0", "training_history": []
    }
    try:
        with open(MEMORY_PATH) as f:
            mem = json.load(f)
        for k in default:
            if k not in mem:
                mem[k] = default[k]
        return mem
    except Exception:
        save_memory(default)
        return default

def save_memory(memory):
    tmp = MEMORY_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=4)
        shutil.move(tmp, MEMORY_PATH)
    except Exception as e:
        print(f"⚠️ 策略记忆保存失败: {e}")

# =========================
# 通知
# =========================
def send_telegram_message(text, config):
    token   = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram消息发送失败: {e}")

def send_email(subject, body, config):
    user, pwd, receiver = (config.get("email_user"),
                           config.get("email_pass"),
                           config.get("email_receiver"))
    if not all([user, pwd, receiver]):
        return
    for host, port, use_tls in [("smtp.139.com", 25, True), ("smtp.139.com", 465, False)]:
        try:
            srv = smtplib.SMTP(host, port, timeout=10) if use_tls else \
                  smtplib.SMTP_SSL(host, port, timeout=10)
            if use_tls:
                srv.starttls()
            srv.login(user, pwd)
            msg = MIMEMultipart()
            msg["From"], msg["To"], msg["Subject"] = user, receiver, subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            srv.send_message(msg)
            srv.quit()
            return
        except Exception as e:
            print(f"邮件发送失败 {host}:{port}: {e}")

def send_notification(content, config, subject=None):
    send_telegram_message(content, config)
    if subject:
        send_email(subject, content, config)

# =========================
# 市场周期
# =========================
def detect_market_cycle():
    try:
        # 使用4h K线判断大方向（18根4h = 72小时趋势），与GPT建议一致
        df_4h   = get_kline("BTC-USDT", interval="4h", limit=60)
        df_1h   = get_kline("BTC-USDT", interval="1h", limit=100)
        ma50_1h = df_1h["close"].rolling(50).mean().iloc[-1]
        price   = df_4h["close"].iloc[-1]
        if pd.isna(ma50_1h):
            return "未知"
        # 4h K线看18根内动量（约72小时）
        momentum = (price - df_4h["close"].iloc[-18]) / df_4h["close"].iloc[-18]
        if price > ma50_1h and momentum > 0.05:
            return "牛市"
        elif price < ma50_1h:
            return "熊市"
        else:
            return "震荡"
    except Exception as e:
        print(f"市场周期识别失败: {e}")
        return "未知"

def apply_cycle_strategy_adjustment(memory, cycle):
    memory = memory.copy()
    # 未训练时只调整权重，不改阈值
    # 阈值固定为55/45，等待训练完成后再动态调整
    if not ai_model.is_trained:
        if cycle == "牛市":
            memory["trend_weight"] = min(memory.get("trend_weight", 0.3) + 0.05, 1.0)
        elif cycle == "熊市":
            memory["trend_weight"] = max(memory.get("trend_weight", 0.3) - 0.05, 0.0)
        elif cycle == "震荡":
            memory["volume_weight"] = min(memory.get("volume_weight", 0.2) + 0.05, 1.0)
        save_memory(memory)
        print(f"周期调整(未训练): {cycle}，仅调整权重，阈值保持55/45")
        return memory
    # 已训练时正常调整阈值
    if cycle == "牛市":
        memory["trend_weight"]   = min(memory.get("trend_weight", 0.3) + 0.05, 1.0)
        memory["buy_threshold"]  = min(memory.get("buy_threshold", 70) + 2, 85)
        memory["sell_threshold"] = max(memory.get("sell_threshold", 35) - 2, 20)
    elif cycle == "熊市":
        memory["trend_weight"]   = max(memory.get("trend_weight", 0.3) - 0.05, 0.0)
        memory["buy_threshold"]  = max(memory.get("buy_threshold", 70) - 2, 55)
        memory["sell_threshold"] = max(memory.get("sell_threshold", 35) - 2, 20)
    elif cycle == "震荡":
        memory["volume_weight"]  = min(memory.get("volume_weight", 0.2) + 0.05, 1.0)
        memory["buy_threshold"]  = max(memory.get("buy_threshold", 70) - 3, 55)
        memory["sell_threshold"] = min(memory.get("sell_threshold", 35) + 3, 45)
    memory["buy_threshold"]  = max(min(memory["buy_threshold"], 85), memory["sell_threshold"] + 5)
    memory["sell_threshold"] = max(min(memory["sell_threshold"], 50), 15)
    save_memory(memory)
    print(f"周期调整: {cycle}, 买入={memory['buy_threshold']}, 卖出={memory['sell_threshold']}")
    return memory

# =========================
# 评分
# =========================
def calculate_score(df, memory, whale, market_cycle, coin, config):
    df = df.dropna().copy()
    if len(df) < 60:
        rule_score = 50
        analysis   = {'trend': '未知', 'momentum': '未知', 'volume': '未知',
                      'volatility': '未知', 'position': 0.5,
                      'position_desc': '中等', 'volume_ratio': 1.0,
                      'trend_4h': '未知', 'trend_1h': '未知'}
    else:
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        ma20       = df["ma20"].iloc[-1]
        ma60       = df["ma60"].iloc[-1]
        momentum   = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

        # 未训练时使用放大权重，让评分更分散，方便积累信号数据
        # 已训练后使用memory里的自适应权重
        if not ai_model.is_trained:
            trend_w    = 0.8   # 放大，原0.3
            momentum_w = 0.6   # 放大，原0.25
            volume_w   = 0.5   # 放大，原0.2
        else:
            trend_w    = memory.get("trend_weight",    0.3)
            momentum_w = memory.get("momentum_weight", 0.25)
            volume_w   = memory.get("volume_weight",   0.2)

        rule_score = 50

        # ======= 三周期共振过滤（核心改动）=======
        # 获取4h大方向和1h主趋势
        trend_4h = "未知"
        trend_1h = "未知"
        try:
            df_4h  = get_kline(coin, interval="4h", limit=50)
            ma20_4h = df_4h['close'].rolling(20).mean().iloc[-1]
            ma5_4h  = df_4h['close'].rolling(5).mean().iloc[-1]
            trend_4h = "↑" if ma5_4h > ma20_4h else "↓"
        except Exception:
            pass
        try:
            df_1h  = get_kline(coin, interval="1h", limit=50)
            ma20_1h = df_1h['close'].rolling(20).mean().iloc[-1]
            ma5_1h  = df_1h['close'].rolling(5).mean().iloc[-1]
            trend_1h = "↑" if ma5_1h > ma20_1h else "↓"
        except Exception:
            pass

        # 三周期共振加权：
        # 4h定方向（权重最高），1h确认趋势，15m入场（当前df）
        # 三周期同向 → 强信号加分；逆向 → 减分
        trend_15m = "↑" if ma20 > ma60 else "↓"
        if trend_4h != "未知" and trend_1h != "未知":
            if trend_4h == "↑" and trend_1h == "↑" and trend_15m == "↑":
                rule_score += 25 * trend_w   # 三周期全做多：最强信号
            elif trend_4h == "↓" and trend_1h == "↓" and trend_15m == "↓":
                rule_score -= 25 * trend_w   # 三周期全做空：最强空信号
            elif trend_4h == "↑" and trend_1h == "↑":
                rule_score += 15 * trend_w   # 4h+1h做多，15m未确认：普通信号
            elif trend_4h == "↓" and trend_1h == "↓":
                rule_score -= 15 * trend_w   # 4h+1h做空
            elif trend_4h == "↑":
                rule_score += 8 * trend_w    # 只有4h看多
            elif trend_4h == "↓":
                rule_score -= 8 * trend_w    # 只有4h看空
        else:
            # 降级到原来的单周期判断（兜底）
            if ma20 > ma60:
                rule_score += 20 * trend_w
            else:
                rule_score -= 20 * trend_w

        # 动量：双向评分（强势加分，弱势减分）
        if momentum > 0.02:
            rule_score += 15 * momentum_w
        elif momentum < -0.02:
            rule_score -= 15 * momentum_w

        volume    = df["volume"].iloc[-1]
        avg_volume= df["volume"].mean()
        vol_ratio = volume / avg_volume if avg_volume != 0 else 1

        # 量能：双向评分（放量加分，缩量减分）
        if vol_ratio > 1.5:
            rule_score += 10 * volume_w
        elif vol_ratio < 0.5:
            rule_score -= 10 * volume_w

        if whale > 0:
            rule_score += 5
        rule_score = max(0, min(100, int(rule_score)))

        close_vals = df['close'].values
        returns    = np.diff(close_vals[-21:]) / close_vals[-21:-1] if len(close_vals) >= 21 else [0]
        vol        = np.std(returns)
        high_20    = np.max(close_vals[-20:])
        low_20     = np.min(close_vals[-20:])
        pos        = (close_vals[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5

        analysis = {
            'trend':         "多头" if ma20 > ma60 else "空头",
            'trend_4h':      trend_4h,
            'trend_1h':      trend_1h,
            'trend_15m':     trend_15m,
            'momentum':      "强" if momentum > 0.03 else ("中等" if momentum > 0.01 else "弱"),
            'volume':        "放量" if vol_ratio > 2 else ("温和放量" if vol_ratio > 1.5 else
                             ("缩量" if vol_ratio < 0.5 else "正常")),
            'volatility':    "高" if vol > 0.03 else ("中" if vol > 0.01 else "低"),
            'position':      pos,
            'position_desc': "高位（接近压力）" if pos > 0.8 else
                             ("低位（接近支撑）" if pos < 0.2 else "中等"),
            'volume_ratio':  vol_ratio
        }

    ml_score = 50
    ml_confidence = 0
    features = None
    if config.get("use_ml_model", True) and ai_model.is_trained:
        try:
            features = ai_model.extract_features(df, whale, market_cycle, coin)
            if features:
                ml_score, ml_info = ai_model.predict(features)
                ml_confidence     = ml_info.get('confidence', 0)
        except Exception as e:
            print(f"AI模型预测失败: {e}")
    else:
        try:
            features = ai_model.extract_features(df, whale, market_cycle, coin)
        except Exception:
            pass

    ml_weight      = config.get("ml_weight", 0.4) if ai_model.is_trained else 0
    combined_score = rule_score * (1 - ml_weight) + ml_score * ml_weight
    up_prob        = ml_score / 100 if ai_model.is_trained else combined_score / 100

    atr       = calculate_atr(df)
    cur_price = df['close'].iloc[-1]

    # 盈亏比优化：止盈ATR×4，止损ATR×1.5，盈亏比2.67:1
    # 即使胜率只有40%，期望值仍为正（0.4×4 - 0.6×1.5 = 0.7 > 0）
    factors = {
        'rule_score':      rule_score,
        'ml_score':        ml_score,
        'ml_confidence':   ml_confidence,
        'combined_score':  combined_score,
        'up_prob':         max(0, min(1, up_prob)),
        'analysis':        analysis,
        'atr':             atr,
        'stop_loss_buy':   round(cur_price - atr * 1.5, 6),
        'take_profit_buy': round(cur_price + atr * 4,   6),
        'stop_loss_sell':  round(cur_price + atr * 1.5, 6),
        'take_profit_sell':round(cur_price - atr * 4,   6),
    }
    return int(combined_score), factors, features

# =========================
# 日志
# features 独立存储，防止主日志文件过大导致OOM
# 主日志只存 feature_id（索引），features内容单独存一个文件
# =========================
def load_features_log():
    """加载特征数据文件"""
    try:
        with open(FEATURES_LOG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_features_log(features_data):
    """原子保存特征数据文件"""
    tmp = FEATURES_LOG_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(features_data, f)  # 不加indent，节省空间
        shutil.move(tmp, FEATURES_LOG_PATH)
    except Exception as e:
        print(f"⚠️ 特征数据保存失败: {e}")

# =========================
# 日志
# =========================
def load_log():
    try:
        with open(LOG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_log(log):
    tmp = LOG_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=4)
        shutil.move(tmp, LOG_PATH)
    except Exception as e:
        print(f"⚠️ 信号日志保存失败: {e}")

def verify_past_signals(config):
    # 三重障碍法验证信号（参考 Marcos Lopez de Prado 量化投资方法论）
    # 在未来2小时窗口内：
    #   止盈+0.3%先触发 → correct
    #   止损-0.3%先触发 → wrong
    #   两者都未触发    → 看2小时后收盘价方向
    # 相比原来25分钟+最高价的方式，熊市横盘下能产生均衡的correct/wrong分布

    BARRIER_PCT   = 0.003   # 止盈止损均为0.3%
    WINDOW_BARS   = 8       # 验证窗口：8根15分钟K线 = 2小时
    MIN_AGE_MIN   = 30      # 信号至少30分钟后才验证（15m系统等一根K线关闭）
    SAVE_EVERY    = 10      # 每验证10条写一次，防OOM Kill丢失

    log           = load_log()
    now_ts        = time.time()
    updated_count = 0

    for entry in log:
        if entry.get("verified", False):
            continue

        coin        = entry["coin"]
        signal_time = entry["timestamp"]
        age_minutes = (now_ts - signal_time) / 60

        if age_minutes < MIN_AGE_MIN:
            continue

        try:
            df = get_kline(coin, interval="15m", limit=100)
            df['ts_sec'] = df['ts'] / 1000
            future_df    = df[df['ts_sec'] > signal_time].sort_values('ts_sec')

            if len(future_df) >= 5:
                window_df    = future_df.iloc[:WINDOW_BARS]
                signal_price = entry["price"]

                if entry["signal"] == "买入":
                    result = None
                    profit = 0
                    for _, bar in window_df.iterrows():
                        if bar['high'] >= signal_price * (1 + BARRIER_PCT):
                            result = "correct"
                            profit = BARRIER_PCT * 100
                            break
                        if bar['low'] <= signal_price * (1 - BARRIER_PCT):
                            result = "wrong"
                            profit = -BARRIER_PCT * 100
                            break
                    if result is None:
                        final_price = window_df['close'].iloc[-1]
                        profit      = (final_price - signal_price) / signal_price * 100
                        result      = "correct" if profit > 0 else "wrong"

                elif entry["signal"] == "卖出":
                    result = None
                    profit = 0
                    for _, bar in window_df.iterrows():
                        if bar['low'] <= signal_price * (1 - BARRIER_PCT):
                            result = "correct"
                            profit = BARRIER_PCT * 100
                            break
                        if bar['high'] >= signal_price * (1 + BARRIER_PCT):
                            result = "wrong"
                            profit = -BARRIER_PCT * 100
                            break
                    if result is None:
                        final_price = window_df['close'].iloc[-1]
                        profit      = (signal_price - final_price) / signal_price * 100
                        result      = "correct" if profit > 0 else "wrong"
                else:
                    continue

                entry["verified"] = True
                entry["result"]   = result
                entry["profit"]   = round(profit, 3)
                updated_count    += 1

            elif age_minutes > 120:
                # K线数据不足时，用当前价格方向判断（不再强求盈利幅度）
                current_price = df['close'].iloc[-1]
                signal_price  = entry["price"]
                if entry["signal"] == "买入":
                    profit = (current_price - signal_price) / signal_price * 100
                    result = "correct" if profit > 0 else "wrong"
                elif entry["signal"] == "卖出":
                    profit = (signal_price - current_price) / signal_price * 100
                    result = "correct" if profit > 0 else "wrong"
                else:
                    continue
                entry["verified"] = True
                entry["result"]   = result
                entry["profit"]   = round(profit, 3)
                entry["note"]     = "超时验证（K线数据不足）"
                updated_count    += 1
                print(f"⏰ {coin} 超时验证（{age_minutes:.0f}分钟）: {result}")

            if updated_count > 0 and updated_count % SAVE_EVERY == 0:
                save_log(log)
                print(f"💾 中途保存：已验证 {updated_count} 条")

        except Exception as e:
            print(f"验证 {coin} 历史信号失败: {e}")

    if updated_count > 0:
        save_log(log)
        print(f"✅ 本次验证完成，共验证 {updated_count} 条并保存")

def log_signal(coin, signal, score, price, whale, market_cycle, factors, features):
    log          = load_log()
    features_data= load_features_log()
    ts           = time.time()

    # features 单独存储，主日志只存 feature_id（时间戳字符串作为索引）
    # 这样主日志每条记录只有几十字节，而不是几KB
    feature_id = None
    if features:
        feature_id = str(int(ts * 1000))  # 毫秒时间戳作为唯一ID
        features_data[feature_id] = features
        save_features_log(features_data)

    entry = {
        "timestamp":     ts,
        "coin":          coin,
        "signal":        signal,
        "score":         score,
        "price":         price,
        "whale":         whale,
        "market_cycle":  market_cycle,
        "rule_score":    factors.get('rule_score', 50),
        "ml_score":      factors.get('ml_score', 50),
        "ml_confidence": factors.get('ml_confidence', 0),
        "feature_id":    feature_id,  # 只存ID，不存完整features
        "verified":      False,
        "result":        None,
        "profit":        None
    }
    log.append(entry)
    if len(log) > MAX_LOG_SIZE:
        verified   = [e for e in log if e.get("verified")]
        unverified = [e for e in log if not e.get("verified")]
        keep       = MAX_LOG_SIZE - len(unverified)
        # 清理已删除日志对应的features，避免features_log无限增长
        kept_ids   = {e.get('feature_id') for e in verified[-keep:] + unverified if e.get('feature_id')}
        features_data = {k: v for k, v in features_data.items() if k in kept_ids}
        save_features_log(features_data)
        log = (verified[-keep:] if keep > 0 else []) + unverified
    save_log(log)

# =========================
# 自适应优化
# =========================
def adaptive_strategy_optimization(config):
    log      = load_log()
    verified = [e for e in log if e.get("verified") and e.get("result") in ("correct", "wrong")]
    if len(verified) < MIN_TRAIN_SAMPLES:
        print(f"自适应优化：已验证 {len(verified)}/{MIN_TRAIN_SAMPLES} 条，样本不足，跳过训练")
        return

    X_df, y = ai_model.prepare_training_data(verified)
    if X_df is None:
        print("特征提取完成后样本仍不足，跳过训练")
        return

    success = ai_model.train(X_df, y)
    if not success:
        return

    ai_model.save()
    cv_acc = ai_model.cv_accuracy
    memory = load_memory()
    memory['feature_importance'] = ai_model.feature_importance

    # 近期信号样本数（用于防止少量样本胜率虚高导致阈值失控）
    recent_signals_count = len(sorted(verified,
                                      key=lambda x: x.get('timestamp', 0),
                                      reverse=True)[:50])

    # 至少需要30条近期已验证信号才能调整阈值
    # 防止3条样本100%胜率就把阈值推到85的情况
    if recent_signals_count >= 30:
        if cv_acc > 0.60:
            memory['buy_threshold']  = min(memory.get('buy_threshold', 65) + 2, 75)  # 上限75（原85）
            memory['sell_threshold'] = max(memory.get('sell_threshold', 35) - 2, 25)  # 下限25（原20）
            memory['ml_weight']      = min(0.5, memory.get('ml_weight', 0.4) + 0.05)  # 上限0.5（原0.7）
        elif cv_acc < 0.50:
            memory['buy_threshold']  = max(memory.get('buy_threshold', 65) - 2, 55)
            memory['sell_threshold'] = min(memory.get('sell_threshold', 35) + 2, 45)
            memory['ml_weight']      = max(0.1, memory.get('ml_weight', 0.4) - 0.1)
            print(f"⚠️ 交叉验证准确率偏低({cv_acc:.3f})，已降低AI模型权重至{memory['ml_weight']:.2f}")
    else:
        print(f"⚠️ 近期样本不足30条（当前{recent_signals_count}条），跳过阈值调整，防止虚高胜率误导")

    # 阈值安全边界（绝对不能超出的范围）
    # 买入最高70（熊市评分普遍40-65，70以上几乎触发不了）
    # 卖出最低30（同理）
    # 两者差值至少10，保证信号区间足够宽
    memory['buy_threshold']  = max(min(memory['buy_threshold'], 70), memory['sell_threshold'] + 10)
    memory['sell_threshold'] = max(min(memory['sell_threshold'], 45), 30)
    save_memory(memory)

    recent_verified = sorted(verified, key=lambda x: x.get('timestamp', 0), reverse=True)[:100]
    recent_correct  = sum(1 for e in recent_verified if e.get('result') == 'correct')
    real_winrate    = recent_correct / len(recent_verified) if recent_verified else 0

    train_acc = ai_model.training_history[-1]['train_acc']
    overfit_warning = ""
    if train_acc - cv_acc > 0.15:
        overfit_warning = f"\n⚠️ 检测到过拟合（训练与验证差距{train_acc - cv_acc:.2%}），已自动降低AI模型权重"

    top5        = sorted(ai_model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    feature_msg = "\n".join([f"  {f}: {v:.4f}" for f, v in top5]) if top5 else "  （暂无）"

    # 计算期望值
    ev_data = calculate_expected_value(recent_verified)
    ev_msg  = format_ev_message(ev_data)

    msg = (f"🤖 AI模型已进化\n"
           f"训练样本: {len(X_df)}\n"
           f"训练准确率: {train_acc:.2%}\n"
           f"交叉验证准确率: {cv_acc:.2%}  ← 真实参考值\n"
           f"实际胜率(近100条): {real_winrate:.2%}\n"
           f"新阈值: 买入={memory['buy_threshold']}, 卖出={memory['sell_threshold']}\n"
           f"AI模型权重: {memory['ml_weight']:.2f}\n\n"
           f"💰 系统期望值:\n{ev_msg}\n\n"
           f"前5重要特征:\n{feature_msg}"
           f"{overfit_warning}")
    # 训练成功后同步更新config阈值，当轮循环立即生效
    config['buy_threshold']  = memory['buy_threshold']
    config['sell_threshold'] = memory['sell_threshold']
    print(f"✅ 模型训练完成，阈值已更新: 买入={config['buy_threshold']}，卖出={config['sell_threshold']}")
    send_notification(msg, config, "AI模型进化报告")

# =========================
# 核心盈利指标：期望值（Expected Value）
# 判断系统是否真正在盈利的唯一可靠指标
# expected_value > 0 → 长期盈利
# expected_value < 0 → 长期亏损，需要优化
# =========================
def calculate_expected_value(verified_list=None):
    """
    计算系统期望值（Expected Value）
    EV = 胜率 × 平均盈利 - 败率 × 平均亏损
    正值说明系统长期可盈利，负值说明需要优化
    """
    if verified_list is None:
        log = load_log()
        verified_list = [e for e in log if e.get("verified")
                         and e.get("result") in ("correct", "wrong")
                         and e.get("profit") is not None]

    if len(verified_list) < 10:
        return None  # 样本不足，无法计算

    correct_list = [e for e in verified_list if e.get("result") == "correct"]
    wrong_list   = [e for e in verified_list if e.get("result") == "wrong"]

    if not correct_list or not wrong_list:
        return None  # 标签不均衡，无法计算

    win_rate     = len(correct_list) / len(verified_list)
    loss_rate    = len(wrong_list)   / len(verified_list)
    avg_win      = np.mean([abs(e['profit']) for e in correct_list])
    avg_loss     = np.mean([abs(e['profit']) for e in wrong_list])

    ev = win_rate * avg_win - loss_rate * avg_loss

    return {
        'ev':        round(ev, 4),          # 每次信号的期望盈利（%）
        'win_rate':  round(win_rate, 4),    # 胜率
        'loss_rate': round(loss_rate, 4),   # 败率
        'avg_win':   round(avg_win, 4),     # 平均盈利（%）
        'avg_loss':  round(avg_loss, 4),    # 平均亏损（%）
        'total':     len(verified_list),    # 样本总数
        'is_profitable': ev > 0             # 是否盈利
    }

def format_ev_message(ev_data):
    """格式化期望值信息用于消息推送"""
    if ev_data is None:
        return "  期望值: 样本不足（需至少10条验证信号）"
    ev      = ev_data['ev']
    emoji   = "✅" if ev > 0 else "❌"
    trend   = "长期盈利" if ev > 0 else "长期亏损，需优化"
    return (
        f"  {emoji} 期望值: {ev:+.3f}% （{trend}）\n"
        f"  胜率:{ev_data['win_rate']:.0%} | 平均盈利:{ev_data['avg_win']:.3f}%\n"
        f"  败率:{ev_data['loss_rate']:.0%} | 平均亏损:{ev_data['avg_loss']:.3f}%\n"
        f"  计算样本: {ev_data['total']}条"
    )

# =========================
# 回测 & 统计
# =========================
def generate_backtest_report():
    log      = load_log()
    verified = [e for e in log if e.get("verified")
                and e.get("result") in ("correct", "wrong")]
    if not verified:
        return "暂无已验证数据"
    total      = len(verified)
    correct    = sum(1 for e in verified if e.get("result") == "correct")
    profits    = [e.get('profit', 0) for e in verified if e.get('profit') is not None]
    avg_profit = np.mean(profits) if profits else 0

    # 计算期望值
    ev_data = calculate_expected_value(verified)
    ev_msg  = format_ev_message(ev_data)

    # 买入/卖出分别统计
    buy_list  = [e for e in verified if e.get("signal") == "买入"]
    sell_list = [e for e in verified if e.get("signal") == "卖出"]
    buy_correct  = sum(1 for e in buy_list  if e.get("result") == "correct")
    sell_correct = sum(1 for e in sell_list if e.get("result") == "correct")
    buy_wr  = buy_correct  / len(buy_list)  if buy_list  else 0
    sell_wr = sell_correct / len(sell_list) if sell_list else 0

    return (f"📊 每日回测报告\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"📈 总体统计:\n"
            f"  总信号: {total}条\n"
            f"  正确: {correct} | 错误: {total - correct}\n"
            f"  综合胜率: {correct/total:.2%}\n"
            f"  平均盈亏: {avg_profit:.3f}%\n\n"
            f"🔍 分类胜率:\n"
            f"  买入信号: {len(buy_list)}次 | 胜率{buy_wr:.0%}\n"
            f"  卖出信号: {len(sell_list)}次 | 胜率{sell_wr:.0%}\n\n"
            f"💰 系统期望值:\n{ev_msg}")

def get_recent_signals(n=3):
    log     = load_log()
    signals = sorted([e for e in log if e.get("signal") in ("买入", "卖出")],
                     key=lambda x: x.get("timestamp", 0), reverse=True)
    now     = time.time()
    result  = []
    for s in signals[:n]:
        mins_ago    = int((now - s.get("timestamp", 0)) / 60)
        t_str       = f"{mins_ago}分钟前" if mins_ago < 60 else f"{mins_ago//60}小时前"
        result_mark = (" ✅" if s.get("result") == "correct" else " ❌") \
                      if s.get("verified") else ""
        result.append(f"{t_str} {s['coin']} {s['signal']} ({s['score']}){result_mark}")
    return result

def get_signal_stats_since(hours):
    log    = load_log()
    cutoff = time.time() - hours * 3600
    recent = [e for e in log if e.get("timestamp", 0) > cutoff
              and e.get("signal") in ("买入", "卖出")]
    total  = len(recent)
    correct= sum(1 for e in recent if e.get("result") == "correct")
    return total, correct

# =========================
# 修改一：优化后的买入过滤
# 区分主流币（BTC联动强）和山寨币（可能独立行情）
# =========================
def check_buy_filters(coin, df, memory):
    base             = coin.split("-")[0].upper()
    is_major         = base in MAJOR_COINS
    btc_info         = get_btc_dominance_trend()
    # 未训练时放宽过滤，目标是积累训练数据
    is_accumulating  = not ai_model.is_trained

    if is_major:
        # 主流币：BTC整体下跌趋势时过滤（无论是否训练都执行）
        if btc_info['btc_falling']:
            return False, f"BTC均线死叉下跌，主流币{coin}过滤"
    else:
        # 山寨币：BTC走独立行情时才过滤
        if btc_info['btc_independent']:
            return False, (f"BTC独立拉升中"
                           f"(BTC+{btc_info['btc_change']*100:.1f}% vs 大盘"
                           f"{btc_info['market_avg']*100:.1f}%)，资金虹吸，山寨暂缓")

    # 已训练后才做成交量和波动率过滤
    # 未训练阶段只做方向性过滤，不做量价过滤，降低门槛积累数据
    if not is_accumulating:
        volume  = df['volume'].iloc[-1]
        avg_vol = df['volume'].mean()
        if volume < avg_vol * VOLUME_RATIO_MIN:
            return False, f"成交量不足({volume:.0f}<{avg_vol*VOLUME_RATIO_MIN:.0f})"

        close = df['close'].values
        if len(close) >= 21:
            vol = np.std(np.diff(close[-21:]) / close[-21:-1])
            if vol < VOLATILITY_MIN:
                return False, f"波动率过低({vol:.4f})"

    return True, "通过"

def generate_risk_analysis(analysis, factors, config):
    risks = []
    if analysis['volatility'] == "高":
        risks.append("波动过大")
    if analysis.get('volume_ratio', 1) < 1.5:
        risks.append("成交量不足")
    try:
        btc_info = get_btc_dominance_trend()
        if btc_info['btc_falling']:
            risks.append("大盘走弱")
    except Exception:
        pass
    if analysis['position'] > 0.8:
        risks.append("接近压力位")
    elif analysis['position'] < 0.2:
        risks.append("接近支撑位")
    risk_level = "高" if len(risks) >= 3 else ("中" if risks else "低")
    return risk_level, risks

# =========================
# 信号通知
# =========================
def build_signal_message(coin, base_signal, score, display_price,
                          factors, analysis, risk_level, risks,
                          confirm_count, avg_score, config,
                          signal_grade="strong"):
    up_prob    = factors['up_prob'] * 100
    confidence = factors.get('ml_confidence', 0) * 100

    if base_signal == "买入":
        sl     = factors['stop_loss_buy']
        tp     = factors['take_profit_buy']
        sl_pct = abs(display_price - sl) / display_price * 100
        tp_pct = abs(tp - display_price) / display_price * 100
    else:
        sl     = factors['stop_loss_sell']
        tp     = factors['take_profit_sell']
        sl_pct = abs(sl - display_price) / display_price * 100
        tp_pct = abs(display_price - tp) / display_price * 100

    if risk_level == "高":
        position_suggest = "轻仓(≤10%)"
    elif risk_level == "中":
        position_suggest = "半仓(≤30%)"
    else:
        position_suggest = "正常仓(≤50%)"

    cv_acc = ai_model.cv_accuracy if ai_model.is_trained else 0
    model_reliability = "⚠️ 模型未训练，仅供参考" if not ai_model.is_trained else \
                        (f"✅ 可靠（交叉验证: {cv_acc:.0%}）" if cv_acc >= 0.55 else
                         f"⚠️ 可信度偏低（交叉验证: {cv_acc:.0%}），谨慎参考")

    # 修改三：显示当前冷却时间
    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
    cooldown_str = f"{cooldown_sec // 60}分钟"

    signal_emoji = "🟢" if base_signal == "买入" else "🔴"
    risk_emoji   = "🔴" if risk_level == "高" else ("🟡" if risk_level == "中" else "🟢")

    # 次级信号用黄色标注，提示用户谨慎参考
    grade_note = ""
    if signal_grade == "weak":
        signal_emoji = "🟡"
        grade_note   = "⚠️ 次级信号（评分接近阈值，谨慎参考）\n\n"

    msg = (
        f"{signal_emoji} <b>{coin} {base_signal}信号</b>\n\n"
        f"{grade_note}"
        f"💰 价格：${display_price:.6g}\n"
        f"📊 综合评分：{score} (规则:{factors['rule_score']} AI:{factors['ml_score']:.0f})\n"
        f"🎯 上涨概率：{up_prob:.1f}%\n"
        f"🔁 连续确认：{confirm_count}次 (均分{avg_score:.0f})\n\n"
        f"📈 趋势：{analysis['trend']} | 动量：{analysis['momentum']}\n"
        f"🔍 三周期: 4h{analysis.get('trend_4h','?')} | 1h{analysis.get('trend_1h','?')} | 15m{analysis.get('trend_15m','?')}\n"
        f"📦 成交量：{analysis['volume']} | 波动：{analysis['volatility']}\n"
        f"📍 位置：{analysis['position_desc']}\n\n"
        f"🛡️ 止损价：${sl:.6g} (-{sl_pct:.1f}%)\n"
        f"🎯 目标价：${tp:.6g} (+{tp_pct:.1f}%)\n"
        f"💼 建议仓位：{position_suggest}\n\n"
        f"{risk_emoji} 风险等级：{risk_level}\n"
        f"⚠️ 风险点：{'、'.join(risks) if risks else '无明显风险'}\n\n"
        f"🤖 模型：{model_reliability}\n"
        f"🌍 当前周期：{current_market_cycle}（下次信号冷却：{cooldown_str}）\n"
        f"⏰ {datetime.now().strftime('%m-%d %H:%M')}"
    )
    return msg

# =========================
# 状态推送
# =========================
def build_status_message(coins, memory, config,
                          cached_scores=None, cached_factors=None):
    """
    构建状态推送消息。
    cached_scores/cached_factors：主循环本轮已计算的评分，
    传入后直接复用，不重复调用 calculate_score。
    """
    now    = time.time()
    uptime = int(now - start_time)
    h, m   = uptime // 3600, (uptime % 3600) // 60

    if ai_model.is_trained and ai_model.training_history:
        last      = ai_model.training_history[-1]
        train_acc = last.get('train_acc', last.get('accuracy', 0))
        cv_acc    = last.get('cv_accuracy', ai_model.cv_accuracy)
        gap       = train_acc - cv_acc
        overfit_tag = f" ⚠️轻微过拟合" if gap > 0.15 else ""
        model_info  = (f"✅ 已训练 | 样本:{last['samples']}\n"
                       f"     训练准确率:{train_acc:.0%} 交叉验证:{cv_acc:.0%}{overfit_tag}")
    else:
        # 显示已积累的训练数据进度，让用户知道还差多少
        try:
            log_data       = load_log()
            verified_count = sum(1 for e in log_data
                                 if e.get("verified") and e.get("result") in ("correct", "wrong"))
            total_logged   = len([e for e in log_data if e.get("signal") in ("买入", "卖出")])
            # 进度条（10格）
            pending_count = sum(1 for e in log_data
                                if e.get("signal") in ("买入","卖出") and not e.get("verified"))
            correct_count = sum(1 for e in log_data if e.get("result") == "correct")
            wrong_count   = sum(1 for e in log_data if e.get("result") == "wrong")
            progress = min(10, int(verified_count / MIN_TRAIN_SAMPLES * 10))
            bar = "█" * progress + "░" * (10 - progress)
            # 胜率说明
            if verified_count > 10 and correct_count == 0:
                winrate_note = "\n     ⚠️ 已验证全部错误，等待多样化行情（不影响积累）"
            elif verified_count > 0:
                winrate_note = f"\n     当前胜率: {correct_count/verified_count:.0%}（{correct_count}对/{wrong_count}错）"
            else:
                winrate_note = ""
            model_info = (
                f"❌ 未训练\n"
                f"     进度: [{bar}] {verified_count}/{MIN_TRAIN_SAMPLES}条\n"
                f"     已验证:{verified_count}条 | 等待中:{pending_count}条 | 已记录:{total_logged}条"
                f"{winrate_note}"
            )
        except Exception:
            model_info = "❌ 未训练（积累中...）"

    today_total, today_correct = get_signal_stats_since(24)
    today_winrate = today_correct / today_total if today_total > 0 else 0
    winrate_tag   = " ⚠️偏低" if today_total >= 10 and today_winrate < 0.4 else ""

    sixh_total, sixh_correct = get_signal_stats_since(6)
    sixh_winrate = sixh_correct / sixh_total if sixh_total > 0 else 0

    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
    cooldown_str = f"{cooldown_sec // 60}分钟"

    coin_lines = []
    for coin in coins:
        try:
            # 优先复用主循环本轮缓存，避免重复计算
            if cached_scores and cached_factors and coin in cached_scores:
                score   = cached_scores[coin]
                factors = cached_factors[coin]
                analysis= factors['analysis']
                price   = get_ticker(coin) or 0
            else:
                df      = get_kline(coin, interval="15m", limit=200)
                score, factors, _ = calculate_score(df, memory, detect_whale(coin),
                                                    current_market_cycle, coin, config)
                analysis= factors['analysis']
                price   = get_ticker(coin) or df['close'].iloc[-1]
            direction = "↑" if analysis['trend'] == "多头" else "↓"
            up_prob   = factors['up_prob'] * 100
            coin_lines.append(
                f"  {coin}: ${price:.6g} | {score:.0f}分 {direction} | {up_prob:.0f}%"
            )
        except Exception:
            coin_lines.append(f"  {coin}: 获取失败")

    recent  = get_recent_signals(3)
    hot     = scan_hot_coins()[:3]
    hot_str = "\n".join([f"  {c}: {ch:+.2f}%" for c, ch in hot]) if hot else "  暂无"

    # 计算期望值
    ev_data = calculate_expected_value()
    ev_line = ""
    if ev_data:
        ev_emoji = "✅" if ev_data['ev'] > 0 else "❌"
        ev_line  = (f"\n💰 系统期望值:\n"
                    f"  {ev_emoji} {ev_data['ev']:+.3f}% | "
                    f"胜率{ev_data['win_rate']:.0%} | "
                    f"盈{ev_data['avg_win']:.2f}% 亏{ev_data['avg_loss']:.2f}%")

    status = (
        f"📡 <b>AI监控状态报告</b>\n\n"
        f"📈 监控币种:\n" + "\n".join(coin_lines) + "\n\n"
        f"⏱️ 最近信号:\n" +
        ("\n".join([f"  {s}" for s in recent]) if recent else "  暂无") + "\n\n"
        f"🔥 市场热点:\n{hot_str}\n\n"
        f"🤖 模型状态:\n  {model_info}\n\n"
        f"📊 胜率统计:\n"
        f"  6小时: {sixh_total}次 | {sixh_winrate:.0%}\n"
        f"  24小时: {today_total}次 | {today_winrate:.0%}{winrate_tag}"
        f"{ev_line}\n\n"
        f"⚙️ 阈值: 买入{config['buy_threshold']} | 卖出{config['sell_threshold']}"
        f"（{'动态' if ai_model.is_trained and len(_score_history)>=20 else '固定'}，基于{len(_score_history)}条历史评分）\n"
        f"⏳ 信号冷却时间: {cooldown_str}（{current_market_cycle}）\n"
        f"🕐 运行: {h}小时{m}分钟 | {current_market_cycle}"
    )
    return status

# =========================
# 启动锁：防止Railway滚动部署时新旧容器同时运行
# =========================
LOCK_FILE    = os.path.join(DATA_DIR, ".process_lock")
LOCK_TIMEOUT = 60  # 锁超时60秒，超过则认为旧进程已死

def acquire_startup_lock():
    """
    获取启动锁，防止新旧容器同时写文件互相覆盖。
    若发现有效锁（旧进程存活），等待其超时后再继续。
    """
    now = time.time()
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                data = json.load(f)
            age = now - data.get('timestamp', 0)
            pid = data.get('pid', 0)
            if age < LOCK_TIMEOUT:
                print(f"⏳ 检测到旧进程运行中 (进程号:{pid}, {age:.0f}秒前)，等待旧进程退出...")
                wait_start = time.time()
                while time.time() - wait_start < LOCK_TIMEOUT:
                    time.sleep(2)
                    try:
                        with open(LOCK_FILE, 'r') as f:
                            d = json.load(f)
                        if time.time() - d.get('timestamp', 0) >= LOCK_TIMEOUT:
                            break
                    except Exception:
                        break
                print("✅ 旧进程已退出，新进程继续启动")
            else:
                print(f"⚠️ 发现过期进程锁（{age:.0f}秒前），直接覆盖")
        except Exception as e:
            print(f"⚠️ 读取进程锁异常（{e}），直接覆盖")
    try:
        with open(LOCK_FILE, 'w') as f:
            json.dump({'timestamp': now, 'pid': os.getpid()}, f)
        print(f"🔒 进程锁已获取 (进程号:{os.getpid()})")
    except Exception as e:
        print(f"⚠️ 无法写入进程锁: {e}")

def release_startup_lock():
    """进程退出时释放启动锁"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            print("🔓 进程锁已释放")
    except Exception:
        pass

def update_lock_heartbeat():
    """
    主循环每轮调用，更新锁文件时间戳（心跳）。
    让其他进程知道当前进程仍然存活。
    """
    try:
        with open(LOCK_FILE, 'w') as f:
            json.dump({'timestamp': time.time(), 'pid': os.getpid()}, f)
    except Exception:
        pass

# =========================
# 主程序
# =========================
def main():
    global last_backtest_time, last_adaptive_time, last_cycle_check
    global last_status_push, last_daily_report, last_signal_time, last_scores
    global current_market_cycle, start_time, _score_history

    config     = load_config()
    start_time = time.time()
    print("=" * 50)
    print("AI自主学习交易系统启动（Gate.io 决策辅助版）")
    print(f"数据目录: {DATA_DIR}")
    print(f"存储状态: {'✅已挂载' if _volume_ok else '❌本地临时目录（数据重启丢失）'}")
    print(f"模型状态: {'已训练，交叉验证准确率=' + str(round(ai_model.cv_accuracy, 4)) if ai_model.is_trained else '未训练，等待积累数据'}")
    print("=" * 50)

    # 获取启动锁，防止新旧容器同时运行
    acquire_startup_lock()

    # =========================
    # 自动重置验证数据（检测到 RESET_VERIFIED=1 环境变量时触发）
    # 用途：部署新验证逻辑后，将旧数据重新验证
    # 使用方法：在Railway Variables里添加 RESET_VERIFIED=1，
    #           部署后系统自动重置，重置完成后删除该变量即可
    # =========================
    if os.getenv("RESET_VERIFIED", "").strip() in ("1", "true", "True", "yes"):
        try:
            log = load_log()
            reset_count = 0
            for entry in log:
                if entry.get("verified", False):
                    entry["verified"] = False
                    entry["result"]   = None
                    entry["profit"]   = None
                    entry.pop("note", None)
                    reset_count += 1
            if reset_count > 0:
                save_log(log)
                msg = (
                    f"🔄 验证数据已重置\n"
                    f"共重置 {reset_count} 条记录\n"
                    f"系统将用新验证逻辑重新验证\n"
                    f"完成后请删除 RESET_VERIFIED 变量"
                )
                print(msg)
                send_telegram_message(msg, config)
            else:
                print("⚠️ 没有需要重置的验证数据")
        except Exception as e:
            print(f"⚠️ 重置验证数据失败: {e}")

    # 从文件恢复定时状态，重启不重复推送
    timing_state       = load_timing_state()
    last_backtest_time = timing_state.get('last_backtest_time', 0)
    last_daily_report  = timing_state.get('last_daily_report',  0)
    last_status_push   = timing_state.get('last_status_push',   0)
    print(f"📅 上次回测报告: "
          f"{datetime.fromtimestamp(last_backtest_time).strftime('%Y-%m-%d %H:%M') if last_backtest_time > 0 else '从未'}")
    print(f"📅 上次每日报告: "
          f"{datetime.fromtimestamp(last_daily_report).strftime('%Y-%m-%d %H:%M') if last_daily_report > 0 else '从未'}")
    print(f"📅 上次状态报告: "
          f"{datetime.fromtimestamp(last_status_push).strftime('%Y-%m-%d %H:%M') if last_status_push > 0 else '从未'}")

    if not ai_model.is_trained:
        memory = load_memory()
        # 未训练时使用宽松阈值，配合放大权重，让规则评分能触发信号
        # 规则评分上限约82（多头+强动量+放量），下限约18（空头+弱动量+缩量）
        # 阈值55/45可以覆盖到明显的涨跌行情
        memory["buy_threshold"]  = 55
        memory["sell_threshold"] = 45
        save_memory(memory)
        config["buy_threshold"]  = 55
        config["sell_threshold"] = 45
        print("⚠️ 模型未训练，使用宽松阈值: 买入55 / 卖出45（加速积累初始训练数据）")
    else:
        # 已训练时检测阈值是否合理，防止历史遗留的异常值（如阈值跑到85/20）
        memory = load_memory()
        buy_thr  = memory.get("buy_threshold", 65)
        sell_thr = memory.get("sell_threshold", 35)
        if buy_thr > 70 or sell_thr < 30 or buy_thr - sell_thr < 10:
            print(f"⚠️ 检测到异常阈值（买入={buy_thr} 卖出={sell_thr}），自动重置为65/35")
            memory["buy_threshold"]  = 65
            memory["sell_threshold"] = 35
            save_memory(memory)
            config["buy_threshold"]  = 65
            config["sell_threshold"] = 35
        # 检测ml_weight是否超出新上限0.5
        ml_w = memory.get("ml_weight", 0.4)
        if ml_w > 0.5:
            print(f"⚠️ AI权重过高（{ml_w:.2f}），自动降至0.5（防过拟合放大错误）")
            memory["ml_weight"] = 0.5
            save_memory(memory)
        else:
            print(f"✅ 模型已训练，阈值正常（买入={buy_thr} 卖出={sell_thr}），AI权重={ml_w:.2f}")

    # 启动通知（含Volume状态，方便排查）
    log_size    = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    volume_info = f"✅ 存储已挂载" if _volume_ok else "❌ 存储未挂载！重启后数据全部丢失"
    send_telegram_message(
        f"🚀 系统启动\n"
        f"模型: {'✅已训练' if ai_model.is_trained else '❌未训练'}\n"
        f"日志大小: {log_size} 字节\n"
        f"存储: {volume_info}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        config
    )

    # 恢复信号冷却时间，防止重启后重复发送信号
    last_signal_time = load_signal_time()
    if last_signal_time:
        active = {k: v for k, v in last_signal_time.items()
                  if time.time() - v < max(COOLDOWN_BY_CYCLE.values())}
        print(f"📅 已恢复信号冷却记录，共 {len(active)} 个币种")

    # 启动后强制推送一次状态，让用户确认系统正常运行
    force_push_on_start = True
    last_forced_status  = 0   # 记录上次强制推送时间，初始为0确保启动后立即推送

    while True:
        try:
            now = time.time()

            # 市场周期检查
            if now - last_cycle_check > MARKET_CYCLE_INTERVAL:
                new_cycle = detect_market_cycle()
                if new_cycle != current_market_cycle:
                    current_market_cycle = new_cycle
                    memory = load_memory()
                    memory = apply_cycle_strategy_adjustment(memory, current_market_cycle)
                    print(f"市场周期已更新: {current_market_cycle}")
                    # 周期变化时通知冷却时间也跟着变了
                    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
                    print(f"信号冷却时间已调整为: {cooldown_sec // 60} 分钟")
                last_cycle_check = now

            verify_past_signals(config)

            # 动态扩展监控币种
            hot_coins = [c for c, _ in scan_hot_coins()[:MAX_DYNAMIC_COINS]]
            coins     = config["coins"].copy()
            for h in hot_coins:
                if hot_coin_filter(h) and h not in coins:
                    coins.append(h)

            memory = load_memory()
            # 未训练时固定使用宽松阈值55/45
            if not ai_model.is_trained:
                config["buy_threshold"]  = 55
                config["sell_threshold"] = 45
            else:
                # ── 百分位数动态阈值（GPT方案一核心）──
                # 用最近500条历史评分自动计算阈值
                # 牛市自动升高，熊市自动降低，完全自适应
                dyn_buy, dyn_sell = calc_dynamic_threshold(_score_history)
                config["buy_threshold"]  = dyn_buy
                config["sell_threshold"] = dyn_sell

                # 同步保存到memory，供模型训练等模块使用
                memory["buy_threshold"]  = dyn_buy
                memory["sell_threshold"] = dyn_sell
                save_memory(memory)

                print(f"[动态阈值] 历史评分{len(_score_history)}条 → "
                      f"买入={dyn_buy} 卖出={dyn_sell}")

            # DEBUG每10轮打印一次，避免日志刷屏
            _loop_count = getattr(main, '_loop_count', 0) + 1
            main._loop_count = _loop_count
            if _loop_count % 10 == 1:
                print(f"[监控] 第{_loop_count}轮 | 阈值: 买入={config['buy_threshold']} 卖出={config['sell_threshold']} | 周期:{current_market_cycle} | 模型:{'已训练' if ai_model.is_trained else '未训练'}")

            # 未训练时缩短冷却时间到15分钟，加速积累训练数据
            # 已训练后恢复按市场周期的动态冷却时间
            if not ai_model.is_trained:
                signal_cooldown = 1800   # 30分钟，加速数据积累（15m系统节奏）
            else:
                signal_cooldown = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)

            # 本轮评分缓存，供信号循环和状态推送共用，避免重复计算
            current_round_scores  = {}
            current_round_factors = {}

            # ======= 核心信号循环 =======
            for coin in coins:
                try:
                    df    = get_kline(coin, interval="15m", limit=200)
                    whale = detect_whale(coin)
                    score, factors, features = calculate_score(
                        df, memory, whale, current_market_cycle, coin, config)
                    analysis = factors['analysis']
                    # 缓存本轮评分供状态推送使用
                    current_round_scores[coin]  = score
                    current_round_factors[coin] = factors

                    # 收集历史评分用于百分位数动态阈值计算
                    _score_history.append(score)
                    if len(_score_history) > _SCORE_HISTORY_MAX:
                        _score_history.pop(0)

                    if score >= config["buy_threshold"]:
                        base_signal  = "买入"
                        signal_grade = "strong"   # 强信号
                    elif score <= config["sell_threshold"]:
                        base_signal  = "卖出"
                        signal_grade = "strong"   # 强信号
                    elif (ai_model.is_trained and
                          score >= config["buy_threshold"] - 5 and
                          score < config["buy_threshold"]):
                        # 次级买入：评分接近阈值但未达到，加⚠️标注
                        base_signal  = "买入"
                        signal_grade = "weak"
                    elif (ai_model.is_trained and
                          score <= config["sell_threshold"] + 5 and
                          score > config["sell_threshold"]):
                        # 次级卖出：评分接近阈值但未达到，加⚠️标注
                        base_signal  = "卖出"
                        signal_grade = "weak"
                    else:
                        base_signal  = "中性"
                        signal_grade = "none"
                        reset_signal_confirm(coin)
                        continue

                    # 连续确认机制
                    # 未训练时只需1次确认（加速积累数据）
                    # 已训练后需要2次确认（提高信号质量）
                    required_confirms = 1 if not ai_model.is_trained else SIGNAL_CONFIRM_COUNT
                    confirmed, avg_score = check_signal_confirm(coin, base_signal, score)
                    confirm_entry = _get_confirm_cache().get(coin, {})
                    if confirm_entry.get('count', 0) < required_confirms:
                        print(f"{coin} {base_signal}信号等待二次确认 ({confirm_entry.get('count',0)}/{required_confirms}次，当前评分:{score})")
                        continue

                    # 修改三：使用动态冷却时间
                    if now - last_signal_time.get(coin, 0) < signal_cooldown:
                        print(f"{coin} 信号冷却中（{signal_cooldown//60}分钟内不重复发送），跳过")
                        continue
                    last_signal_time[coin] = now
                    save_signal_time(last_signal_time)  # 立即持久化，防止并发容器重复发送

                    price = get_ticker(coin) or df["close"].iloc[-1]

                    # 记录日志（仅自定义币种）
                    if coin in config["coins"]:
                        log_signal(coin, base_signal, score, price, whale,
                                   current_market_cycle, factors, features)
                        print(f"📝 信号已记录: {coin} {base_signal} 评分{score}")
                    else:
                        print(f"📝 热门币种 {coin} 信号不计入训练数据")

                    # 修改一：使用新的买入过滤逻辑
                    if base_signal == "买入":
                        ok, reason = check_buy_filters(coin, df, memory)
                        if not ok:
                            print(f"{coin} 买入信号已过滤: {reason}")
                            continue

                    risk_level, risks = generate_risk_analysis(analysis, factors, config)

                    if score >= SIGNAL_STRONG_BUY:
                        display_signal = "🚀强看多"
                    elif score >= SIGNAL_BUY:
                        display_signal = "📈偏多"
                    elif score <= SIGNAL_STRONG_SELL:
                        display_signal = "💥强看空"
                    else:
                        display_signal = "📉偏空"

                    print(f"{coin} {display_signal} {score:.1f}分 "
                          f"({factors['up_prob']*100:.1f}%) {current_market_cycle} 价格:{price}")

                    msg = build_signal_message(
                        coin, base_signal, score, price,
                        factors, analysis, risk_level, risks,
                        required_confirms, avg_score, config,
                        signal_grade=signal_grade
                    )
                    subject = f"{coin} {base_signal}信号" if signal_grade == "strong"                               else f"⚠️ {coin} {base_signal}次级信号（谨慎参考）"
                    send_notification(msg, config, subject)
                    reset_signal_confirm(coin)

                except Exception as e:
                    print(f"处理{coin}时出错: {e}")
                    import traceback; traceback.print_exc()

            # ======= 定时任务 =======
            if now - last_adaptive_time > ADAPTIVE_OPTIMIZATION_INTERVAL:
                adaptive_strategy_optimization(config)
                last_adaptive_time = now

            # 修改二：回测报告发送后持久化时间戳
            if now - last_backtest_time > BACKTEST_INTERVAL:
                send_notification(generate_backtest_report(), config, "每日回测报告")
                last_backtest_time = now
                save_timing_state({
                    'last_backtest_time': last_backtest_time,
                    'last_daily_report':  last_daily_report,
                    'last_status_push':   last_status_push
                })

            if now - last_status_push > STATUS_PUSH_INTERVAL:
                # 直接复用主循环已计算的评分，不再重复调用 calculate_score
                current_scores = {c: current_round_scores.get(c) for c in coins}

                # 评分是否有明显变化
                score_changed = (not last_scores) or any(
                    last_scores.get(c) is None or
                    (s is not None and abs(s - last_scores.get(c, 0)) >= SCORE_CHANGE_THRESHOLD)
                    for c, s in current_scores.items()
                )

                cur_interval  = int(now / STATUS_PUSH_INTERVAL)
                last_interval = int(last_status_push / STATUS_PUSH_INTERVAL) \
                                if last_status_push > 0 else -1

                # 超过30分钟没推送，强制发一次（熊市横盘评分不变，但要让用户知道系统在运行）
                force_by_interval = (now - last_forced_status > FORCE_STATUS_INTERVAL)

                # 满足任一条件就推送：启动强制、30分钟到了、评分有变化
                if force_push_on_start or force_by_interval or (score_changed and cur_interval != last_interval):
                    status = build_status_message(coins, memory, config,
                                                  current_round_scores, current_round_factors)
                    send_telegram_message(status, config)
                    last_scores         = current_scores.copy()
                    force_push_on_start = False     # 只在启动时强制一次
                    last_forced_status  = now        # 记录本次强制推送时间

                last_status_push = now
                # 状态推送时间持久化，重启后不重复推送
                save_timing_state({
                    'last_backtest_time': last_backtest_time,
                    'last_daily_report':  last_daily_report,
                    'last_status_push':   last_status_push
                })

            # 每日报告已合并到 BACKTEST_INTERVAL，此处同步更新last_daily_report即可
            if now - last_daily_report > DAILY_REPORT_INTERVAL:
                last_daily_report = now
                save_timing_state({
                    'last_backtest_time': last_backtest_time,
                    'last_daily_report':  last_daily_report,
                    'last_status_push':   last_status_push
                })

        except Exception as e:
            print(f"主循环运行异常: {e}")
            import traceback; traceback.print_exc()

        # 每轮循环更新心跳，证明进程仍存活
        update_lock_heartbeat()

        # 安全sleep：最小300秒（15m系统每15分钟扫一次足够），防止check_interval异常值导致疯狂循环
        sleep_time = max(300, int(config.get("check_interval", 900)))
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ 收到中断信号，系统正在退出...")
    except Exception as e:
        print(f"❌ 系统异常退出: {e}")
        import traceback; traceback.print_exc()
    finally:
        # 无论正常退出还是异常退出，都释放启动锁
        release_startup_lock()
