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
# 自动适配 Railway Volume
# =========================
DATA_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/app/data")
if not os.path.exists(DATA_DIR):
    print(f"⚠️ Volume路径不存在，fallback到本地data目录")
    DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
print(f"📁 数据目录: {DATA_DIR}")

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
    print("提示: 未安装 scikit-optimize，贝叶斯优化功能将禁用。")

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

MEMORY_PATH        = os.path.join(DATA_DIR, MEMORY_FILE)
LOG_PATH           = os.path.join(DATA_DIR, LOG_FILE)
MODEL_PATH         = os.path.join(DATA_DIR, MODEL_FILE)
SCALER_PATH        = os.path.join(DATA_DIR, SCALER_FILE)
FEATURES_PATH      = os.path.join(DATA_DIR, FEATURES_FILE)
SIGNAL_CONFIRM_PATH= os.path.join(DATA_DIR, SIGNAL_CONFIRM_FILE)
TIMING_PATH        = os.path.join(DATA_DIR, TIMING_FILE)

# 时间间隔
STATUS_PUSH_INTERVAL          = 300
DAILY_REPORT_INTERVAL         = 86400
BACKTEST_INTERVAL             = 86400
ADAPTIVE_OPTIMIZATION_INTERVAL= 3600
MARKET_CYCLE_INTERVAL         = 3600

# 修改三：按市场周期动态冷却时间（秒）
# 牛市行情快，缩短冷却避免错过机会
# 熊市保守，延长冷却避免频繁抄底
# 震荡使用默认30分钟
COOLDOWN_BY_CYCLE = {
    "牛市": 900,    # 15分钟
    "震荡": 1800,   # 30分钟
    "熊市": 3600,   # 60分钟
    "未知": 1800    # 默认30分钟
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
PROFIT_THRESHOLD  = 0.015

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
            'last_daily_report':  0
        }

def save_timing_state(state):
    """原子保存定时状态"""
    tmp = TIMING_PATH + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        shutil.move(tmp, TIMING_PATH)
    except Exception as e:
        print(f"⚠️ save_timing_state 失败: {e}")

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
            print(f"请求失败 {r.status_code}: {url}")
        except Exception as e:
            print(f"请求异常: {e}")
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

@cache(ttl_seconds=30, max_size=64)
def get_kline(inst, interval="5m", limit=300):
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
        df_btc     = get_kline("BTC-USDT", interval="5m", limit=60)
        btc_change = (df_btc['close'].iloc[-1] - df_btc['close'].iloc[-12]) \
                     / df_btc['close'].iloc[-12]   # 近1小时涨跌幅（12根5分钟K线）

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
        print(f"BTC主导度判断失败: {e}")
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
        print(f"⚠️ save_signal_confirm 失败: {e}")

def check_signal_confirm(coin, signal_type, score):
    confirms = load_signal_confirm()
    entry    = confirms.get(coin, {'signal': None, 'count': 0, 'scores': []})
    if entry['signal'] != signal_type:
        entry = {'signal': signal_type, 'count': 1, 'scores': [score]}
    else:
        entry['count'] += 1
        entry['scores'].append(score)
        entry['scores'] = entry['scores'][-SIGNAL_CONFIRM_COUNT:]
    confirms[coin] = entry
    save_signal_confirm(confirms)
    if entry['count'] >= SIGNAL_CONFIRM_COUNT:
        return True, float(np.mean(entry['scores']))
    return False, score

def reset_signal_confirm(coin):
    confirms = load_signal_confirm()
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
        features['coin_id']      = abs(hash(coin_name)) % 100 / 100

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
                df_btc = get_kline("BTC-USDT", interval="5m", limit=100)
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

        dt = datetime.now()
        features['hour']       = dt.hour
        features['weekday']    = dt.weekday()
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0

        sentiment = get_market_sentiment()
        if sentiment:
            features.update(sentiment)

        funding = get_funding_rate(coin_name)
        if funding is not None:
            features['funding_rate'] = funding

        return features

    def prepare_training_data(self, logs):
        X_list, y_list = [], []
        for log in logs:
            if not log.get('verified') or log.get('result') not in ['correct', 'wrong']:
                continue
            if 'features' not in log or not log['features']:
                continue
            X_list.append(log['features'])
            y_list.append(1 if log['result'] == 'correct' else 0)
        if len(X_list) < MIN_TRAIN_SAMPLES:
            return None, None
        X_df = pd.DataFrame(X_list)
        self.feature_names = X_df.columns.tolist()
        return X_df, np.array(y_list)

    def train(self, X, y):
        if X is None or len(X) < MIN_TRAIN_SAMPLES:
            print(f"训练样本不足: {len(X) if X is not None else 0}/{MIN_TRAIN_SAMPLES}")
            return False

        if isinstance(X, pd.DataFrame):
            orig = len(X)
            X    = X.dropna()
            y    = y[X.index] if hasattr(y, '__getitem__') else y[:len(X)]
            if len(X) < MIN_TRAIN_SAMPLES:
                print(f"清洗后样本不足{MIN_TRAIN_SAMPLES}，跳过训练")
                return False
            if len(X) < orig:
                print(f"删除了{orig - len(X)}行NaN样本")

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
            print(f"⚠️ 过拟合警告: 训练={train_acc:.3f}, CV={cv_acc:.3f}, 差距={overfit_gap:.3f}")

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

        print(f"✅ 训练完成: 样本={len(X)}, 训练准确率={train_acc:.4f}, CV准确率={cv_acc:.4f}")
        if self.feature_importance:
            top5 = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top5特征: {top5}")
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
            print("✅ 模型原子保存成功")
            return True
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
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
            print(f"⚠️ 模型文件缺失: {missing}")
            return False
        try:
            self.model = joblib.load(MODEL_PATH)
            print("  ✔ model 加载成功")
        except Exception as e:
            print(f"  ✘ model 加载失败: {e}"); return False
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("  ✔ scaler 加载成功")
        except Exception as e:
            print(f"  ✘ scaler 加载失败: {e}"); return False
        try:
            with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.feature_names      = data.get('feature_names', [])
            self.training_history   = data.get('training_history', [])
            self.feature_importance = data.get('feature_importance', {})
            self.cv_accuracy        = data.get('cv_accuracy', 0.0)
            print(f"  ✔ features 加载成功，特征数: {len(self.feature_names)}")
        except Exception as e:
            print(f"  ✘ features 加载失败: {e}"); return False
        self.is_trained = True
        last = self.training_history[-1] if self.training_history else {}
        print(f"✅ 模型加载成功 | CV准确率={self.cv_accuracy:.4f} | 样本={last.get('samples','?')}")
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
        "check_interval": 300,
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
        print(f"⚠️ save_memory 失败: {e}")

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
        print(f"Telegram发送失败: {e}")

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
            print(f"邮件 {host}:{port} 失败: {e}")

def send_notification(content, config, subject=None):
    send_telegram_message(content, config)
    if subject:
        send_email(subject, content, config)

# =========================
# 市场周期
# =========================
def detect_market_cycle():
    try:
        df      = get_kline("BTC-USDT", interval="5m", limit=300)
        ma200   = df["close"].rolling(200).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        if pd.isna(ma200):
            return "未知"
        momentum = (price - df["close"].iloc[-30]) / df["close"].iloc[-30]
        if price > ma200 and momentum > 0.05:
            return "牛市"
        elif price < ma200:
            return "熊市"
        else:
            return "震荡"
    except Exception as e:
        print(f"周期识别失败: {e}")
        return "未知"

def apply_cycle_strategy_adjustment(memory, cycle):
    memory = memory.copy()
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
                      'position_desc': '中等', 'volume_ratio': 1.0}
    else:
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        ma20       = df["ma20"].iloc[-1]
        ma60       = df["ma60"].iloc[-1]
        momentum   = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

        rule_score = 50
        if ma20 > ma60:
            rule_score += 20 * memory.get("trend_weight", 0.3)
        if momentum > 0.02:
            rule_score += 15 * memory.get("momentum_weight", 0.25)

        volume    = df["volume"].iloc[-1]
        avg_volume= df["volume"].mean()
        vol_ratio = volume / avg_volume if avg_volume != 0 else 1
        if vol_ratio > 1.5:
            rule_score += 10 * memory.get("volume_weight", 0.2)
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
            print(f"ML预测失败: {e}")
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

    factors = {
        'rule_score':      rule_score,
        'ml_score':        ml_score,
        'ml_confidence':   ml_confidence,
        'combined_score':  combined_score,
        'up_prob':         max(0, min(1, up_prob)),
        'analysis':        analysis,
        'atr':             atr,
        'stop_loss_buy':   round(cur_price - atr * 2, 6),
        'take_profit_buy': round(cur_price + atr * 3, 6),
        'stop_loss_sell':  round(cur_price + atr * 2, 6),
        'take_profit_sell':round(cur_price - atr * 3, 6),
    }
    return int(combined_score), factors, features

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
        print(f"⚠️ save_log 失败: {e}")

def verify_past_signals(config):
    log     = load_log()
    updated = False
    for entry in log:
        if entry.get("verified", False):
            continue
        coin        = entry["coin"]
        signal_time = entry["timestamp"]
        try:
            df = get_kline(coin, interval="5m", limit=10)
            df['ts_sec'] = df['ts'] / 1000
            future_df    = df[df['ts_sec'] > signal_time].sort_values('ts_sec')
            if len(future_df) < 5:
                continue
            if entry["signal"] == "买入":
                profit = (future_df['high'].iloc[:5].max() - entry["price"]) / entry["price"]
            elif entry["signal"] == "卖出":
                profit = (entry["price"] - future_df['low'].iloc[:5].min()) / entry["price"]
            else:
                continue
            entry["verified"] = True
            entry["result"]   = "correct" if profit > PROFIT_THRESHOLD else "wrong"
            entry["profit"]   = round(profit * 100, 3)
            updated = True
        except Exception as e:
            print(f"验证{coin}失败: {e}")
    if updated:
        save_log(log)

def log_signal(coin, signal, score, price, whale, market_cycle, factors, features):
    log   = load_log()
    entry = {
        "timestamp":     time.time(),
        "coin":          coin,
        "signal":        signal,
        "score":         score,
        "price":         price,
        "whale":         whale,
        "market_cycle":  market_cycle,
        "rule_score":    factors.get('rule_score', 50),
        "ml_score":      factors.get('ml_score', 50),
        "ml_confidence": factors.get('ml_confidence', 0),
        "features":      features,
        "verified":      False,
        "result":        None,
        "profit":        None
    }
    log.append(entry)
    if len(log) > MAX_LOG_SIZE:
        verified   = [e for e in log if e.get("verified")]
        unverified = [e for e in log if not e.get("verified")]
        keep       = MAX_LOG_SIZE - len(unverified)
        log        = (verified[-keep:] if keep > 0 else []) + unverified
    save_log(log)

# =========================
# 自适应优化
# =========================
def adaptive_strategy_optimization(config):
    log      = load_log()
    verified = [e for e in log if e.get("verified") and e.get("result") in ("correct", "wrong")]
    if len(verified) < MIN_TRAIN_SAMPLES:
        print(f"自适应优化：已验证{len(verified)}/{MIN_TRAIN_SAMPLES}条，跳过")
        return

    X_df, y = ai_model.prepare_training_data(verified)
    if X_df is None:
        print("特征提取后样本不足")
        return

    success = ai_model.train(X_df, y)
    if not success:
        return

    ai_model.save()
    cv_acc = ai_model.cv_accuracy
    memory = load_memory()
    memory['feature_importance'] = ai_model.feature_importance

    if cv_acc > 0.60:
        memory['buy_threshold']  = min(memory.get('buy_threshold', 70) + 2, 85)
        memory['sell_threshold'] = max(memory.get('sell_threshold', 35) - 2, 20)
        memory['ml_weight']      = min(0.7, memory.get('ml_weight', 0.4) + 0.05)
    elif cv_acc < 0.50:
        memory['buy_threshold']  = max(memory.get('buy_threshold', 70) - 2, 55)
        memory['sell_threshold'] = min(memory.get('sell_threshold', 35) + 2, 45)
        memory['ml_weight']      = max(0.1, memory.get('ml_weight', 0.4) - 0.1)
        print(f"⚠️ CV准确率偏低({cv_acc:.3f})，降低ML权重至{memory['ml_weight']:.2f}")

    memory['buy_threshold']  = max(min(memory['buy_threshold'], 85), memory['sell_threshold'] + 5)
    memory['sell_threshold'] = max(min(memory['sell_threshold'], 50), 15)
    save_memory(memory)

    recent_verified = sorted(verified, key=lambda x: x.get('timestamp', 0), reverse=True)[:100]
    recent_correct  = sum(1 for e in recent_verified if e.get('result') == 'correct')
    real_winrate    = recent_correct / len(recent_verified) if recent_verified else 0

    train_acc = ai_model.training_history[-1]['train_acc']
    overfit_warning = ""
    if train_acc - cv_acc > 0.15:
        overfit_warning = f"\n⚠️ 检测到过拟合(差距{train_acc - cv_acc:.2%})，已自动降低模型权重"

    top5        = sorted(ai_model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    feature_msg = "\n".join([f"  {f}: {v:.4f}" for f, v in top5]) if top5 else "  （暂无）"

    msg = (f"🤖 AI模型已进化\n"
           f"训练样本: {len(X_df)}\n"
           f"训练准确率: {train_acc:.2%}\n"
           f"CV准确率: {cv_acc:.2%}  ← 真实参考值\n"
           f"实际胜率(近100条): {real_winrate:.2%}\n"
           f"新阈值: 买入={memory['buy_threshold']}, 卖出={memory['sell_threshold']}\n"
           f"ML权重: {memory['ml_weight']:.2f}\n"
           f"Top5特征:\n{feature_msg}"
           f"{overfit_warning}")
    send_notification(msg, config, "AI模型进化报告")

# =========================
# 回测 & 统计
# =========================
def generate_backtest_report():
    log      = load_log()
    verified = [e for e in log if e.get("verified")]
    if not verified:
        return "暂无已验证数据"
    total      = len(verified)
    correct    = sum(1 for e in verified if e.get("result") == "correct")
    profits    = [e.get('profit', 0) for e in verified if e.get('profit') is not None]
    avg_profit = np.mean(profits) if profits else 0
    return (f"📊 回测报告\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"总信号: {total}\n"
            f"正确: {correct} | 错误: {total - correct}\n"
            f"胜率: {correct/total:.2%}\n"
            f"平均利润: {avg_profit:.3f}%")

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
    base       = coin.split("-")[0].upper()
    is_major   = base in MAJOR_COINS
    btc_info   = get_btc_dominance_trend()

    if is_major:
        # 主流币：BTC整体下跌趋势时过滤
        if btc_info['btc_falling']:
            return False, f"BTC均线死叉下跌，主流币{coin}过滤"
    else:
        # 山寨币：BTC走独立行情时才过滤（资金被虹吸）
        # BTC普跌不过滤山寨币（山寨可能有独立行情）
        if btc_info['btc_independent']:
            return False, (f"BTC独立拉升中"
                           f"(BTC+{btc_info['btc_change']*100:.1f}% vs 大盘"
                           f"{btc_info['market_avg']*100:.1f}%)，资金虹吸，山寨暂缓")

    # 成交量过滤（所有币种共用）
    volume  = df['volume'].iloc[-1]
    avg_vol = df['volume'].mean()
    if volume < avg_vol * VOLUME_RATIO_MIN:
        return False, f"成交量不足({volume:.0f}<{avg_vol*VOLUME_RATIO_MIN:.0f})"

    # 波动率过滤（所有币种共用）
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
                          confirm_count, avg_score, config):
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
    model_reliability = "⚠️ 模型未训练" if not ai_model.is_trained else \
                        (f"✅ 可靠(CV:{cv_acc:.0%})" if cv_acc >= 0.55 else
                         f"⚠️ 低可信(CV:{cv_acc:.0%})，仅供参考")

    # 修改三：显示当前冷却时间
    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
    cooldown_str = f"{cooldown_sec // 60}分钟"

    signal_emoji = "🟢" if base_signal == "买入" else "🔴"
    risk_emoji   = "🔴" if risk_level == "高" else ("🟡" if risk_level == "中" else "🟢")

    msg = (
        f"{signal_emoji} <b>{coin} {base_signal}信号</b>\n\n"
        f"💰 价格：${display_price:.6g}\n"
        f"📊 综合评分：{score} (规则:{factors['rule_score']} ML:{factors['ml_score']:.0f})\n"
        f"🎯 上涨概率：{up_prob:.1f}%\n"
        f"🔁 连续确认：{confirm_count}次 (均分{avg_score:.0f})\n\n"
        f"📈 趋势：{analysis['trend']} | 动量：{analysis['momentum']}\n"
        f"📦 成交量：{analysis['volume']} | 波动：{analysis['volatility']}\n"
        f"📍 位置：{analysis['position_desc']}\n\n"
        f"🛡️ 止损价：${sl:.6g} (-{sl_pct:.1f}%)\n"
        f"🎯 目标价：${tp:.6g} (+{tp_pct:.1f}%)\n"
        f"💼 建议仓位：{position_suggest}\n\n"
        f"{risk_emoji} 风险等级：{risk_level}\n"
        f"⚠️ 风险点：{'、'.join(risks) if risks else '无明显风险'}\n\n"
        f"🤖 模型：{model_reliability}\n"
        f"🌍 市场周期：{current_market_cycle}（冷却{cooldown_str}）\n"
        f"⏰ {datetime.now().strftime('%m-%d %H:%M')}"
    )
    return msg

# =========================
# 状态推送
# =========================
def build_status_message(coins, memory, config):
    now    = time.time()
    uptime = int(now - start_time)
    h, m   = uptime // 3600, (uptime % 3600) // 60

    if ai_model.is_trained and ai_model.training_history:
        last      = ai_model.training_history[-1]
        train_acc = last.get('train_acc', last.get('accuracy', 0))
        cv_acc    = last.get('cv_accuracy', ai_model.cv_accuracy)
        gap       = train_acc - cv_acc
        overfit_tag = f" ⚠️过拟合" if gap > 0.15 else ""
        model_info  = (f"✅ 已训练 | 样本:{last['samples']}\n"
                       f"     训练准确率:{train_acc:.0%} CV:{cv_acc:.0%}{overfit_tag}")
    else:
        model_info = "❌ 未训练（积累中...）"

    today_total, today_correct = get_signal_stats_since(24)
    today_winrate = today_correct / today_total if today_total > 0 else 0
    winrate_tag   = " ⚠️偏低" if today_total >= 10 and today_winrate < 0.4 else ""

    sixh_total, sixh_correct = get_signal_stats_since(6)
    sixh_winrate = sixh_correct / sixh_total if sixh_total > 0 else 0

    # 修改三：显示当前周期对应的冷却时间
    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
    cooldown_str = f"{cooldown_sec // 60}分钟"

    coin_lines = []
    for coin in coins:
        try:
            df      = get_kline(coin, interval="5m", limit=300)
            whale   = detect_whale(coin)
            score, factors, _ = calculate_score(df, memory, whale,
                                                current_market_cycle, coin, config)
            analysis  = factors['analysis']
            price     = get_ticker(coin) or df['close'].iloc[-1]
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

    status = (
        f"📡 <b>AI监控状态报告</b>\n\n"
        f"📈 监控币种:\n" + "\n".join(coin_lines) + "\n\n"
        f"⏱️ 最近信号:\n" +
        ("\n".join([f"  {s}" for s in recent]) if recent else "  暂无") + "\n\n"
        f"🔥 市场热点:\n{hot_str}\n\n"
        f"🤖 模型状态:\n  {model_info}\n\n"
        f"📊 胜率统计:\n"
        f"  6小时: {sixh_total}次 | {sixh_winrate:.0%}\n"
        f"  24小时: {today_total}次 | {today_winrate:.0%}{winrate_tag}\n\n"
        f"⚙️ 阈值: 买入{config['buy_threshold']} | 卖出{config['sell_threshold']}\n"
        f"⏳ 信号冷却: {cooldown_str}（{current_market_cycle}）\n"
        f"🕐 运行: {h}小时{m}分钟 | {current_market_cycle}"
    )
    return status

# =========================
# 主程序
# =========================
def main():
    global last_backtest_time, last_adaptive_time, last_cycle_check
    global last_status_push, last_daily_report, last_signal_time, last_scores
    global current_market_cycle, start_time

    config     = load_config()
    start_time = time.time()
    print("=" * 50)
    print("AI自主学习交易系统启动 (Gate.io 决策辅助版)")
    print(f"模型状态: {'已训练 CV=' + str(round(ai_model.cv_accuracy, 4)) if ai_model.is_trained else '未训练'}")
    print("=" * 50)

    # 修改二：从文件恢复定时状态，重启不重复推送
    timing_state     = load_timing_state()
    last_backtest_time = timing_state.get('last_backtest_time', 0)
    last_daily_report  = timing_state.get('last_daily_report',  0)
    print(f"📅 上次回测推送: "
          f"{datetime.fromtimestamp(last_backtest_time).strftime('%Y-%m-%d %H:%M') if last_backtest_time > 0 else '从未'}")
    print(f"📅 上次日报推送: "
          f"{datetime.fromtimestamp(last_daily_report).strftime('%Y-%m-%d %H:%M') if last_daily_report > 0 else '从未'}")

    if not ai_model.is_trained:
        memory = load_memory()
        memory["buy_threshold"]  = 65
        memory["sell_threshold"] = 35
        save_memory(memory)
        config["buy_threshold"]  = 65
        config["sell_threshold"] = 35
        print("⚠️ 模型未训练，使用保守阈值: 买入65 / 卖出35")
    else:
        print("✅ 模型已训练，使用动态阈值")

    # 启动通知
    log_size = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    send_telegram_message(
        f"🚀 系统启动\n"
        f"模型: {'✅已训练' if ai_model.is_trained else '❌未训练'}\n"
        f"日志大小: {log_size} 字节\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        config
    )

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
                    print(f"市场周期更新: {current_market_cycle}")
                    # 周期变化时通知冷却时间也跟着变了
                    cooldown_sec = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)
                    print(f"信号冷却时间调整为: {cooldown_sec // 60}分钟")
                last_cycle_check = now

            verify_past_signals(config)

            # 动态扩展监控币种
            hot_coins = [c for c, _ in scan_hot_coins()[:MAX_DYNAMIC_COINS]]
            coins     = config["coins"].copy()
            for h in hot_coins:
                if hot_coin_filter(h) and h not in coins:
                    coins.append(h)

            memory = load_memory()
            config["buy_threshold"]  = memory.get("buy_threshold",  config["buy_threshold"])
            config["sell_threshold"] = memory.get("sell_threshold", config["sell_threshold"])
            print(f"[DEBUG] 阈值: 买入={config['buy_threshold']}, 卖出={config['sell_threshold']}")

            # 修改三：获取当前周期对应的冷却时间
            signal_cooldown = COOLDOWN_BY_CYCLE.get(current_market_cycle, 1800)

            # ======= 核心信号循环 =======
            for coin in coins:
                try:
                    df    = get_kline(coin, interval="5m", limit=300)
                    whale = detect_whale(coin)
                    score, factors, features = calculate_score(
                        df, memory, whale, current_market_cycle, coin, config)
                    analysis = factors['analysis']

                    if score >= config["buy_threshold"]:
                        base_signal = "买入"
                    elif score <= config["sell_threshold"]:
                        base_signal = "卖出"
                    else:
                        base_signal = "中性"
                        reset_signal_confirm(coin)
                        continue

                    # 连续确认机制
                    confirmed, avg_score = check_signal_confirm(coin, base_signal, score)
                    if not confirmed:
                        print(f"{coin} {base_signal}信号等待确认({score}分)...")
                        continue

                    # 修改三：使用动态冷却时间
                    if now - last_signal_time.get(coin, 0) < signal_cooldown:
                        print(f"{coin} 信号冷却中({signal_cooldown//60}分钟)，跳过")
                        continue
                    last_signal_time[coin] = now

                    price = get_ticker(coin) or df["close"].iloc[-1]

                    # 记录日志（仅自定义币种）
                    if coin in config["coins"]:
                        log_signal(coin, base_signal, score, price, whale,
                                   current_market_cycle, factors, features)
                        print(f"📝 记录信号: {coin} {base_signal} 评分{score}")
                    else:
                        print(f"📝 热门币 {coin} 信号不记录训练日志")

                    # 修改一：使用新的买入过滤逻辑
                    if base_signal == "买入":
                        ok, reason = check_buy_filters(coin, df, memory)
                        if not ok:
                            print(f"{coin} 买入被过滤: {reason}")
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
                        SIGNAL_CONFIRM_COUNT, avg_score, config
                    )
                    send_notification(msg, config, f"{coin} {base_signal}信号")
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
                    'last_daily_report':  last_daily_report
                })

            if now - last_status_push > STATUS_PUSH_INTERVAL:
                current_scores = {}
                for coin in coins:
                    try:
                        df    = get_kline(coin, interval="5m", limit=300)
                        score, _, _ = calculate_score(df, memory, detect_whale(coin),
                                                      current_market_cycle, coin, config)
                        current_scores[coin] = score
                    except Exception:
                        current_scores[coin] = None

                need_push = (not last_scores) or any(
                    last_scores.get(c) is None or
                    (s is not None and abs(s - last_scores.get(c, 0)) >= SCORE_CHANGE_THRESHOLD)
                    for c, s in current_scores.items()
                )

                cur_interval  = int(now / STATUS_PUSH_INTERVAL)
                last_interval = int(last_status_push / STATUS_PUSH_INTERVAL) \
                                if last_status_push > 0 else -1

                if need_push and cur_interval != last_interval:
                    status = build_status_message(coins, memory, config)
                    send_telegram_message(status, config)
                    last_scores = current_scores.copy()

                last_status_push = now

            # 修改二：日报发送后持久化时间戳
            if now - last_daily_report > DAILY_REPORT_INTERVAL:
                send_notification(generate_backtest_report(), config, "每日回测报告")
                last_daily_report = now
                save_timing_state({
                    'last_backtest_time': last_backtest_time,
                    'last_daily_report':  last_daily_report
                })

        except Exception as e:
            print(f"主循环异常: {e}")
            import traceback; traceback.print_exc()

        time.sleep(config["check_interval"])


if __name__ == "__main__":
    main()
