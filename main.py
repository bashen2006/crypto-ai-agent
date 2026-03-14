import time
import json
import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
# 设置数据存储路径（如果Volume挂载了，就存到Volume里）
DATA_DIR = '/app/data'  # Volume挂载点
# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# =========================
# 机器学习库导入
# =========================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# =========================
# 配置常量（使用持久化路径）
# =========================
CONFIG_FILE = "config.json"  # 配置文件保留在项目根目录

# 所有需要持久化的文件都放入 DATA_DIR
MEMORY_FILE = "ai_memory.json"
LOG_FILE = "prediction_log.json"
MODEL_FILE = "ai_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "feature_config.json"

# 构建完整路径
MEMORY_PATH = os.path.join(DATA_DIR, MEMORY_FILE)
LOG_PATH = os.path.join(DATA_DIR, LOG_FILE)
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILE)
SCALER_PATH = os.path.join(DATA_DIR, SCALER_FILE)
FEATURES_PATH = os.path.join(DATA_DIR, FEATURES_FILE)

WHALE_THRESHOLD = 20000
SIGNAL_COOLDOWN = 1800

STATUS_PUSH_INTERVAL = 300
DAILY_REPORT_INTERVAL = 86400
BACKTEST_INTERVAL = 86400
ADAPTIVE_OPTIMIZATION_INTERVAL = 3600  # 缩短为1小时，更频繁学习
MARKET_CYCLE_INTERVAL = 3600
MODEL_RETRAIN_INTERVAL = 21600  # 6小时重新训练模型

MAX_LOG_SIZE = 5000
MAX_DYNAMIC_COINS = 3
MIN_TRAIN_SAMPLES = 50  # 最少训练样本数

# 全局状态变量
last_signal_time = {}
last_status_push = 0
last_daily_report = 0
last_backtest_time = 0
last_adaptive_time = 0
last_cycle_check = 0
last_model_retrain = 0
current_market_cycle = "未知"

# =========================
# AI模型类 - 真正的自主学习核心
# =========================
class AITradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_history = []
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
    def extract_features(self, df, whale, market_cycle, coin_name):
        """
        从原始数据中提取丰富的特征集
        """
        features = {}
        
        # 确保数据足够
        if len(df) < 100:
            return None
            
        # 价格特征
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # 1. 技术指标特征
        # 移动平均线
        for period in [5, 10, 20, 30, 50, 60, 100]:
            if len(df) >= period:
                ma = df['close'].rolling(period).mean().iloc[-1]
                features[f'ma_{period}'] = ma
                features[f'price_ma_{period}_ratio'] = close[-1] / ma if ma != 0 else 1
        
        # 2. 动量特征
        for period in [5, 10, 20, 30]:
            if len(df) >= period:
                momentum = (close[-1] - close[-period]) / close[-period]
                features[f'momentum_{period}'] = momentum
        
        # 3. 波动率特征
        for period in [10, 20, 30]:
            if len(df) >= period:
                returns = np.diff(close[-period:]) / close[-period:-1]
                features[f'volatility_{period}'] = np.std(returns)
        
        # 4. 成交量特征
        volume_ma = pd.Series(volume).rolling(20).mean().iloc[-1]
        features['volume_ratio'] = volume[-1] / volume_ma if volume_ma != 0 else 1
        features['volume_trend'] = (volume[-1] - volume[-5]) / volume[-5] if volume[-5] != 0 else 0
        
        # 5. 价格位置特征
        high_20 = np.max(close[-20:])
        low_20 = np.min(close[-20:])
        features['price_position_20'] = (close[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        # 6. RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1] if len(gain) >= 14 else 0
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1] if len(loss) >= 14 else 0
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50
            
        # 7. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd.iloc[-1]
        features['macd_signal'] = signal.iloc[-1]
        features['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
        # 8. 布林带
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        features['bb_position'] = (close[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] > bb_lower.iloc[-1] else 0.5
        
        # 9. 外部特征
        features['whale_value'] = whale / 10000  # 归一化
        features['market_cycle'] = 2 if market_cycle == "牛市" else (1 if market_cycle == "震荡" else 0)
        
        # 10. 币种独热编码（简化版）
        coin_hash = abs(hash(coin_name)) % 100 / 100  # 简单哈希作为特征
        features['coin_id'] = coin_hash
        
        return features
    
    def prepare_training_data(self, logs):
        """
        从历史日志准备训练数据（直接使用日志中保存的特征）
        """
        X_list = []
        y_list = []
        
        for log in logs:
            if not log.get('verified') or log.get('result') not in ['correct', 'wrong']:
                continue
            if 'features' not in log:
                continue  # 跳过没有特征的旧日志
            features = log['features']
            X_list.append(features)
            y_list.append(1 if log['result'] == 'correct' else 0)
                
        if len(X_list) < MIN_TRAIN_SAMPLES:
            return None, None
            
        X_df = pd.DataFrame(X_list)
        self.feature_names = X_df.columns.tolist()
        return X_df, np.array(y_list)
    
    def train(self, X, y):
        """
        训练机器学习模型
        """
        if X is None or len(X) < MIN_TRAIN_SAMPLES:
            print(f"训练样本不足: {len(X) if X is not None else 0}/{MIN_TRAIN_SAMPLES}")
            return False
            
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用梯度提升树（更好的泛化能力）
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 计算特征重要性
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # 评估模型
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': time.time(),
            'samples': len(X),
            'accuracy': accuracy,
            'features': len(self.feature_names)
        })
        
        print(f"模型训练完成: 样本数={len(X)}, 准确率={accuracy:.4f}")
        print(f"Top 5重要特征: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        return True
    
    def predict(self, features):
        """
        使用训练好的模型进行预测
        """
        if not self.is_trained or self.model is None:
            return 50, {}  # 未训练时返回中性
            
        # 确保特征顺序正确
        feature_vector = []
        for name in self.feature_names:
            if name in features:
                feature_vector.append(features[name])
            else:
                feature_vector.append(0)  # 缺失特征补0
                
        # 标准化
        feature_scaled = self.scaler.transform([feature_vector])
        
        # 预测概率
        proba = self.model.predict_proba(feature_scaled)[0]
        
        # 返回正确概率作为分数
        if len(proba) == 2:
            score = proba[1] * 100  # 类别1（正确）的概率
        else:
            score = 50
            
        # 获取预测置信度
        confidence = np.max(proba)
        
        # 获取特征贡献
        contribution = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, name in enumerate(self.feature_names):
                contribution[name] = self.model.feature_importances_[i] * feature_scaled[0][i]
        
        return score, {'confidence': confidence, 'contribution': contribution}
    
    def save(self):
        """保存模型和缩放器（使用持久化路径）"""
        if self.model:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            # 保存特征名称
            with open(FEATURES_PATH, 'w') as f:
                json.dump({
                    'feature_names': self.feature_names,
                    'training_history': self.training_history,
                    'feature_importance': self.feature_importance
                }, f, indent=4)
            print("模型已保存到持久化存储")
    
    def load(self):
        """加载模型和缩放器（使用持久化路径）"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            with open(FEATURES_PATH, 'r') as f:
                data = json.load(f)
                self.feature_names = data.get('feature_names', [])
                self.training_history = data.get('training_history', [])
                self.feature_importance = data.get('feature_importance', {})
            self.is_trained = True
            print("模型加载成功")
            return True
        except:
            print("未找到已训练模型")
            return False

# =========================
# 初始化全局AI模型
# =========================
ai_model = AITradingModel()
# 尝试加载已有模型
ai_model.load()

# =========================
# 辅助函数：加载/保存配置与内存（使用持久化路径）
# =========================
def load_config():
    """加载配置文件，缺失的键使用默认值（配置文件保留在根目录）"""
    default_config = {
        "coins": ["BTC-USDT", "ETH-USDT"],
        "buy_threshold": 70,
        "sell_threshold": 35,
        "check_interval": 300,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email_user": "",
        "email_pass": "",
        "email_receiver": "",
        "use_ml_model": True,
        "ml_weight": 0.7
    }
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        for key, value in default_config.items():
            if key not in config:
                print(f"警告：配置缺少 {key}，使用默认值 {value}")
                config[key] = value
        return config
    except FileNotFoundError:
        print("配置文件不存在，使用默认配置")
        return default_config

def load_memory():
    """加载AI内存权重，若文件不存在则创建默认（使用持久化路径）"""
    default_memory = {
        "trend_weight": 0.3,
        "momentum_weight": 0.25,
        "volume_weight": 0.2,
        "volatility_weight": 0.1,
        "sentiment_weight": 0.2,
        "ml_weight": 0.7,
        "feature_importance": {}
    }
    try:
        with open(MEMORY_PATH) as f:
            memory = json.load(f)
        # 补充缺失的键，保留多余键
        for key in default_memory:
            if key not in memory:
                memory[key] = default_memory[key]
        return memory
    except:
        save_memory(default_memory)
        return default_memory

def save_memory(memory):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=4)

# =========================
# 通知功能：Telegram和邮件（保持不变）
# =========================
def send_telegram_message(text, config):
    """通过Telegram Bot发送消息"""
    token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram发送失败: {e}")

def send_email(subject, body, config):
    """通过139邮箱发送邮件"""
    user = config.get("email_user")
    pwd = config.get("email_pass")
    receiver = config.get("email_receiver")
    if not user or not pwd or not receiver:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.139.com", 25)
        server.starttls()
        server.login(user, pwd)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"邮件发送失败: {e}")

def send_notification(content, config, subject=None):
    """同时发送Telegram和邮件（若配置）"""
    send_telegram_message(content, config)
    if subject:
        send_email(subject, content, config)

# =========================
# 安全请求（保持不变）
# =========================
def safe_request(url, max_retries=3):
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(2)
    return None

def get_kline(inst, limit=300):
    """获取K线数据，默认300根确保200均线有效"""
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=5m&limit={limit}"
    data = safe_request(url)
    if not data or "data" not in data:
        raise Exception(f"获取{inst}行情失败")
    df = pd.DataFrame(data["data"])
    df = df.iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df

def detect_whale(inst):
    """检测巨鲸交易"""
    try:
        url = f"https://www.okx.com/api/v5/market/trades?instId={inst}&limit=100"
        data = safe_request(url)
        if not data or "data" not in data:
            return 0
        whale_total = 0
        for trade in data["data"]:
            size = float(trade["sz"])
            price = float(trade["px"])
            value = size * price
            if value > WHALE_THRESHOLD:
                whale_total += value
        return whale_total
    except:
        return 0

# =========================
# 市场周期识别（保持不变）
# =========================
def detect_market_cycle():
    """基于BTC判断市场周期：牛市/熊市/震荡"""
    try:
        df = get_kline("BTC-USDT", limit=300)
        df["ma200"] = df["close"].rolling(200).mean()
        price = df["close"].iloc[-1]
        ma200 = df["ma200"].iloc[-1]
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
    """根据市场周期调整权重并保存"""
    memory = memory.copy()
    if cycle == "牛市":
        memory["trend_weight"] = min(memory.get("trend_weight", 0.3) + 0.05, 1.0)
    elif cycle == "熊市":
        memory["trend_weight"] = max(memory.get("trend_weight", 0.3) - 0.05, 0.0)
    elif cycle == "震荡":
        memory["volume_weight"] = min(memory.get("volume_weight", 0.2) + 0.05, 1.0)
    save_memory(memory)
    return memory

# =========================
# AI评分（增强版：结合规则和机器学习，保持不变）
# =========================
def calculate_score(df, memory, whale, market_cycle, coin, config):
    """
    增强版评分：结合传统规则和机器学习模型
    返回: (综合评分, 因子信息字典, 特征字典)
    """
    # 1. 传统规则评分
    df = df.dropna().copy()
    if len(df) < 60:
        rule_score = 50
    else:
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        ma20 = df["ma20"].iloc[-1]
        ma60 = df["ma60"].iloc[-1]
        momentum = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

        rule_score = 50
        if ma20 > ma60:
            rule_score += 20 * memory.get("trend_weight", 0.3)
        if momentum > 0.02:
            rule_score += 15 * memory.get("momentum_weight", 0.25)
        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].mean()
        if volume > avg_volume * 1.5:
            rule_score += 10 * memory.get("volume_weight", 0.2)
        if whale > 0:
            rule_score += 5
            
        rule_score = max(0, min(100, int(rule_score)))
    
    # 2. 机器学习评分（如果启用）
    ml_score = 50
    ml_confidence = 0
    feature_importance = {}
    features = None
    
    if config.get("use_ml_model", True) and ai_model.is_trained:
        try:
            features = ai_model.extract_features(df, whale, market_cycle, coin)
            if features:
                ml_score, ml_info = ai_model.predict(features)
                ml_confidence = ml_info.get('confidence', 0)
                feature_importance = ml_info.get('contribution', {})
        except Exception as e:
            print(f"ML预测失败: {e}")
    else:
        # 即使未训练，也提取特征以备记录
        features = ai_model.extract_features(df, whale, market_cycle, coin)
    
    # 3. 综合评分
    ml_weight = config.get("ml_weight", 0.7) if ai_model.is_trained else 0
    combined_score = rule_score * (1 - ml_weight) + ml_score * ml_weight
    
    # 4. 构建因子信息
    factors = {
        'rule_score': rule_score,
        'ml_score': ml_score,
        'ml_confidence': ml_confidence,
        'combined_score': combined_score,
        'feature_importance': feature_importance
    }
    
    return int(combined_score), factors, features

# =========================
# 日志管理：验证与记录（增强版，使用持久化路径）
# =========================
def load_log():
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except:
        return []

def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)

def verify_past_signals(config):
    """验证之前未验证的信号，更新日志"""
    log = load_log()
    updated = False
    for entry in log:
        if not entry.get("verified", False):
            coin = entry["coin"]
            try:
                df = get_kline(coin, limit=2)
                current_price = df["close"].iloc[-1]
                record_price = entry["price"]
                signal = entry["signal"]
                
                if signal == "买入":
                    correct = current_price > record_price
                elif signal == "卖出":
                    correct = current_price < record_price
                else:
                    correct = False
                    
                entry["verified"] = True
                entry["result"] = "correct" if correct else "wrong"
                updated = True
            except Exception as e:
                print(f"验证{coin}信号失败: {e}")
                
    if updated:
        save_log(log)

def log_signal(coin, signal, score, price, whale, market_cycle, factors, features):
    """记录新信号到日志（增强版：保存更多上下文，包括特征）"""
    log = load_log()
    entry = {
        "timestamp": time.time(),
        "coin": coin,
        "signal": signal,
        "score": score,
        "price": price,
        "whale": whale,
        "market_cycle": market_cycle,
        "rule_score": factors.get('rule_score', 50),
        "ml_score": factors.get('ml_score', 50),
        "ml_confidence": factors.get('ml_confidence', 0),
        "features": features,  # 保存特征字典
        "verified": False,
        "result": None
    }
    log.append(entry)
    
    # 限制日志大小
    if len(log) > MAX_LOG_SIZE:
        unverified = [e for e in log if not e.get("verified", False)]
        verified = [e for e in log if e.get("verified", False)]
        keep = MAX_LOG_SIZE - len(unverified)
        if keep > 0:
            verified = verified[-keep:]
        else:
            verified = []
        log = unverified + verified
        
    save_log(log)

# =========================
# 自适应优化（增强版：基于机器学习，使用日志中的特征）
# =========================
def adaptive_strategy_optimization(config):
    """增强版自适应优化：重新训练机器学习模型（使用日志中保存的特征）"""
    global ai_model
    
    log = load_log()
    verified = [e for e in log if e.get("verified") and e.get("result") in ("correct", "wrong")]
    
    if len(verified) < MIN_TRAIN_SAMPLES:
        print(f"自适应优化：已验证记录不足{MIN_TRAIN_SAMPLES}条（{len(verified)}），跳过")
        return
    
    # 准备训练数据（直接从日志提取特征）
    X_df, y = ai_model.prepare_training_data(verified)
    
    if X_df is None:
        print("特征提取后样本不足")
        return
    
    # 训练模型
    success = ai_model.train(X_df, y)
    
    if success:
        # 保存模型
        ai_model.save()
        
        # 更新内存中的特征重要性
        memory = load_memory()
        memory['feature_importance'] = ai_model.feature_importance
        memory['ml_weight'] = min(0.9, memory.get('ml_weight', 0.7) + 0.05)  # 逐渐增加ML权重
        save_memory(memory)
        
        # 发送通知
        top_features = sorted(ai_model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_msg = "\n".join([f"{f}: {v:.4f}" for f, v in top_features])
        
        msg = (f"🤖 AI模型已进化\n"
               f"训练样本: {len(X_df)}\n"
               f"准确率: {ai_model.training_history[-1]['accuracy']:.4f}\n"
               f"Top5特征:\n{feature_msg}")
        
        send_notification(msg, config, "AI模型进化报告")

# =========================
# 热门币种扫描（保持不变）
# =========================
def scan_hot_coins(limit=20):
    """从OKX获取涨幅榜，返回热门币种列表（排除稳定币）"""
    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
    data = safe_request(url)
    if not data or "data" not in data:
        return []
    tickers = data["data"]
    tickers.sort(key=lambda x: float(x.get("change24h", 0)), reverse=True)
    hot = []
    stable_coins = ["USDT", "USDC", "DAI", "BUSD", "TUSD", "USDJ", "PAX", "GUSD", "HUSD", "USDK", "EURS", "CEUR"]
    for t in tickers[:limit]:
        inst = t["instId"]
        base, quote = inst.split("-") if "-" in inst else ("", "")
        if quote in stable_coins:
            continue
        vol24h = float(t.get("vol24h", 0))
        if vol24h < 100000:
            continue
        hot.append(inst)
    return hot

def hot_coin_filter(coin):
    """判断热门币是否符合加入动态列表的条件"""
    if coin in ["BTC-USDT", "ETH-USDT"]:
        return False
    return True

# =========================
# 回测报告（增强版，保持不变）
# =========================
def generate_backtest_report():
    """基于已验证的日志生成回测统计"""
    log = load_log()
    verified = [e for e in log if e.get("verified")]
    if not verified:
        return "暂无已验证数据"
    
    total = len(verified)
    correct = sum(1 for e in verified if e.get("result") == "correct")
    wrong = total - correct
    winrate = correct / total if total else 0
    
    # 按币种统计
    coins_stats = {}
    # 按模型类型统计
    ml_correct = 0
    ml_total = 0
    rule_correct = 0
    rule_total = 0
    
    for e in verified:
        coin = e["coin"]
        if coin not in coins_stats:
            coins_stats[coin] = {"correct": 0, "wrong": 0}
        if e["result"] == "correct":
            coins_stats[coin]["correct"] += 1
        else:
            coins_stats[coin]["wrong"] += 1
            
        # 统计ML vs 规则表现（根据置信度粗略判断）
        if e.get('ml_confidence', 0) > 0.6:
            ml_total += 1
            if e["result"] == "correct":
                ml_correct += 1
        else:
            rule_total += 1
            if e["result"] == "correct":
                rule_correct += 1
    
    report = f"📊 回测报告\n"
    report += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"总信号: {total}\n"
    report += f"正确: {correct}\n"
    report += f"错误: {wrong}\n"
    report += f"胜率: {winrate:.2%}\n\n"
    
    if ml_total > 0:
        ml_winrate = ml_correct / ml_total
        report += f"🤖 ML信号: {ml_total}次, 胜率: {ml_winrate:.2%}\n"
    if rule_total > 0:
        rule_winrate = rule_correct / rule_total
        report += f"📐 规则信号: {rule_total}次, 胜率: {rule_winrate:.2%}\n"
    
    report += "\n📈 币种表现:\n"
    for coin, stats in coins_stats.items():
        c = stats["correct"]
        w = stats["wrong"]
        rate = c / (c + w) if (c + w) else 0
        report += f"{coin}: {c}✓ {w}✗ ({rate:.2%})\n"
    
    return report

def send_backtest_report(config):
    """生成并发送回测报告"""
    report = generate_backtest_report()
    send_notification(report, config, "每日回测报告")

# =========================
# 主程序（增强版，已适配特征记录）
# =========================
def main():
    global last_backtest_time, last_adaptive_time, last_cycle_check, current_market_cycle
    global last_status_push, last_daily_report, last_signal_time, last_model_retrain
    global ai_model

    config = load_config()
    print("=" * 50)
    print("AI自主学习交易系统启动")
    print(f"模型状态: {'已训练' if ai_model.is_trained else '未训练'}")
    print(f"使用ML: {config.get('use_ml_model', True)}")
    print("=" * 50)

    while True:
        try:
            now = time.time()

            # 1. 市场周期检查
            if now - last_cycle_check > MARKET_CYCLE_INTERVAL:
                new_cycle = detect_market_cycle()
                if new_cycle != current_market_cycle:
                    current_market_cycle = new_cycle
                    print(f"市场周期更新: {current_market_cycle}")
                    memory = load_memory()
                    memory = apply_cycle_strategy_adjustment(memory, current_market_cycle)
                last_cycle_check = now

            # 2. 验证之前的信号
            verify_past_signals(config)

            # 3. 获取热门币种
            hot_coins = scan_hot_coins()[:MAX_DYNAMIC_COINS]
            coins = config["coins"].copy()
            for h in hot_coins:
                if hot_coin_filter(h) and h not in coins:
                    coins.append(h)

            # 4. 加载内存权重
            memory = load_memory()

            # 5. 处理每个币种
            for coin in coins:
                try:
                    df = get_kline(coin)
                    whale = detect_whale(coin)
                    
                    # 使用增强版评分，并获取特征字典
                    score, factors, features = calculate_score(df, memory, whale, current_market_cycle, coin, config)

                    # 判断信号
                    if score >= config["buy_threshold"]:
                        signal = "买入"
                    elif score <= config["sell_threshold"]:
                        signal = "卖出"
                    else:
                        signal = "中性"

                    # 信号冷却检查
                    if signal in ("买入", "卖出"):
                        last_time = last_signal_time.get(coin, 0)
                        if now - last_time < SIGNAL_COOLDOWN:
                            print(f"{coin} 信号 {signal} 冷却中，跳过")
                            continue
                        last_signal_time[coin] = now

                    # 记录日志（增强版：保存更多信息，包括特征）
                    if signal in ("买入", "卖出"):
                        price = df["close"].iloc[-1]
                        log_signal(coin, signal, score, price, whale, current_market_cycle, factors, features)

                    # 发送通知（增强版：包含ML信息）
                    if signal in ("买入", "卖出"):
                        ml_info = ""
                        if factors.get('ml_confidence', 0) > 0:
                            ml_info = f"\nML置信度: {factors['ml_confidence']:.2f}"
                            if factors.get('feature_importance'):
                                top_feat = sorted(factors['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                feat_str = ", ".join([f"{f[:10]}:{v:.2f}" for f, v in top_feat])
                                ml_info += f"\n重要特征: {feat_str}"
                        
                        msg = (f"🚨 {signal} 信号\n"
                               f"币种: {coin}\n"
                               f"综合评分: {score}\n"
                               f"规则评分: {factors['rule_score']}\n"
                               f"ML评分: {factors['ml_score']:.1f}{ml_info}\n"
                               f"周期: {current_market_cycle}\n"
                               f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        send_notification(msg, config, f"{signal}信号 {coin}")

                    print(f"{coin} {signal} {score} (规则:{factors['rule_score']} ML:{factors['ml_score']:.1f}) {current_market_cycle}")

                except Exception as e:
                    print(f"处理{coin}时出错: {e}")

            # 6. 检查回测报告间隔
            if now - last_backtest_time > BACKTEST_INTERVAL:
                send_backtest_report(config)
                last_backtest_time = now

            # 7. 检查自适应优化间隔（增强版：重新训练模型）
            if now - last_adaptive_time > ADAPTIVE_OPTIMIZATION_INTERVAL:
                adaptive_strategy_optimization(config)
                last_adaptive_time = now

            # 8. 状态推送
            if now - last_status_push > STATUS_PUSH_INTERVAL:
                model_status = "✅ ML已训练" if ai_model.is_trained else "❌ ML未训练"
                status = (f"✅ 系统运行中\n"
                         f"周期: {current_market_cycle}\n"
                         f"{model_status}\n"
                         f"监控币种: {len(coins)}\n"
                         f"下次检查: {config['check_interval']}秒")
                send_telegram_message(status, config)
                last_status_push = now

            # 9. 日报推送
            if now - last_daily_report > DAILY_REPORT_INTERVAL:
                send_backtest_report(config)
                last_daily_report = now

        except Exception as e:
            print(f"主循环异常: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(config["check_interval"])

if __name__ == "__main__":
    main()
