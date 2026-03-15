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
        "buy_threshold": 70,        # 动态买入阈值
        "sell_threshold": 35,        # 动态卖出阈值
        "feature_importance": {},
        "last_training_time": 0,
        "total_training_sessions": 0,
        "best_accuracy": 0.0,
        "model_version": "1.0.0",
        "training_history": []
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
# 通知功能：Telegram和邮件
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
    """通过139邮箱发送邮件，仅使用官方端口25和465"""
    user = config.get("email_user")
    pwd = config.get("email_pass")
    receiver = config.get("email_receiver")
    if not user or not pwd or not receiver:
        print("邮件配置不完整，跳过发送")
        return

    # 使用官方端口：25（STARTTLS）和465（SSL）
    smtp_servers = [
        ("smtp.139.com", 25, True),   # 端口25 + STARTTLS
        ("smtp.139.com", 465, False)  # 端口465 + SSL
    ]

    for host, port, use_tls in smtp_servers:
        try:
            if use_tls:
                server = smtplib.SMTP(host, port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(host, port, timeout=10)

            server.login(user, pwd)
            msg = MIMEMultipart()
            msg["From"] = user
            msg["To"] = receiver
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            server.send_message(msg)
            server.quit()
            print(f"邮件发送成功（使用 {host}:{port}）")
            return
        except smtplib.SMTPAuthenticationError:
            print(f"认证失败（{host}:{port}），请检查邮箱授权码是否正确")
        except (smtplib.SMTPServerDisconnected, ConnectionRefusedError, TimeoutError) as e:
            print(f"连接失败（{host}:{port}）: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"未知错误（{host}:{port}）: {type(e).__name__}: {e}")

    print("所有邮件服务器尝试均失败，请检查网络或邮箱配置")

def send_notification(content, config, subject=None):
    """同时发送Telegram和邮件（若配置）"""
    send_telegram_message(content, config)
    if subject:
        send_email(subject, content, config)

# =========================
# 安全请求
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
# 市场周期识别
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
    """
    根据市场周期调整权重和阈值，并保存
    牛市：增强趋势，提高买入阈值、降低卖出阈值（顺势而为）
    熊市：降低买入阈值、降低卖出阈值（谨慎抄底，快速止损）
    震荡：放宽阈值，增加交易频率
    """
    memory = memory.copy()
    if cycle == "牛市":
        memory["trend_weight"] = min(memory.get("trend_weight", 0.3) + 0.05, 1.0)
        # 提高买入门槛，降低卖出门槛，避免回调下车
        memory["buy_threshold"] = min(memory.get("buy_threshold", 70) + 2, 85)
        memory["sell_threshold"] = max(memory.get("sell_threshold", 35) - 2, 20)
    elif cycle == "熊市":
        memory["trend_weight"] = max(memory.get("trend_weight", 0.3) - 0.05, 0.0)
        # 降低买入门槛（博反弹），降低卖出门槛（及时止损）
        memory["buy_threshold"] = max(memory.get("buy_threshold", 70) - 2, 55)
        memory["sell_threshold"] = max(memory.get("sell_threshold", 35) - 2, 20)
    elif cycle == "震荡":
        memory["volume_weight"] = min(memory.get("volume_weight", 0.2) + 0.05, 1.0)
        # 放宽阈值，增加交易频率
        memory["buy_threshold"] = max(memory.get("buy_threshold", 70) - 3, 55)
        memory["sell_threshold"] = min(memory.get("sell_threshold", 35) + 3, 45)

    # 确保买入阈值 > 卖出阈值，且在一定范围内
    memory["buy_threshold"] = max(memory["buy_threshold"], memory["sell_threshold"] + 5)
    memory["buy_threshold"] = min(memory["buy_threshold"], 85)
    memory["sell_threshold"] = max(memory["sell_threshold"], 15)
    memory["sell_threshold"] = min(memory["sell_threshold"], 50)

    save_memory(memory)
    print(f"周期调整: {cycle}, 新阈值: 买入={memory['buy_threshold']}, 卖出={memory['sell_threshold']}")
    return memory

# =========================
# AI评分（增强版：结合规则和机器学习）
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
# 日志管理：验证与记录
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
    """记录新信号到日志（保存特征）"""
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
        "features": features,
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
# 自适应优化（增强版：基于机器学习，同时微调阈值）
# =========================
def adaptive_strategy_optimization(config):
    """增强版自适应优化：重新训练机器学习模型，并根据胜率微调阈值"""
    global ai_model

    log = load_log()
    verified = [e for e in log if e.get("verified") and e.get("result") in ("correct", "wrong")]

    if len(verified) < MIN_TRAIN_SAMPLES:
        print(f"自适应优化：已验证记录不足{MIN_TRAIN_SAMPLES}条（{len(verified)}），跳过")
        return

    # 准备训练数据
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
        memory['ml_weight'] = min(0.9, memory.get('ml_weight', 0.7) + 0.05)

        # 根据模型胜率微调阈值
        accuracy = ai_model.training_history[-1]['accuracy']
        if accuracy > 0.65:
            # 胜率高，可以收紧阈值，追求更高胜率
            memory['buy_threshold'] = min(memory.get('buy_threshold', 70) + 2, 85)
            memory['sell_threshold'] = max(memory.get('sell_threshold', 35) - 2, 20)
        elif accuracy < 0.5:
            # 胜率低，放宽阈值，增加交易频率以积累更多样本
            memory['buy_threshold'] = max(memory.get('buy_threshold', 70) - 2, 55)
            memory['sell_threshold'] = min(memory.get('sell_threshold', 35) + 2, 45)

        # 确保买入 > 卖出
        memory['buy_threshold'] = max(memory['buy_threshold'], memory['sell_threshold'] + 5)
        memory['buy_threshold'] = min(memory['buy_threshold'], 85)
        memory['sell_threshold'] = max(memory['sell_threshold'], 15)
        memory['sell_threshold'] = min(memory['sell_threshold'], 50)

        save_memory(memory)

        # 发送通知
        top_features = sorted(ai_model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_msg = "\n".join([f"{f}: {v:.4f}" for f, v in top_features])

        msg = (f"🤖 AI模型已进化\n"
               f"训练样本: {len(X_df)}\n"
               f"准确率: {accuracy:.4f}\n"
               f"新阈值: 买入={memory['buy_threshold']}, 卖出={memory['sell_threshold']}\n"
               f"Top5特征:\n{feature_msg}")

        send_notification(msg, config, "AI模型进化报告")

# =========================
# 热门币种扫描
# =========================
def scan_hot_coins(limit=20):
    """从OKX获取涨幅榜，返回热门币种列表（只包含以USDT计价的交易对）"""
    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
    data = safe_request(url)
    if not data or "data" not in data:
        return []
    tickers = data["data"]
    tickers.sort(key=lambda x: float(x.get("change24h", 0)), reverse=True)
    hot = []
    for t in tickers[:limit]:
        inst = t["instId"]
        if "-" not in inst:
            continue
        base, quote = inst.split("-")
        if quote != "USDT":          # 只保留USDT交易对
            continue
        vol24h = float(t.get("vol24h", 0))
        if vol24h < 100000:          # 成交量过滤（可选）
            continue
        change24h = float(t.get("change24h", 0))
        hot.append((inst, change24h))
    return hot

def hot_coin_filter(coin_name):
    if coin_name in ["BTC-USDT", "ETH-USDT"]:
        return False
    return True

# =========================
# 回测报告
# =========================
def generate_backtest_report():
    log = load_log()
    verified = [e for e in log if e.get("verified")]
    if not verified:
        return "暂无已验证数据"

    total = len(verified)
    correct = sum(1 for e in verified if e.get("result") == "correct")
    wrong = total - correct
    winrate = correct / total if total else 0

    coins_stats = {}
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

def get_recent_signals(n=3):
    log = load_log()
    signals = [e for e in log if e.get("signal") in ("买入", "卖出")]
    signals.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    recent = []
    now = time.time()
    for s in signals[:n]:
        ts = s.get("timestamp", 0)
        minutes_ago = int((now - ts) / 60) if ts else 0
        time_str = f"{minutes_ago}分钟前" if minutes_ago < 60 else f"{minutes_ago//60}小时前"
        recent.append(f"{time_str} {s['coin']} {s['signal']} ({s['score']})")
    return recent

def get_signal_stats_since(hours):
    log = load_log()
    cutoff = time.time() - hours * 3600
    recent = [e for e in log if e.get("timestamp", 0) > cutoff and e.get("signal") in ("买入", "卖出")]
    total = len(recent)
    correct = sum(1 for e in recent if e.get("result") == "correct")
    return total, correct

def send_backtest_report(config):
    report = generate_backtest_report()
    send_notification(report, config, "每日回测报告")

# =========================
# 主程序
# =========================
def main():
    global last_backtest_time, last_adaptive_time, last_cycle_check, current_market_cycle
    global last_status_push, last_daily_report, last_signal_time, last_model_retrain
    global ai_model

    config = load_config()
    print("=" * 50)
    start_time = time.time()
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
            hot_coins_with_change = scan_hot_coins()[:MAX_DYNAMIC_COINS]
            hot_coins = [coin for coin, _ in hot_coins_with_change]
            coins = config["coins"].copy()
            for h in hot_coins:
                if hot_coin_filter(h) and h not in coins:
                    coins.append(h)

            # 4. 加载内存权重，并用内存中的阈值覆盖配置（实现动态阈值）
            memory = load_memory()
            config["buy_threshold"] = memory.get("buy_threshold", config["buy_threshold"])
            config["sell_threshold"] = memory.get("sell_threshold", config["sell_threshold"])

            # 5. 处理每个币种
            for coin in coins:
                try:
                    df = get_kline(coin)
                    whale = detect_whale(coin)

                    score, factors, features = calculate_score(df, memory, whale, current_market_cycle, coin, config)

                    if score >= config["buy_threshold"]:
                        signal = "买入"
                    elif score <= config["sell_threshold"]:
                        signal = "卖出"
                    else:
                        signal = "中性"

                    if signal in ("买入", "卖出"):
                        last_time = last_signal_time.get(coin, 0)
                        if now - last_time < SIGNAL_COOLDOWN:
                            print(f"{coin} 信号 {signal} 冷却中，跳过")
                            continue
                        last_signal_time[coin] = now

                    if signal in ("买入", "卖出"):
                        price = df["close"].iloc[-1]
                        log_signal(coin, signal, score, price, whale, current_market_cycle, factors, features)

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
                               f"当前阈值: 买{config['buy_threshold']}/卖{config['sell_threshold']}\n"
                               f"周期: {current_market_cycle}\n"
                               f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        send_notification(msg, config, f"{signal}信号 {coin}")

                    print(f"{coin} {signal} {score} (规则:{factors['rule_score']} ML:{factors['ml_score']:.1f}) {current_market_cycle}")

                except Exception as e:
                    print(f"处理{coin}时出错: {e}")

            # 6. 回测报告
            if now - last_backtest_time > BACKTEST_INTERVAL:
                send_backtest_report(config)
                last_backtest_time = now

            # 7. 自适应优化（模型训练+阈值微调）
            if now - last_adaptive_time > ADAPTIVE_OPTIMIZATION_INTERVAL:
                adaptive_strategy_optimization(config)
                last_adaptive_time = now

            # 8. 状态推送（使用动态阈值）
            if now - last_status_push > STATUS_PUSH_INTERVAL:
                uptime_seconds = int(now - start_time)
                hours = uptime_seconds // 3600
                minutes = (uptime_seconds % 3600) // 60

                if ai_model.is_trained:
                    model_acc = ai_model.training_history[-1]['accuracy']
                    model_samples = ai_model.training_history[-1]['samples']
                    model_info = f"✅ 已训练 (准确率: {model_acc:.2%}, 样本: {model_samples})"
                else:
                    model_info = "❌ 未训练"

                coin_lines = []
                for coin in coins:
                    try:
                        df = get_kline(coin)
                        price = df["close"].iloc[-1]
                        whale = detect_whale(coin)
                        score, factors, _ = calculate_score(df, memory, whale, current_market_cycle, coin, config)
                        if score >= config["buy_threshold"]:
                            signal_icon = "🟢 买入"
                        elif score <= config["sell_threshold"]:
                            signal_icon = "🔴 卖出"
                        else:
                            signal_icon = "⚪ 中性"
                        coin_lines.append(f"{coin}: ${price:.4f} {score} {signal_icon}")
                    except:
                        coin_lines.append(f"{coin}: 获取失败")

                recent_signals = get_recent_signals(3)
                recent_str = "\n".join(recent_signals) if recent_signals else "暂无"

                hot_coins_with_change = scan_hot_coins()[:3]
                hot_lines = [f"{coin}: {change:+.2f}%" for coin, change in hot_coins_with_change]
                hot_str = "\n".join(hot_lines) if hot_lines else "暂无"

                today_total, today_correct = get_signal_stats_since(24)
                today_winrate = today_correct / today_total if today_total > 0 else 0
                sixh_total, sixh_correct = get_signal_stats_since(6)
                sixh_winrate = sixh_correct / sixh_total if sixh_total > 0 else 0

                status = f"🔥 AI超级信号\n\n"
                status += "📈 当前监控币种:\n" + "\n".join(coin_lines) + "\n\n"
                status += f"⏱️ 最近信号:\n{recent_str}\n\n"
                status += f"📊 市场热点:\n{hot_str}\n\n"
                status += f"🤖 模型表现: {model_info}\n"
                status += f"📆 今日信号: {today_total}次 (胜率: {today_winrate:.2%})\n\n"
                status += f"📊 AI复盘报告:\n"
                status += f"6小时: {sixh_total}次, 胜率 {sixh_winrate:.2%}\n"
                status += f"24小时: {today_total}次, 胜率 {today_winrate:.2%}\n"
                status += f"当前阈值: 买入{config['buy_threshold']} / 卖出{config['sell_threshold']}\n"
                status += f"运行时间: {hours}小时{minutes}分钟 | 周期: {current_market_cycle}"

                # 防重复机制
                current_interval = int(now / STATUS_PUSH_INTERVAL)
                last_interval = int(last_status_push / STATUS_PUSH_INTERVAL) if last_status_push > 0 else -1
                if current_interval != last_interval:
                    send_telegram_message(status, config)
                else:
                    print("状态推送跳过（同一周期内已发送）")
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
