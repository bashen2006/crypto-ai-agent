import json
import os

def init_config_files():
    """初始化所有配置文件"""
    
    # 初始化 config.json
    if not os.path.exists('config.json'):
        config = {
            "coins": ["BTC-USDT", "ETH-USDT", "OKB-USDT"],
            "buy_threshold": 70,
            "sell_threshold": 35,
            "check_interval": 300,
            "telegram_bot_token": "8546952065:AAFx7E4-wx4AeGWS5sQthgnFLhtUvExOkrQ",
            "telegram_chat_id": "1392813886",
            "email_user": "13781411151@139.com",
            "email_pass": "98adfa5c93a3a509df00",
            "email_receiver": "13781411151@139.com",
            "use_ml_model": True,
            "ml_weight": 0.7,
            "min_train_samples": 50,
            "max_features": 50,
            "model_retrain_interval": 21600,
            "enable_deep_learning": False
        }
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("config.json 已创建")
    
    # 初始化 ai_memory.json
    if not os.path.exists('ai_memory.json'):
        memory = {
            "trend_weight": 0.3,
            "volume_weight": 0.2,
            "momentum_weight": 0.2,
            "volatility_weight": 0.1,
            "sentiment_weight": 0.2,
            "ml_weight": 0.7,
            "last_training_time": 0,
            "total_training_sessions": 0,
            "best_accuracy": 0.0,
            "feature_importance": {},
            "model_version": "1.0.0",
            "training_history": []
        }
        with open('ai_memory.json', 'w') as f:
            json.dump(memory, f, indent=2)
        print("ai_memory.json 已创建")
    
    # 初始化 prediction_log.json
    if not os.path.exists('prediction_log.json'):
        with open('prediction_log.json', 'w') as f:
            json.dump([], f)
        print("prediction_log.json 已创建")
    
    # 初始化 feature_config.json
    if not os.path.exists('feature_config.json'):
        feature_config = {
            "feature_names": [],
            "training_history": [],
            "feature_importance": {},
            "model_metrics": {}
        }
        with open('feature_config.json', 'w') as f:
            json.dump(feature_config, f, indent=2)
        print("feature_config.json 已创建")
    
    print("\n所有配置文件初始化完成！")
    print("请检查并修改 config.json 中的API密钥和邮箱密码。")

if __name__ == "__main__":
    init_config_files()
