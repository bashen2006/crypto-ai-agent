# AI评分引擎

def calculate_score(indicators, derivatives, whale_data):

    score = 50


    # =========================
    # 技术指标评分
    # =========================

    if indicators["ma7"] > indicators["ma30"]:

        score += 8

    if indicators["rsi"] < 35:

        score += 6

    if indicators["macd"] > indicators["macd_signal"]:

        score += 8

    if indicators["price"] < indicators["boll_lower"]:

        score += 6

    if indicators["price"] > indicators["boll_upper"]:

        score -= 6


    # =========================
    # 衍生品数据评分
    # =========================

    if derivatives["funding_rate"] is not None:

        if derivatives["funding_rate"] < 0:

            score += 5

        else:

            score -= 3


    if derivatives["open_interest"] is not None:

        if derivatives["open_interest"] > 0:

            score += 4


    if derivatives["long_short_ratio"] is not None:

        if derivatives["long_short_ratio"] < 0.5:

            score += 4

        elif derivatives["long_short_ratio"] > 1.5:

            score -= 4


    # =========================
    # 巨鲸数据评分
    # =========================

    if whale_data:

        score += 5


    # 限制范围

    score = max(0, min(100, score))

    return score


# =========================
# 根据评分生成建议
# =========================

def generate_signal(score):

    if score >= 75:

        return "强买入"

    elif score >= 60:

        return "建议买入"

    elif score <= 25:

        return "强卖出"

    elif score <= 40:

        return "建议卖出"

    else:

        return "观望"
