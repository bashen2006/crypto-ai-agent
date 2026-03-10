Python
import requests
import time

coins = ["bitcoin","ethereum","solana","okb"]

def get_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    r = requests.get(url)
    data = r.json()
    return data[coin]["usd"]

while True:
    print("------AI Crypto Monitor------")

    for coin in coins:
        try:
            price = get_price(coin)
            print(f"{coin} price: ${price}")
        except:
            print(f"{coin} error")

    print("next check in 300 seconds")
    time.sleep(300)
