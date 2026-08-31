"""
Fetches the public datasets used across this repository into ./data/.

Run this once before executing any of the notebooks locally:
    python download_data.py

Data is not committed to the repository (see .gitignore) since the
equities file alone is ~50MB; each notebook documents its exact source
in its own README / opening markdown cell.
"""
import os
import urllib.request

SOURCES = {
    "data/nyse_prices.csv":
        "https://raw.githubusercontent.com/kyi3081/stock-analysis/master/prices-split-adjusted.csv",
    "data/nyse_securities.csv":
        "https://raw.githubusercontent.com/kyi3081/stock-analysis/master/securities.csv",
    "data/btc_price.csv":
        "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv",
    "data/eth_price.csv":
        "https://raw.githubusercontent.com/blockchain-unica/ethereum-ponzi/master/price-eth-usd.csv",
    "data/fx_daily.csv":
        "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv",
    # Broader crypto universe used by the pairs-trading pair scan
    # (02_crypto_stat_arb) to find a genuinely cointegrated pair, rather
    # than assuming BTC/ETH is one.
    "data/bitcoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/bitcoin.csv",
    "data/litecoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/litecoin.csv",
    "data/ethereum.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/ethereum.csv",
    "data/xrp.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/xrp.csv",
    "data/dogecoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/dogecoin.csv",
    "data/chainlink.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/chainlink.csv",
    "data/cardano.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/cardano.csv",
}

def main():
    os.makedirs("data", exist_ok=True)
    for dest, url in SOURCES.items():
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("Done. You can now run the notebooks in each project folder.")

if __name__ == "__main__":
    main()
