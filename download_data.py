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
}

def main():
    os.makedirs("data", exist_ok=True)
    for dest, url in SOURCES.items():
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("Done. You can now run the notebooks in each project folder.")

if __name__ == "__main__":
    main()
