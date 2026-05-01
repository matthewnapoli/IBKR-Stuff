import threading
import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

ET = ZoneInfo("America/New_York")

ROLL_DAYS_BEFORE_FND = 5

DATA_PATH = "C:/Users/mnapo/Desktop/ALL_DATA/"

RETRY_SCHEDULE = [1,2,4,8,15,30,60]

# -------------------
# INSTRUMENT UNIVERSE
# -------------------


def build_s_universe():

    inst = []

    # # ---------------- FX ----------------

    fx_pairs = [
        ("EUR","USD"),("GBP","USD"),("AUD","USD"),("NZD","USD"),
        ("USD","CAD"),("USD","JPY"),("USD","CHF"),
        ("USD","SEK"),("USD","CNH"),("USD","MXN"),
        ("USD","ILS"),("USD","SGD"),("USD","SAR"),
        ("USD","KRW"),("USD","TRY"),("USD","PLN"),
    ]

    for base,quote in fx_pairs:

        inst.append({
            "country":quote,
            "asset_class":"FX",
            "symbol":base,
            "currency":quote,
            "secType":"CASH",
            "exchange":"IDEALPRO"
        })


    # ---------------- US EQUITIES ----------------


    us_equities = {
        "AAPL":"NASDAQ","MSFT":"NASDAQ","AMZN":"NASDAQ","NVDA":"NASDAQ","GOOGL":"NASDAQ",
        "META":"NASDAQ","BRK B":"NYSE","LLY":"NYSE","TSLA":"NASDAQ","AVGO":"NASDAQ",
        "JPM":"NYSE","V":"NYSE","MA":"NYSE","XOM":"NYSE","UNH":"NYSE",
        "HD":"NYSE","PG":"NYSE","COST":"NASDAQ","MRK":"NYSE","ABBV":"NYSE",
        "PEP":"NASDAQ","KO":"NYSE","BAC":"NYSE","CRM":"NYSE","ADBE":"NASDAQ",
        "NFLX":"NASDAQ","WMT":"NASDAQ","LIN":"NASDAQ","ACN":"NYSE","AMD":"NASDAQ",
        "DIS":"NYSE","MCD":"NYSE","DHR":"NYSE","CSCO":"NASDAQ","ABT":"NYSE",
        "TMO":"NYSE","VZ":"NYSE","INTC":"NASDAQ","NKE":"NYSE","TXN":"NASDAQ",
        "CMCSA":"NASDAQ","PFE":"NYSE","PM":"NYSE","UPS":"NYSE","RTX":"NYSE",
        "LOW":"NYSE","HON":"NASDAQ","QCOM":"NASDAQ","NEE":"NYSE","IBM":"NYSE"
    }

    for s, primary in us_equities.items():

        inst.append({
            "country": "USD",
            "asset_class": "EQUITY",
            "symbol": s,
            "currency": "USD",
            "secType": "STK",
            "exchange": primary
        })


    # # ---------------- INDEXES ----------------
    # NEEDS MORE SUBSCRIPTION
    # indexes = [
    #     ("USD","SPX","CBOE"),
    #     ("USD","NDX","NASDAQ"),
    #     ("USD","RUT","RUSSELL"),      # fixed
    #     ("EUR","ESTX50","EUREX"),  # fixed
    #     ("EUR","CAC40","MONEP"),     # CAC40
    #     ("GBP","Z","ICEEU"),       # fixed
    #     ("JPY","N225","OSE.JPN")
    # ]

    # for c,s,e in indexes:

    #     inst.append({
    #         "country":c,
    #         "asset_class":"INDEX",
    #         "symbol":s,
    #         "currency":c,
    #         "secType":"IND",
    #         "exchange":e
    #     })

    index_futures = [
        ("USD","ES","CME","FUT"),     # S&P 500
        ("USD","NQ","CME","FUT"),     # Nasdaq-100
        ("EUR","ESTX50","EUREX","FUT"), # Euro STOXX 50
        #("GBP","Z","ICEEU","FUT"),        # FTSE 100 (ICE does not support FUT) -> need money
        ("JPY","MNI","CME","FUT")     # Nikkei 225 (OSE has no FUT)
    ]

    for c,s,e,t in index_futures:

        inst.append({
            "country": c,
            "asset_class": "INDEX_FUTURE",
            "symbol": s,
            "currency": c,
            "secType": t,
            "exchange": e
        })
    # ---------------- COMMODITY FUTURES ----------------

    futures = [
        ("CL","NYMEX"),
        ("BZ","NYMEX"),
        ("NG","NYMEX"),
        ("GC","COMEX"),
        ("SI","COMEX"),
        ("PL","NYMEX"),
        ("HG","COMEX"),
        ("ZC","CBOT"),
        ("ZW","CBOT"),
        ("ZS","CBOT"),
        #("KC","NYBOT"),  <- ICE!
        #("CC","NYBOT"),  <- ICE!
        ("LE","CME"),
        ("HE","CME")
    ]

    for s,e in futures:

        inst.append({
            "country":"USD",
            "asset_class":"FUTURE",
            "symbol":s,
            "currency":"USD",
            "secType":"FUT",
            "exchange":e
        })


    # ---------------- RATES ----------------

    rates = [
        ("USD","SOFR3","CME"),
        ("USD", "YIA", "CBOT"),
        ("EUR","GBL","EUREX"),
        #("GBP","R","ICEEU"), #need money
        #("CAD","CGB","CDE"), #need money
        ("CHF","CONF","EUREX")
    ]

    for c,s,e in rates:

        inst.append({
            "country":c,
            "asset_class":"RATE",
            "symbol":s,
            "currency":c,
            "secType":"FUT",
            "exchange":e
        })


    return inst