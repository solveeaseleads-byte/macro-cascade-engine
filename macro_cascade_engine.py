import os
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ==========================================
# SETUP PERSISTENT LOGGING FOR FORWARD TEST
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("forward_test.log"),
        logging.StreamHandler()
    ]
)

METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID", "")

# ==========================================
# 1. 4H RESAMPLED DATA & INDICATOR PIPELINE
# ==========================================
def fetch_and_prepare_4h_data(asset_ticker="EURUSD=X", macro_ticker="^TNX", period="60d"):
    try:
        df_price = yf.download(asset_ticker, period=period, interval="1h", progress=False, auto_adjust=True)
        df_macro = yf.download(macro_ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        logging.error(f"Download failed for {asset_ticker}: {e}")
        return None

    if df_price.empty or df_macro.empty:
        return None

    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    if isinstance(df_macro.columns, pd.MultiIndex):
        df_macro.columns = df_macro.columns.get_level_values(0)

    df_price = df_price[["Open", "High", "Low", "Close"]].copy()
    df_price.columns = ["open", "high", "low", "close"]

    if df_price.index.tz is not None:
        df_price.index = df_price.index.tz_localize(None)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)

    # Resample 1H to 4H candles
    df_4h = df_price.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    df_macro = df_macro[["Close"]].rename(columns={"Close": "macro"})
    df = df_4h.join(df_macro, how="left").ffill().dropna()

    if len(df) < 30:
        return None

    # Indicators
    df["price_range_activity_proxy"] = (df["high"] - df["low"]) * (1 / df["close"]) * 1_000_000
    activity_mean = df["price_range_activity_proxy"].rolling(window=20).mean()
    df["relative_activity_rvol"] = df["price_range_activity_proxy"] / activity_mean.replace(0, 1)
    
    df["macro_diff"] = df["macro"].diff()
    df["price_diff"] = df["close"].diff()
    df["price_percentile_vah_proxy"] = df["close"].rolling(window=20).quantile(0.65)
    
    tr1 = df["high"] - df["low"]
    tr2 = np.abs(df["high"] - df["close"].shift(1))
    tr3 = np.abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    
    df["ema"] = df["close"].ewm(span=20).mean()
    df["trend_filter"] = df["ema"] > df["ema"].shift(3)
    
    # Cascade entry conditions
    conditions = (
        (df["macro_diff"] >= 0) &   
        (df["price_diff"] > 0) &       
        (df["relative_activity_rvol"] > 1.05) &        
        (df["close"] > df["price_percentile_vah_proxy"]) & 
        (df["trend_filter"] == True)
    )
    
    df["signal"] = 0
    df.loc[conditions, "signal"] = 1  
    return df

# ==========================================
# 2. RISK & CIRCUIT BREAKER EXECUTION ENGINE
# ==========================================
class ProductionExecutionEngine:
    def __init__(self, starting_balance=10000.0, risk_per_trade=0.01):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.risk_per_trade = risk_per_trade
        self.active_assets = set()
        self.consecutive_losses = 0
        self.max_drawdown_threshold = starting_balance * 0.75
        self.trading_halted = False

    def check_circuit_breakers(self):
        if self.trading_halted:
            return False
        if self.balance <= self.max_drawdown_threshold:
            self.trading_halted = True
            logging.critical(f"🛑 [CIRCUIT BREAKER] 25% Account Drawdown limit reached! Trading halted.")
            return False
        if self.consecutive_losses >= 3:
            self.trading_halted = True
            logging.critical(f"🛑 [CIRCUIT BREAKER] 3 consecutive losses reached. Trading halted.")
            return False
        return True

    def execute_trade(self, asset_name, entry_price, atr):
        if not self.check_circuit_breakers():
            logging.warning(f"[BLOCKED] Trade signal for {asset_name} ignored due to safety limits.")
            return

        if asset_name in self.active_assets:
            return  

        stop_loss = entry_price - (atr * 1.5)
        take_profit = entry_price + (atr * 4.5)
        risk_distance = entry_price - stop_loss
        
        if risk_distance <= 0:
            return

        dollar_risk = self.balance * self.risk_per_trade
        position_units = dollar_risk / risk_distance
        contract_size = 100_000 if "EUR" in asset_name else 100
        lot_size = round(max(0.01, min((position_units / contract_size), 20.0)), 2)

        print(f"\n🚨 [SIGNAL TRIGGERED] Asset: {asset_name} | Entry: {entry_price:,.4f} | SL: {stop_loss:,.4f} | TP: {take_profit:,.4f} | Lots: {lot_size}")

        if METAAPI_TOKEN and METAAPI_ACCOUNT_ID:
            self.dispatch_to_metaapi(asset_name, lot_size, stop_loss, take_profit)
        else:
            print(f"[PAPER MODE] Trade simulated successfully for {asset_name}.")
            
        self.active_assets.add(asset_name)

    def dispatch_to_metaapi(self, asset_name, lot_size, stop_loss, take_profit):
        symbol_map = {"EUR/USD": "EURUSD", "Gold": "XAUUSD"}
        mt5_symbol = symbol_map.get(asset_name, "EURUSD")
        
        base_url = f"https://mt-client-api-v1.agiliumtrade.ai/users/current/accounts/{METAAPI_ACCOUNT_ID}/trade"
        headers = {"auth-token": METAAPI_TOKEN, "Content-Type": "application/json"}
        
        payload = {
            "actionType": "ORDER_TYPE_BUY",
            "symbol": mt5_symbol,
            "volume": float(lot_size),
            "stopLoss": float(round(stop_loss, 5)),
            "takeProfit": float(round(take_profit, 5)),
            "comment": "Macro-Cascade 4H Forward Test"
        }
        try:
            res = requests.post(f"{base_url}/orders", json=payload, headers=headers, timeout=10)
            if res.status_code in [200, 201]:
                print(f"[MT5 EXECUTED] Order successfully placed on Exness demo for {mt5_symbol}.")
            else:
                print(f"[MT5 REJECTED] {res.text}")
        except Exception as e:
            print(f"[API CONNECTION ERROR]: {e}")

# ==========================================
# 3. CONTINUOUS POLLING WORKER LOOP
# ==========================================
MARKET_BASKET = [
    {"asset": "EURUSD=X", "macro": "^TNX", "name": "EUR/USD"},
    {"asset": "GC=F", "macro": "DX-Y.NYB", "name": "Gold"}
]

if __name__ == "__main__":
    print("=== MACRO-CASCADE 4H CLOUD WORKER INITIALIZED ===")
    engine = ProductionExecutionEngine(starting_balance=10000.0, risk_per_trade=0.01)

    try:
        print("\n[DIAGNOSTIC] Running initial market scan...")
        for item in MARKET_BASKET:
            df = fetch_and_prepare_4h_data(item["asset"], item["macro"])
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                print(f"-> Asset: {item['name']} | Close: {latest['close']:.4f} | RVOL: {latest['relative_activity_rvol']:.2f} | ATR: {latest['atr']:.5f} | Signal: {int(latest['signal'])}")
                if latest["signal"] == 1:
                    engine.execute_trade(item["name"], latest["close"], latest["atr"])
                else:
                    print(f"   Status: No trade signal triggered on current 4H bar.")
            else:
                print(f"-> Asset: {item['name']} | Data fetch returned empty.")

        print("\n[INFO] Diagnostic check complete. Entering 24/7 background polling loop (checking every 1 hour)...")
        
        while True:
            time.sleep(3600)
            for item in MARKET_BASKET:
                df = fetch_and_prepare_4h_data(item["asset"], item["macro"])
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    if latest["signal"] == 1:
                        engine.execute_trade(item["name"], latest["close"], latest["atr"])
            
    except KeyboardInterrupt:
        print("\nWorker gracefully shut down.")
