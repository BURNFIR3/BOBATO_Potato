"""scripts/simulate_live_stream.py - Simulate live transaction stream against API"""

import sys
import os
import time
import requests
import random
import pandas as pd
from loguru import logger

# Try to insert src path for constants, else hardcode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from utils import DATA_DIR
    csv_path = DATA_DIR / "ato_dataset_processed.csv"
except ImportError:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "ato_dataset_processed.csv")

API_URL = "http://localhost:8000/api/v1/detect-ato"

def stream_transactions(delay_sec: float = 0.5):
    """Read processed dataset and POST to API to simulate live traffic."""
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found: {csv_path}. Run process_dataset.py first.")
        return

    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Shuffle the dataset to get a random mix of fraud/normal
    df = df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Starting live stream simulation ({len(df)} transactions)...")
    logger.info(f"Target API: {API_URL}")
    logger.info(f"Delay: {delay_sec}s per transaction")
    
    success_count = 0
    fail_count = 0

    for i, row in df.iterrows():
        # Convert row to dict, handle NaNs and floats
        payload = row.to_dict()
        for k, v in payload.items():
            if pd.isna(v):
                payload[k] = None
        
        # Inject API required fields that are missing in Base.csv
        if "transaction_id" not in payload:
            payload["transaction_id"] = f"TXN-LIVE-{random.randint(10000, 99999)}"
        if "account_number" not in payload:
            payload["account_number"] = f"BOB{random.randint(1000000, 9999999)}"
        if "transaction_amount" not in payload:
            # Map intended_balcon_amount if available, else random
            if "intended_balcon_amount" in payload and payload["intended_balcon_amount"] > 0:
                payload["transaction_amount"] = payload["intended_balcon_amount"]
            else:
                payload["transaction_amount"] = random.uniform(100, 50000)
        if "transaction_type" not in payload:
            payload["transaction_type"] = "NEFT"
        
        # Ensure device_os is a string (API expects string, dataset has floats like 2.0)
        if "device_os" in payload and not isinstance(payload["device_os"], str):
            payload["device_os"] = str(payload["device_os"])
        
        # We don't send is_ato to the detection endpoint (it's the target!)
        is_ato = payload.pop("is_ato", 0)

        try:
            resp = requests.post(API_URL, json=payload, timeout=2.0)
            if resp.status_code == 200:
                success_count += 1
                result = resp.json()
                action = result.get("action", "UNKNOWN")
                prob = result.get("ato_probability", 0)
                
                # Format output nicely
                status_symbol = "🔴" if action == "SUSPEND" else "⚠️" if action == "OTP_REQUIRED" else "✅"
                actual = "FRAUD" if is_ato == 1 else "NORMAL"
                
                print(f"[{i+1:5d}] {status_symbol} {action:<12} | "
                      f"Prob: {prob:.3f} | Actual: {actual:<6} | "
                      f"Txn: {payload.get('transaction_id')}")
            else:
                fail_count += 1
                logger.warning(f"API Error {resp.status_code}: {resp.text}")
                
        except requests.exceptions.RequestException as e:
            fail_count += 1
            logger.error(f"Connection error (Is the API running?): {e}")
            time.sleep(2) # Backoff if server is down
            
        time.sleep(delay_sec)

    logger.success(f"Stream complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate live transaction stream")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between transactions in seconds")
    args = parser.parse_args()
    
    try:
        stream_transactions(delay_sec=args.delay)
    except KeyboardInterrupt:
        logger.info("\nStream simulation stopped by user.")
