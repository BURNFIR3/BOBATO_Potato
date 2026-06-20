"""scripts/start_stream.py – Step 5: Start Kafka/Redis ATO consumer stream"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from loguru import logger
from stream import start_consumer


def on_alert(result: dict) -> None:
    print(f"\n🚨 ATO ALERT: {result.get('transaction_id')} | "
          f"Account: {result.get('account_number')} | "
          f"Action: {result.get('action')} | "
          f"Prob: {result.get('ato_probability'):.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Starting ATO Transaction Stream Consumer")
    print("Ctrl+C to stop")
    print("=" * 60)
    consumer = start_consumer(on_alert=on_alert)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        consumer.stop()
        print("\nStream consumer stopped.")
