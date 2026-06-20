"""streaming/kafka_consumer.py – Dedicated consumer entry point"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from loguru import logger
from stream import ATOConsumer


def main():
    def on_alert(result: dict) -> None:
        action = result.get("action", "UNKNOWN")
        prob   = result.get("ato_probability", 0)
        beh    = result.get("behavioral_anomaly_score", 0)
        acc    = result.get("account_number", "?")
        txn    = result.get("transaction_id", "?")
        logger.warning(
            f"[{action}] TXN={txn} | ACC={acc} | "
            f"Prob={prob:.3f} | Behavioral={beh:.3f}"
        )

    logger.info("ATO Kafka Consumer starting…")
    consumer = ATOConsumer(on_alert=on_alert)
    consumer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        consumer.stop()
        logger.info("Consumer stopped.")


if __name__ == "__main__":
    main()
