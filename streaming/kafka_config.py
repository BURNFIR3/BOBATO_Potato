"""streaming/kafka_config.py – Kafka topic and connection configuration"""

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

TOPICS = {
    "transactions": "ato_transactions",
    "alerts":       "ato_alerts",
    "responses":    "ato_responses",
}

CONSUMER_CONFIG = {
    "group_id":          "ato_consumer_group",
    "auto_offset_reset": "latest",
    "enable_auto_commit": True,
    "max_poll_records":  100,
    "session_timeout_ms": 10000,
}

PRODUCER_CONFIG = {
    "retries":       3,
    "acks":          "all",
    "linger_ms":     5,
    "batch_size":    16384,
}
