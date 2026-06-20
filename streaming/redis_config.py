"""streaming/redis_config.py – Redis fallback configuration"""

REDIS_HOST     = "localhost"
REDIS_PORT     = 6379
REDIS_DB       = 0
REDIS_PASSWORD = None

KEYS = {
    "transactions": "ato:transactions",
    "alerts":       "ato:alerts",
    "stats":        "ato:stats",
    "suspended":    "ato:suspended",
}

MAX_LIST_LENGTH = 10000  # Max items to retain in Redis lists
