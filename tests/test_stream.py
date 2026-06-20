"""tests/test_stream.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from stream import TransactionProducer, ATOConsumer


def test_producer_handles_no_backend():
    """Producer should not crash when Kafka/Redis are unavailable."""
    p = TransactionProducer(bootstrap_servers="localhost:9999")
    result = p.send_transaction({"transaction_id": "TXN001", "account_number": "BOB1"})
    # May be True (if backend available) or False (graceful failure)
    assert isinstance(result, bool)


def test_consumer_creates_without_crash():
    """Consumer should instantiate without connecting."""
    consumer = ATOConsumer(bootstrap_servers="localhost:9999")
    assert consumer is not None
