"""Tests for broker_connector.py."""

import pytest
from src.trading.broker_connector import PaperBroker, BrokerConnector
from src.models import TradeSignal


class TestPaperBroker:
    def test_connect(self):
        broker = PaperBroker(initial_balance=10000)
        assert broker.connect()
        assert broker.is_connected()

    def test_disconnect(self):
        broker = PaperBroker()
        broker.connect()
        assert broker.disconnect()
        assert not broker.is_connected()

    def test_get_account_info(self):
        broker = PaperBroker(initial_balance=5000, leverage=50)
        broker.connect()
        info = broker.get_account_info()
        assert info.balance == 5000
        assert info.leverage == 50
        assert info.margin_level == 100.0

    def test_send_order_buy(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        assert result["ticket"] > 0
        assert len(broker.get_positions()) == 1

    def test_send_order_sell(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="SELL",
            entry_price=2450, sl=2455, tp=2440, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        assert len(broker.get_positions()) == 1

    def test_send_order_not_connected(self):
        broker = PaperBroker()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert not result["success"]
        assert "Not connected" in result["error"]

    def test_send_order_insufficient_margin(self):
        broker = PaperBroker(initial_balance=100, leverage=10)
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=10.0)
        result = broker.send_order(signal)
        assert not result["success"]
        assert "Insufficient margin" in result["error"]

    def test_close_position(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert broker.close_position(result["ticket"])
        assert len(broker.get_positions()) == 0

    def test_close_position_not_found(self):
        broker = PaperBroker()
        broker.connect()
        assert not broker.close_position(99999)

    def test_close_all_positions(self):
        broker = PaperBroker()
        broker.connect()
        for _ in range(3):
            broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
                entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        assert len(broker.get_positions()) == 3
        assert broker.close_all_positions()
        assert len(broker.get_positions()) == 0

    def test_modify_sl_tp(self):
        broker = PaperBroker()
        broker.connect()
        result = broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        ticket = result["ticket"]
        assert broker.modify_position_sl_tp(ticket, sl=2440, tp=2470)
        pos = broker.get_positions()[0]
        assert pos.sl == 2440
        assert pos.tp == 2470

    def test_partial_close(self):
        broker = PaperBroker()
        broker.connect()
        result = broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=1.0))
        ticket = result["ticket"]
        result = broker.close_position_partial(ticket, 0.5)
        assert result["success"]
        assert result["closed_volume"] == 0.5
        assert result["remaining_volume"] == 0.5

    def test_margin_calculation(self):
        broker = PaperBroker(initial_balance=10000, leverage=100)
        broker.connect()
        margin = broker._margin_required(1.0, 2450.0)
        assert margin == 2450.0

    def test_notional_calculation(self):
        broker = PaperBroker()
        notional = broker._notional(1.0, 2450.0)
        assert notional == 2450.0 * 100.0

    def test_check_sl_tp(self):
        broker = PaperBroker()
        broker.connect()
        broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        broker.update_price("XAUUSD", 2444.0, 2444.5)
        broker.check_sl_tp("XAUUSD", 2444.0, 2444.5)
        assert len(broker.get_positions()) == 0

    def test_health_check(self):
        broker = PaperBroker()
        broker.connect()
        health = broker.health_check()
        assert health["connected"]
        assert health["terminal_ok"]


class TestPaperBrokerMultiSymbolContractSize:
    """Regression tests for the multi-symbol contract-size bug found in
    audit: gold uses 100 oz/lot, standard forex pairs use 100,000 units/lot.
    Before the fix, every symbol used gold's contract size, producing
    wildly wrong P&L/margin for forex trades."""

    def test_forex_price_is_not_gold_fallback(self):
        broker = PaperBroker()
        broker.connect()
        price = broker.get_price("EURUSD")
        assert price["bid"] < 10  # a real EURUSD price, not gold's ~2450

    def test_forex_margin_uses_100k_contract_size(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="EURUSD", direction="BUY",
            entry_price=1.0850, sl=1.0800, tp=1.0900, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        account = broker.get_account_info()
        # notional = 0.1 lot * 1.0852 * 100,000 = ~10,852; margin = notional/100 = ~108.5
        expected_margin = 0.1 * 1.0852 * 100_000 / 100.0
        assert abs(account.margin - expected_margin) < 1.0

    def test_gold_still_uses_100oz_contract_size(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        broker.send_order(signal)
        account = broker.get_account_info()
        expected_margin = 0.1 * 2450.5 * 100 / 100.0
        assert abs(account.margin - expected_margin) < 1.0

    def test_forex_pl_on_close_is_realistic(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="EURUSD", direction="BUY",
            entry_price=1.0850, sl=1.0800, tp=1.0900, lot_size=1.0)
        result = broker.send_order(signal)
        ticket = result["ticket"]
        broker.update_price("EURUSD", bid=1.0950, ask=1.0952)
        broker.close_position(ticket)
        # 0.0100 move * 1.0 lot * 100,000 = ~$1000, not ~$1 (which the old
        # gold-sized 100x multiplier would have produced)
        assert broker._balance > 10000 + 500


class TestBrokerConnectorABC:
    def test_abstract_class(self):
        with pytest.raises(TypeError):
            BrokerConnector()
