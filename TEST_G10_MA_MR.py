from __future__ import annotations

from collections import deque

from datamodel import Order, TradingState


SHORT_WINDOW = 60
LONG_WINDOW = 300
MIN_MA_DIFF = 0.0010
TRADE_SIZE = 25_000.0

BUY_MARKETABLE_PRICE = 9999999
SELL_MARKETABLE_PRICE = -9999999


class Trader:
    def __init__(self):
        self.mid_history = {}
        self.signal_position = {}

    def run(self, state: TradingState):
        orders = []

        for product, depth in state.order_depths.items():
            mid = self.mid_price(depth)
            if mid is None:
                continue

            history = self.mid_history.setdefault(product, deque(maxlen=LONG_WINDOW))
            history.append(mid)

            if len(history) < LONG_WINDOW:
                continue

            short_ma = sum(list(history)[-SHORT_WINDOW:]) / SHORT_WINDOW
            long_ma = sum(history) / LONG_WINDOW
            ma_diff = short_ma / long_ma - 1.0

            current_signal = self.signal_position.get(product, 0)

            if ma_diff <= -MIN_MA_DIFF and current_signal != 1:
                quantity = (1 - current_signal) * TRADE_SIZE
                orders.append(Order(product=product, quantity=quantity, price=BUY_MARKETABLE_PRICE))
                self.signal_position[product] = 1

            elif ma_diff >= MIN_MA_DIFF and current_signal != -1:
                quantity = (-1 - current_signal) * TRADE_SIZE
                orders.append(Order(product=product, quantity=quantity, price=SELL_MARKETABLE_PRICE))
                self.signal_position[product] = -1

        print(
            {
                "timestamp": state.timestamp,
                "orders": [
                    {"product": order.product, "quantity": order.quantity, "price": order.price}
                    for order in orders
                ],
            }
        )

        return orders

    @staticmethod
    def mid_price(depth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
