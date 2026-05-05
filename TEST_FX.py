from __future__ import annotations

from collections import deque

import pandas as pd

from datamodel import Order, TradingState


DXY_WEIGHTS = {
    "EUR-USD": -0.576,
    "USD-JPY": 0.136,
    "GBP-USD": -0.119,
    "USD-CAD": 0.091,
    "USD-SEK": 0.042,
    "USD-CHF": 0.036,
}

DXY_SCALE = 50.14348112
TRADE_SIZE = 100_000.0
LOOKBACK_SECONDS = 180
MIN_RETURN_DIFF = 0.0005


class Trader:
    def __init__(self):
        self.history = deque()

    def run(self, state: TradingState):
        mids = {
            product: self.mid_price(depth)
            for product, depth in state.order_depths.items()
            if self.mid_price(depth) is not None
        }

        dxy = self.dxy_value(mids)
        if dxy is None:
            self.history.append((state.timestamp, mids, None))
            return []

        self.history.append((state.timestamp, mids.copy(), dxy))
        self.trim_history(state.timestamp)

        baseline_mids, baseline_dxy = self.lookback_baseline()
        if baseline_dxy is None:
            return []

        dxy_return = dxy / baseline_dxy - 1.0
        orders = []

        for product in DXY_WEIGHTS.keys():
            if product not in mids:
                continue

            mid = mids[product]
            previous_mid = baseline_mids.get(product)
            if previous_mid is None or previous_mid <= 0:
                continue

            product_return = mid / previous_mid - 1.0
            diff = product_return - dxy_return

            if diff < -MIN_RETURN_DIFF:
                orders.append(Order(product=product, quantity=TRADE_SIZE*1000, price=9999999))
            elif diff > MIN_RETURN_DIFF:
                orders.append(Order(product=product, quantity=-TRADE_SIZE*1000, price=-999999))

        print(
            {
                "timestamp": state.timestamp,
                "dxy": dxy,
                "dxy_return": dxy_return,
                "orders": [
                    {"product": order.product, "quantity": order.quantity, "price": order.price}
                    for order in orders
                ],
            }
        )

        return orders

    def trim_history(self, timestamp):
        cutoff = timestamp - pd.Timedelta(seconds=LOOKBACK_SECONDS)
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def lookback_baseline(self):
        if not self.history:
            return {}, None

        oldest_timestamp, oldest_mids, oldest_dxy = self.history[0]
        newest_timestamp = self.history[-1][0]
        elapsed_seconds = (newest_timestamp - oldest_timestamp).total_seconds()
        if elapsed_seconds < LOOKBACK_SECONDS:
            return {}, None

        return oldest_mids, oldest_dxy

    @staticmethod
    def mid_price(depth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    @staticmethod
    def dxy_value(mids):
        if any(product not in mids or mids[product] <= 0 for product in DXY_WEIGHTS):
            return None

        value = DXY_SCALE
        for product, exponent in DXY_WEIGHTS.items():
            value *= mids[product] ** exponent
        return value
