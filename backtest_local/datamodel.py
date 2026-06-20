from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrderDepth:
    """L1 book for one product.

    buy_orders maps bid_price -> bid_size.
    sell_orders maps ask_price -> ask_size.
    """

    buy_orders: dict[float, float] = field(default_factory=dict)
    sell_orders: dict[float, float] = field(default_factory=dict)


@dataclass
class TradingState:
    """State passed to each algorithm call."""

    timestamp: Any
    order_depths: dict[str, OrderDepth] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)


@dataclass
class Order:
    product: str
    quantity: float
    price: float | None = None


@dataclass
class Trade:
    product: str
    quantity: float
    price: float
    timestamp: Any
