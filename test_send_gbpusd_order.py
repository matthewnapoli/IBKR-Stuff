import argparse
import threading
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper


HOST = "127.0.0.1"
PAPER_PORT = 7497
CLIENT_ID = 22
FX_MIN_ORDER_GBP = 25_000
FX_ORDER_INCREMENT_GBP = 1_000


def gbpusd_contract() -> Contract:
    contract = Contract()
    contract.symbol = "GBP"
    contract.secType = "CASH"
    contract.exchange = "IDEALPRO"
    contract.currency = "USD"
    return contract


def market_order(action: str, quantity: int) -> Order:
    order = Order()
    order.action = action
    order.orderType = "MKT"
    order.totalQuantity = quantity
    order.tif = "DAY"
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    return order


def normalize_quantity(quantity: int) -> int:
    rounded = round(quantity / FX_ORDER_INCREMENT_GBP) * FX_ORDER_INCREMENT_GBP
    if rounded < FX_MIN_ORDER_GBP:
        raise ValueError(f"quantity must round to at least {FX_MIN_ORDER_GBP} GBP")
    return int(rounded)


class TestOrderApp(EWrapper, EClient):
    def __init__(self, quantity: int) -> None:
        EClient.__init__(self, self)
        self.quantity = quantity
        self.next_order_id: int | None = None

    def nextValidId(self, orderId: int) -> None:
        self.next_order_id = orderId
        self.send_order("BUY")
        self.send_order("SELL")

    def send_order(self, action: str) -> None:
        if self.next_order_id is None:
            return
        order_id = self.next_order_id
        self.next_order_id += 1
        print({"event": "placing_order", "order_id": order_id, "action": action, "quantity_gbp": self.quantity})
        self.placeOrder(order_id, gbpusd_contract(), market_order(action, self.quantity))

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice,
    ) -> None:
        print({
            "event": "order_status",
            "order_id": orderId,
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
        })

    def openOrder(self, orderId, contract, order, orderState) -> None:
        print({
            "event": "open_order",
            "order_id": orderId,
            "symbol": contract.symbol,
            "currency": contract.currency,
            "action": order.action,
            "quantity": order.totalQuantity,
            "status": orderState.status,
        })

    def execDetails(self, reqId, contract, execution) -> None:
        print({
            "event": "execution",
            "order_id": execution.orderId,
            "side": execution.side,
            "shares": execution.shares,
            "price": execution.price,
            "time": execution.time,
        })

    def error(self, reqId, errorTime, errorCode, errorMsg, advancedOrderRejectJson="") -> None:
        print({"event": "error", "req_id": reqId, "code": errorCode, "message": errorMsg})


def main() -> None:
    parser = argparse.ArgumentParser(description="Send paper GBP.USD IDEALPRO BUY then SELL test market orders.")
    parser.add_argument("--quantity", type=int, default=25_000, help="GBP quantity; rounded to 1000 GBP.")
    parser.add_argument("--seconds", type=int, default=20, help="How long to keep the connection open.")
    args = parser.parse_args()

    quantity = normalize_quantity(args.quantity)
    app = TestOrderApp(quantity)
    app.connect(HOST, PAPER_PORT, clientId=CLIENT_ID)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    try:
        time.sleep(args.seconds)
    finally:
        app.disconnect()


if __name__ == "__main__":
    main()
