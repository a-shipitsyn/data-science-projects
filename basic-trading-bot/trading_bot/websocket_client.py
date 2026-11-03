import asyncio
import json
import logging
from collections.abc import Callable

import websockets

log = logging.getLogger(__name__)


class CoinbaseWebsocket:
    def __init__(self, product_ids: list[str] | None = None, max_backoff: float = 60.0):
        # Coinbase public market data feed (no authentication needed for the ticker channel)
        self.url = "wss://ws-feed.exchange.coinbase.com"
        self.product_ids = product_ids or ["BTC-USD"]
        self.max_backoff = max_backoff
        self.ws = None

    async def reconnect(self) -> None:
        """Close the current connection; start() then connects again (used by the feed watchdog)."""
        if self.ws is not None:
            await self.ws.close()

    async def start(self, callback: Callable) -> None:
        """Connects, subscribes to the ticker and routes messages to callback; reconnects with backoff."""
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    # Subscription message format required by Coinbase
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": self.product_ids,
                                              "channels": ["ticker"]}))
                    log.info("Connected to Coinbase WebSocket. Subscribed to %s", self.product_ids)
                    backoff = 1.0

                    async for message in ws:
                        data = json.loads(message)
                        # We only care about ticker updates
                        if data.get("type") == "ticker":
                            await callback(data)
            except (websockets.exceptions.WebSocketException, OSError) as e:
                log.warning("WebSocket error: %s. Reconnecting in %.0f s...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
