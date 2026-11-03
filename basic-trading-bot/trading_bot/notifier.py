import json
import logging
import os
import threading
import urllib.request

log = logging.getLogger(__name__)


class Notifier:
    """Sends important events (trades, kill switch, silent feed) to Telegram.

    Enabled only when the environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set;
    otherwise messages are only logged. Sending runs in a background thread, so a slow or failing
    network never blocks the price feed.
    """

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> None:
        log.info("NOTIFY: %s", text)
        if self.enabled:
            threading.Thread(target=self._post, args=(text,), daemon=True).start()

    def _post(self, text: str) -> None:
        body = json.dumps({"chat_id": self.chat_id, "text": text}).encode()
        request = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except OSError as e:
            log.warning("Telegram notification failed: %s", e)
