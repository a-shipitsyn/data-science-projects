"""Polite HTTP client: fixed delay between requests, retries on 429 / 5xx / network errors."""

import logging
import time

import requests

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimitedClient:
    def __init__(self, delay: float, timeout: int, user_agent: str, retries: int = 3):
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last = 0.0

    def _wait(self):
        pause = self.delay - (time.monotonic() - self._last)
        if pause > 0:
            time.sleep(pause)
        self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, allow_404: bool = False) -> requests.Response | None:
        """GET with retries. Returns None for 404 when allow_404, raises after the last failed retry."""
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, self.retries, e)
                if attempt == self.retries:
                    raise
            else:
                if response.status_code == 404 and allow_404:
                    return None
                if response.status_code not in RETRY_STATUS:
                    response.raise_for_status()
                    return response
                logger.warning("GET %s -> HTTP %d (attempt %d/%d)", url, response.status_code,
                               attempt, self.retries)
                if attempt == self.retries:
                    response.raise_for_status()
            time.sleep(self.delay * 2 ** attempt)
        raise RuntimeError("unreachable")
