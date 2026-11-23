"""
Sending an alert out of CatchCam: a generic webhook (works with Slack, Discord, Teams, n8n, Zapier...) and/or a Telegram bot.

Settings come from environment variables or <data_dir>/alerts_config.json. They are NOT settable through the web API, so
nobody who can open the dashboard can point alerts (or the server) at an address of their choosing.

  CATCHCAM_WEBHOOK_URL         https://hooks.example.com/...        (a JSON POST: {"text": "...", "content": "...", "alert": {...}})
  CATCHCAM_TELEGRAM_TOKEN      123456:ABC...   (from @BotFather)
  CATCHCAM_TELEGRAM_CHAT       your chat id    (message the bot once, then open https://api.telegram.org/bot<token>/getUpdates)
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

TIMEOUT = 6


def load_config(data_dir) -> dict:
    cfg = {}
    f = Path(data_dir) / "alerts_config.json"
    if f.exists():
        try:
            cfg = json.loads(f.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            cfg = {}
    env = {"webhook_url": os.environ.get("CATCHCAM_WEBHOOK_URL"), "telegram_token": os.environ.get("CATCHCAM_TELEGRAM_TOKEN"),
           "telegram_chat_id": os.environ.get("CATCHCAM_TELEGRAM_CHAT")}
    cfg.update({k: v for k, v in env.items() if v})
    return cfg


def configured(cfg: dict) -> dict:
    return {"webhook": bool(cfg.get("webhook_url")), "telegram": bool(cfg.get("telegram_token") and cfg.get("telegram_chat_id"))}


def _post_json(url: str, payload: dict) -> None:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) addresses are allowed")
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        if r.status >= 300:
            raise RuntimeError(f"server answered {r.status}")


def deliver(cfg: dict, text: str, alert: dict) -> str:
    """Tries every configured channel. Returns e.g. 'webhook: sent; telegram: failed (timed out)' (never raises)."""
    results = []
    if cfg.get("webhook_url"):
        try:
            _post_json(cfg["webhook_url"], {"text": text, "content": text, "alert": alert})
            results.append("webhook: sent")
        except Exception as e:                                                         # noqa: BLE001
            results.append(f"webhook: failed ({str(e)[:80]})")
    if cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        base = (cfg.get("telegram_api_base") or "https://api.telegram.org").rstrip("/")
        try:
            _post_json(f"{base}/bot{cfg['telegram_token']}/sendMessage", {"chat_id": cfg["telegram_chat_id"], "text": text})
            results.append("telegram: sent")
        except Exception as e:                                                         # noqa: BLE001
            msg = str(e).replace(cfg["telegram_token"], "***")                           # never echo the bot token
            results.append(f"telegram: failed ({msg[:80]})")
    return "; ".join(results) if results else "not configured"
