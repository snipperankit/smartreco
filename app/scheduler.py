"""Scheduled proactive delivery.

Multi-channel digest delivery: in-app mailbox (always) + Telegram Bot API
(free — create a bot via @BotFather, get chat_id from @userinfobot).

Scheduler modes:
  - Cron (default): runs once a day at DIGEST_HOUR:DIGEST_MINUTE UTC
  - Interval: set DIGEST_INTERVAL_MINUTES > 0 to repeat every N minutes (demo)
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from html import escape

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from app.agent.triggers import hydrate_products, maybe_generate
from app.config import settings
from app.database import SessionLocal
from app.models import BehavioralEvent, User

log = logging.getLogger("smartreco.scheduler")
_scheduler: AsyncIOScheduler | None = None

_mailbox: deque[dict] = deque(maxlen=50)


async def _active_user_ids(db) -> list[int]:
    """Users with activity in the last 24h who haven't opted out of delivery."""
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    res = await db.execute(
        select(BehavioralEvent.user_id)
        .join(User, User.id == BehavioralEvent.user_id)
        .where(
            BehavioralEvent.created_at >= since,
            User.proactive_delivery_enabled == True,  # noqa: E712
            User.role != "admin",
        )
        .distinct()
    )
    return [row[0] for row in res.all()]


def _send_whatsapp(phone: str, apikey: str, text: str) -> bool:
    """Send a WhatsApp message via CallMeBot (free, HTTP GET)."""
    encoded = urllib.parse.quote_plus(text)
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded}&apikey={apikey}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
            if ok:
                log.info("WhatsApp sent to %s", phone)
            else:
                log.warning("WhatsApp API returned %d", resp.status)
            return ok
    except Exception as exc:
        log.warning("WhatsApp send failed: %s", exc)
        return False


def _format_whatsapp_card(user_email: str, products: list, narrative: str) -> str:
    """Format a WhatsApp-friendly card message with emoji course cards."""
    name = user_email.split("@")[0].title()
    lines = [f"🎓 *SmartReco Daily Picks for {name}*", "", f"_{narrative}_", ""]
    for i, p in enumerate(products[:5], 1):
        lines.append(f"*{i}. {p.title}*")
        lines.append(f"   📂 {p.category} · 💰 ${p.price} · 📊 {p.level}")
        lines.append("")
    lines.append("🔗 Open SmartReco to explore →")
    return "\n".join(lines)


def _format_telegram_card(user_email: str, products: list, narrative: str) -> str:
    """Format a premium Telegram notification card."""
    name = escape(user_email.split("@")[0].title())
    narrative = escape(narrative)
    lines = [
        "🧠 <b>SmartReco</b>",
        "",
        f"Hey <b>{name}</b>, new picks just for you:",
        "",
        f"<i>\u201c{narrative[:250]}\u201d</i>",
        "",
    ]
    level_icons = {"beginner": "\U0001F331", "intermediate": "\u26A1", "advanced": "\U0001F525"}
    for i, p in enumerate(products[:5], 1):
        icon = level_icons.get(p.level, "\U0001F4D8")
        cat = escape(p.category.replace("-", " ").title())
        title = escape(p.title)
        level = escape(p.level.title())
        lines.append(f"{icon} <b>{title}</b>")
        lines.append(f"     {cat} \u2022 ${p.price:.0f} \u2022 {level}")
        if i < min(len(products), 5):
            lines.append("")

    lines.extend([
        "",
        "\u2500" * 28,
        "\U0001F4CA Personalized from your activity",
        "\U0001F916 LangGraph + Mesh API",
        "",
        "\u25B6\uFE0F <b>Open SmartReco</b> to start learning",
    ])
    return "\n".join(lines)


def _send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API (free, HTTP POST)."""
    import httpx
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = httpx.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if r.status_code == 200:
            log.info("Telegram sent to chat %s", chat_id)
            return True
        log.warning("Telegram API returned %d: %s", r.status_code, r.text[:200])
        return False
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


def deliver(email: str, subject: str, body: str, products: list | None = None, narrative: str = "") -> None:
    """Multi-channel delivery: mailbox (always) + Telegram (if configured)."""
    channels = [c.strip() for c in settings.delivery_channels.split(",")]
    now = datetime.now(timezone.utc).isoformat()

    tg_sent = False
    telegram_recipient = settings.telegram_recipient_email.strip().lower()
    if (
        "telegram" in channels
        and settings.telegram_bot_token
        and settings.telegram_chat_id
        and telegram_recipient
        and email.strip().lower() == telegram_recipient
    ):
        card_text = _format_telegram_card(email, products or [], narrative or body)
        tg_sent = _send_telegram(settings.telegram_bot_token, settings.telegram_chat_id, card_text)

    msg = {
        "to": email,
        "subject": subject,
        "body": body,
        "channel": "telegram + mailbox" if tg_sent else "mailbox",
        "telegram_card": _format_telegram_card(email, products or [], narrative or body) if products else None,
        "delivered_at": now,
    }
    _mailbox.appendleft(msg)
    log.info("DIGEST [%s] -> %s | %s", msg["channel"], email, subject)


def get_mailbox() -> list[dict]:
    return list(_mailbox)


async def run_daily_digest() -> None:
    async with SessionLocal() as db:
        user_ids = await _active_user_ids(db)
        log.info("Daily digest sweep: %d active users", len(user_ids))
        for uid in user_ids:
            from hashlib import sha256
            from sqlalchemy import select as sa_select
            from app.models import Recommendation
            res = await db.execute(
                sa_select(Recommendation)
                .where(Recommendation.user_id == uid, Recommendation.is_sent == False)
                .order_by(Recommendation.updated_at.desc())
                .limit(1)
            )
            rec = res.scalar_one_or_none()
            if rec is None:
                continue
            # Content fingerprint: skip if same products as last digest
            digest_hash = sha256(str(sorted(rec.recommended_product_ids)).encode()).hexdigest()[:16]
            user = await db.get(User, uid)
            if user.last_digest_hash == digest_hash:
                log.info("Digest skip %s — same content as last send", user.email)
                rec.is_sent = True
                await db.commit()
                continue
            products = await hydrate_products(db, rec.recommended_product_ids)
            lines = "\n".join(f"  • {p.title} — ${p.price}" for p in products)
            body = f"{rec.narrative_copy}\n\nPicked for you today:\n{lines}"
            deliver(user.email, "Your SmartReco picks for today", body,
                    products=products, narrative=rec.narrative_copy)
            rec.is_sent = True
            user.last_digest_hash = digest_hash
            await db.commit()


def start_scheduler() -> None:
    global _scheduler
    if not settings.enable_scheduler or _scheduler is not None:
        return
    _scheduler = AsyncIOScheduler(timezone="UTC")

    if settings.digest_interval_minutes > 0:
        trigger = IntervalTrigger(minutes=settings.digest_interval_minutes)
        label = f"every {settings.digest_interval_minutes} min"
    else:
        trigger = CronTrigger(hour=settings.digest_hour, minute=settings.digest_minute)
        label = f"daily at {settings.digest_hour:02d}:{settings.digest_minute:02d} UTC"

    _scheduler.add_job(
        run_daily_digest, trigger, id="daily_digest", replace_existing=True,
    )

    channels = [c.strip() for c in settings.delivery_channels.split(",")]
    tg_status = "\u2713 Telegram active" if ("telegram" in channels and settings.telegram_bot_token) else "mailbox only"
    _scheduler.start()
    log.info("Scheduler started — %s (%s)", label, tg_status)


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
