"""Market hours and session awareness for NQ futures.

NQ (E-mini NASDAQ-100) trading hours:
- Sunday 5:00 PM CT to Friday 3:10 PM CT
- Daily halt: 3:10 PM CT to 5:00 PM CT (Mon-Thu)
- Weekend close: Friday 3:10 PM CT to Sunday 5:00 PM CT

All times in Central Time (CT) since CME is in Chicago.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import zoneinfo
    CT = zoneinfo.ZoneInfo("America/Chicago")
except ImportError:
    CT = None


def _now_ct() -> datetime:
    """Get current time in Central Time."""
    utc_now = datetime.now(timezone.utc)
    if CT:
        return utc_now.astimezone(CT)
    # Fallback: UTC-5 (CST) or UTC-6 (CDT) - approximate with UTC-5
    return utc_now - timedelta(hours=5)


def is_market_open() -> bool:
    """Check if NQ futures market is currently open.

    Open: Sunday 5:00 PM CT through Friday 3:10 PM CT
    Daily halt: 3:10 PM CT to 5:00 PM CT (Mon-Thu)
    """
    now = _now_ct()
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    minute = now.minute
    time_minutes = hour * 60 + minute

    open_time = 17 * 60  # 5:00 PM = 1020 minutes
    close_time = 15 * 60 + 10  # 3:10 PM = 910 minutes

    # Saturday: always closed
    if weekday == 5:
        return False

    # Sunday: open only after 5:00 PM CT
    if weekday == 6:
        return time_minutes >= open_time

    # Friday: open until 3:10 PM CT only
    if weekday == 4:
        return time_minutes < close_time

    # Mon-Thu: closed between 3:10 PM and 5:00 PM CT (daily halt)
    if close_time <= time_minutes < open_time:
        return False

    return True


def time_until_open() -> timedelta:
    """Get time until market opens. Returns zero if already open."""
    if is_market_open():
        return timedelta(0)

    now = _now_ct()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    time_minutes = hour * 60 + minute

    open_time = 17 * 60  # 5:00 PM CT

    if weekday == 5:
        # Saturday: opens Sunday 5 PM CT
        days_until = 1
        next_open = now.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=days_until)
    elif weekday == 6:
        # Sunday before 5 PM
        if time_minutes < open_time:
            next_open = now.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            return timedelta(0)  # already open
    elif weekday == 4 and time_minutes >= 15 * 60 + 10:
        # Friday after close: opens Sunday 5 PM CT
        days_until = 2
        next_open = now.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=days_until)
    else:
        # Mon-Thu during daily halt (3:10 PM - 5:00 PM)
        next_open = now.replace(hour=17, minute=0, second=0, microsecond=0)

    return max(timedelta(0), next_open - now)


def get_session_info() -> dict:
    """Get current market session info."""
    now = _now_ct()
    open_status = is_market_open()
    time_to_open = time_until_open()

    # Determine session name
    if not open_status:
        weekday = now.weekday()
        if weekday in (5, 6) or (weekday == 4 and now.hour >= 15):
            session = "WEEKEND"
        else:
            session = "DAILY_HALT"
    else:
        hour = now.hour
        if 17 <= hour or hour < 2:
            session = "GLOBEX_EVENING"
        elif 2 <= hour < 8:
            session = "GLOBEX_OVERNIGHT"
        elif 8 <= hour < 9:
            session = "PRE_MARKET"
        elif 9 <= hour < 12:
            session = "RTH_MORNING"  # Regular Trading Hours
        elif 12 <= hour < 15:
            session = "RTH_AFTERNOON"
        else:
            session = "RTH_CLOSE"

    return {
        "open": open_status,
        "session": session,
        "ct_time": now.strftime("%H:%M:%S CT"),
        "time_to_open": str(time_to_open).split(".")[0] if time_to_open > timedelta(0) else None,
    }
