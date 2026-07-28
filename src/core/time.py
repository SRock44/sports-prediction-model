"""UTC-aware datetime helpers. Everything in this system is stored and compared UTC."""

from __future__ import annotations

from datetime import UTC, date, datetime


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def end_of_today_et() -> datetime:
    """UTC instant of the next Eastern-time midnight (i.e. the end of "today" ET).

    Used to cap the daily score_nba_upcoming/score_mlb_upcoming window so a game
    tomorrow doesn't get a "preview" prediction today that then changes when
    tomorrow's own run (and possibly a newly-promoted champion) rescores it -
    picks should only exist once, on the morning of the game itself.
    """
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    end_et = (now_et + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return end_et.astimezone(UTC)


def as_of_for_game(scheduled_utc: datetime) -> datetime:
    """The as-of timestamp used for features: 1 hour before tip-off / first pitch.

    This is the critical constant that prevents leakage: we only use data that
    existed at this point in time when building features for this game.
    """
    from datetime import timedelta

    return scheduled_utc - timedelta(hours=1)


def nba_season_for_date(d: date) -> int:
    """Return the NBA season year for a given date.
    NBA seasons start in October; we label by the calendar year in which the season begins.
    This matches the nba_api convention: 2024 → '2024-25'.
    """
    return d.year if d.month >= 10 else d.year - 1


def mlb_season_for_date(d: date) -> int:
    """MLB season is calendar-year based."""
    return d.year


def wnba_season_for_date(d: date) -> int:
    """WNBA season is calendar-year based (runs May-September within one year)."""
    return d.year


def nfl_season_for_date(d: date) -> int:
    """Return the NFL season year for a given date.

    NFL seasons start in September and run through the Super Bowl in February;
    we label by the calendar year in which the season begins (matches
    nfl_data_py/nflverse convention: the 2026 season covers Sep 2026-Feb 2027).
    """
    return d.year if d.month >= 3 else d.year - 1


def nhl_season_for_date(d: date) -> int:
    """NHL season runs October-June; label by the calendar year the season begins
    (matches the NHL API's own `season` convention, e.g. 20242025 encodes the
    2024-25 season — this returns the leading year, 2024)."""
    return d.year if d.month >= 10 else d.year - 1


def soccer_season_for_date(d: date) -> int:
    """Top-5 European league season runs August-May; label by the calendar year the
    season begins (matches Understat/soccerdata's own season convention, e.g.
    "2024-2025" -> season_start_year=2024 — this returns that leading year)."""
    return d.year if d.month >= 7 else d.year - 1


def nascar_season_for_date(d: date) -> int:
    """NASCAR Cup season is calendar-year based (runs Feb-Nov within one year),
    same shape as MLB/WNBA's helpers above."""
    return d.year
