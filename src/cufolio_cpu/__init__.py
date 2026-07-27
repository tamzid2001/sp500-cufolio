"""CPU-only portfolio-optimization building blocks."""

from .returns import daily_returns_from_minute_bars, portfolio_daily_returns

__all__ = ["daily_returns_from_minute_bars", "portfolio_daily_returns"]
