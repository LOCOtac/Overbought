"""
overbought_oversold_tool.py
────────────────────────────────────────────────────────────────────────────────
Multi-indicator OVERBOUGHT / OVERSOLD / NEUTRAL classifier.
Powered by Financial Modeling Prep (FMP) stable API.

Classification requires at least 3 confirming signals — no forced binary result.

API Key (checked in order):
  1. .env file next to script:  FMP_API_KEY=your_key_here
  2. Environment variable:      set FMP_API_KEY=your_key_here
  3. CLI flag:                  --api-key YOUR_KEY

Usage:
  python overbought_oversold_tool.py AAPL
  python overbought_oversold_tool.py TSLA --lookback 300
  python overbought_oversold_tool.py SPY  --api-key YOUR_KEY

Python import:
  from overbought_oversold_tool import analyze_overbought_oversold
  result = analyze_overbought_oversold("AAPL")
  print(result)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np


# =============================================================================
# .env Loader  (pure Python — zero pip packages needed)
# =============================================================================

def _load_dotenv_manual() -> None:
    """Read KEY=VALUE pairs from a .env file next to this script into os.environ."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    env_path = os.path.join(script_dir, ".env")
    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_manual()


def _resolve_api_key(cli_key: Optional[str] = None) -> str:
    """Resolve FMP API key: CLI flag → .env / environment variable."""
    key = (cli_key or "").strip() or os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "No FMP API key found.\n\n"
            "  Option 1 — add to .env file next to the script:\n"
            "      FMP_API_KEY=your_key_here\n\n"
            "  Option 2 — set an environment variable:\n"
            "      Windows : set FMP_API_KEY=your_key_here\n"
            "      Mac/Linux: export FMP_API_KEY=your_key_here\n\n"
            "  Option 3 — pass it on the CLI:\n"
            "      python overbought_oversold_tool.py AAPL --api-key your_key_here\n\n"
            "  Get a free key at: https://site.financialmodelingprep.com/developer/docs"
        )
    return key


# =============================================================================
# FMP Data Fetching
# =============================================================================

def fetch_ohlcv(symbol: str, api_key: str, lookback_days: int = 300) -> pd.DataFrame:
    """
    Fetch daily OHLCV from FMP stable API.
    Uses from/to date range — works on free-tier accounts.
    """
    end_dt    = datetime.today()
    start_dt  = end_dt - timedelta(days=int(lookback_days * 1.6))
    end_str   = end_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%Y-%m-%d")

    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={symbol.upper()}&from={start_str}&to={end_str}&apikey={api_key}"
    )

    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not reach Financial Modeling Prep. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("FMP request timed out. Try again.")

    if response.status_code in (401, 403):
        raise ValueError(
            f"FMP returned {response.status_code}.\n\n"
            "  Your API key appears to be invalid or expired.\n"
            "  Steps to fix:\n"
            "    1. Go to https://site.financialmodelingprep.com/developer/docs\n"
            "    2. Sign in and copy your API key from the dashboard\n"
            "    3. Update your .env file: FMP_API_KEY=your_correct_key"
        )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise ValueError(f"FMP HTTP error {response.status_code}: {exc}") from exc

    try:
        payload = response.json()
    except Exception:
        raise ValueError("FMP returned an unexpected non-JSON response.")

    if isinstance(payload, list):
        historical = payload
    elif isinstance(payload, dict):
        if "Error Message" in payload:
            raise ValueError(f"FMP error for '{symbol}': {payload['Error Message']}")
        historical = payload.get("historical", [])
    else:
        historical = []

    if not historical:
        raise ValueError(
            f"No data returned for '{symbol}'. "
            "Check the ticker symbol (e.g. AAPL, TSLA, SPY)."
        )
    if len(historical) < 60:
        raise ValueError(
            f"Insufficient data for '{symbol}': "
            f"got {len(historical)} bars, need at least 60."
        )

    df = pd.DataFrame(historical)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise column names to lowercase
    col_map = {c.lower(): c for c in df.columns}
    needed  = {"open", "high", "low", "close", "volume"}
    missing = needed - set(col_map.keys())
    if missing:
        raise ValueError(f"FMP response missing columns: {missing}")

    df = df.rename(columns={col_map[c]: c for c in needed})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# Indicator Calculations
# =============================================================================

def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(rsi_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Stochastic RSI — normalises RSI into 0–100 range.
    StochRSI = (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n)) * 100
    """
    rsi_min = rsi_series.rolling(period).min()
    rsi_max = rsi_series.rolling(period).max()
    denom   = (rsi_max - rsi_min).replace(0, np.nan)
    return ((rsi_series - rsi_min) / denom) * 100


def _bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple:
    """Returns (middle, upper, lower) Bollinger Band series."""
    middle = series.rolling(period).mean()
    std    = series.rolling(period).std(ddof=0)
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return middle, upper, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all ten indicators and return scalar values for the latest bar.

    Returns
    -------
    dict with keys:
        price, rsi14, rsi5, stoch_rsi,
        bb_upper, bb_middle, bb_lower,
        ma20, ma50,
        dist_20ma_pct, dist_50ma_pct,
        ret3d, ret5d,
        volume_ratio, atr_pct
    """
    close  = df["close"]
    volume = df["volume"]

    # RSI
    rsi14_series   = _rsi(close, 14)
    rsi5_series    = _rsi(close, 5)
    stochrsi_series= _stoch_rsi(rsi14_series, 14)

    # Bollinger Bands
    bb_mid, bb_up, bb_lo = _bollinger_bands(close, 20, 2.0)

    # Moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # Volume
    vol_ma20 = volume.rolling(20).mean()

    # ATR
    atr_series = _atr(df, 14)

    # Latest values
    price    = close.iloc[-1]
    lat_vol  = volume.iloc[-1]
    avg_vol  = vol_ma20.iloc[-1]
    atr14    = atr_series.iloc[-1]

    # Returns
    prev3 = close.iloc[-4] if len(close) >= 4 else close.iloc[0]
    prev5 = close.iloc[-6] if len(close) >= 6 else close.iloc[0]

    def _pct(a, b):
        return ((a / b) - 1) * 100 if b and b != 0 else 0.0

    return {
        "price":          price,
        "rsi14":          rsi14_series.iloc[-1],
        "rsi5":           rsi5_series.iloc[-1],
        "stoch_rsi":      stochrsi_series.iloc[-1],
        "bb_upper":       bb_up.iloc[-1],
        "bb_middle":      bb_mid.iloc[-1],
        "bb_lower":       bb_lo.iloc[-1],
        "ma20":           ma20.iloc[-1],
        "ma50":           ma50.iloc[-1],
        "dist_20ma_pct":  _pct(price, ma20.iloc[-1]),
        "dist_50ma_pct":  _pct(price, ma50.iloc[-1]),
        "ret3d":          _pct(price, prev3),
        "ret5d":          _pct(price, prev5),
        "volume_ratio":   (lat_vol / avg_vol) if avg_vol else 1.0,
        "atr_pct":        (atr14 / price * 100) if price else 0.0,
    }


# =============================================================================
# Signal Detection
# =============================================================================

def detect_signals(ind: dict) -> dict:
    """
    Evaluate each of the 7 OVERBOUGHT and 7 OVERSOLD signals.

    Returns
    -------
    dict with keys:
        ob_signals  : list of triggered overbought signal names
        os_signals  : list of triggered oversold signal names
        ob_count    : int
        os_count    : int
    """
    price        = ind["price"]
    rsi14        = ind["rsi14"]
    rsi5         = ind["rsi5"]
    stoch_rsi    = ind["stoch_rsi"]
    bb_upper     = ind["bb_upper"]
    bb_lower     = ind["bb_lower"]
    dist_20ma    = ind["dist_20ma_pct"]
    ret5d        = ind["ret5d"]
    vol_ratio    = ind["volume_ratio"]

    ob_signals = []
    os_signals = []

    # ── Overbought signals ────────────────────────────────────────────────────
    if rsi14 >= 70:
        ob_signals.append(f"RSI14={rsi14:.1f} >= 70")

    if rsi5 >= 80:
        ob_signals.append(f"RSI5={rsi5:.1f} >= 80")

    if not np.isnan(stoch_rsi) and stoch_rsi >= 80:
        ob_signals.append(f"StochRSI={stoch_rsi:.1f} >= 80")

    if not np.isnan(bb_upper):
        bb_gap_pct = ((price - bb_upper) / bb_upper) * 100
        if bb_gap_pct >= -1.0:   # at or within 1% below upper band
            ob_signals.append(
                f"Price ${price:.2f} near/above upper BB (${bb_upper:.2f})"
            )

    if dist_20ma >= 3.0:
        ob_signals.append(f"Price {dist_20ma:+.1f}% above 20-MA (>= +3%)")

    if ret5d >= 5.0:
        ob_signals.append(f"5-day return {ret5d:+.1f}% >= +5%")

    if vol_ratio >= 1.5 and ret5d > 0:
        ob_signals.append(
            f"Volume {vol_ratio:.1f}x avg on upside move"
        )

    # ── Oversold signals ──────────────────────────────────────────────────────
    if rsi14 <= 30:
        os_signals.append(f"RSI14={rsi14:.1f} <= 30")

    if rsi5 <= 20:
        os_signals.append(f"RSI5={rsi5:.1f} <= 20")

    if not np.isnan(stoch_rsi) and stoch_rsi <= 20:
        os_signals.append(f"StochRSI={stoch_rsi:.1f} <= 20")

    if not np.isnan(bb_lower):
        bb_gap_pct = ((price - bb_lower) / bb_lower) * 100
        if bb_gap_pct <= 1.0:    # at or within 1% above lower band
            os_signals.append(
                f"Price ${price:.2f} near/below lower BB (${bb_lower:.2f})"
            )

    if dist_20ma <= -3.0:
        os_signals.append(f"Price {dist_20ma:+.1f}% below 20-MA (<= -3%)")

    if ret5d <= -5.0:
        os_signals.append(f"5-day return {ret5d:+.1f}% <= -5%")

    if vol_ratio >= 1.5 and ret5d < 0:
        os_signals.append(
            f"Volume {vol_ratio:.1f}x avg on downside move"
        )

    return {
        "ob_signals": ob_signals,
        "os_signals": os_signals,
        "ob_count":   len(ob_signals),
        "os_count":   len(os_signals),
    }


# =============================================================================
# Classification + Strength
# =============================================================================

def classify(ob_count: int, os_count: int) -> tuple:
    """
    Returns (classification, strength).

    OVERBOUGHT / OVERSOLD requires >= 3 signals.
    Strength: 4+ = STRONG, 3 = MODERATE.
    If neither side reaches 3, returns NEUTRAL / NONE.
    """
    # Both sides strong — pick the dominant side
    if ob_count >= 3 and os_count >= 3:
        if ob_count >= os_count:
            side  = "OVERBOUGHT"
            count = ob_count
        else:
            side  = "OVERSOLD"
            count = os_count
    elif ob_count >= 3:
        side  = "OVERBOUGHT"
        count = ob_count
    elif os_count >= 3:
        side  = "OVERSOLD"
        count = os_count
    else:
        return "NEUTRAL", "NONE"

    strength = "STRONG" if count >= 4 else "MODERATE"
    return side, strength


def build_reason(
    classification: str,
    strength: str,
    ob_signals: list,
    os_signals: list,
    ind: dict,
) -> str:
    """Generate a concise one-to-two sentence plain-English reason."""
    rsi14     = ind["rsi14"]
    dist_20ma = ind["dist_20ma_pct"]
    ret5d     = ind["ret5d"]
    vol_ratio = ind["volume_ratio"]

    if classification == "OVERBOUGHT":
        lead = f"{len(ob_signals)} overbought signals fired"
        body = (
            f"RSI-14 at {rsi14:.1f}, price is {dist_20ma:+.1f}% above the 20-MA, "
            f"and the 5-day return is {ret5d:+.1f}%"
        )
        if vol_ratio >= 1.5 and ret5d > 0:
            body += f" with volume running {vol_ratio:.1f}x the 20-day average"
        return f"{lead}: {body}."

    if classification == "OVERSOLD":
        lead = f"{len(os_signals)} oversold signals fired"
        body = (
            f"RSI-14 at {rsi14:.1f}, price is {dist_20ma:+.1f}% from the 20-MA, "
            f"and the 5-day return is {ret5d:+.1f}%"
        )
        if vol_ratio >= 1.5 and ret5d < 0:
            body += f" with volume running {vol_ratio:.1f}x the 20-day average"
        return f"{lead}: {body}."

    # NEUTRAL
    return (
        f"RSI-14 is mid-range at {rsi14:.1f} and price is only {dist_20ma:+.1f}% "
        f"from the 20-MA, so the stock is not truly stretched in either direction."
    )


# =============================================================================
# Main Analysis Function (public API)
# =============================================================================

def analyze_overbought_oversold(
    symbol: str,
    api_key: Optional[str] = None,
    lookback_days: int = 300,
) -> dict:
    """
    Full multi-indicator OVERBOUGHT / OVERSOLD / NEUTRAL analysis.

    Parameters
    ----------
    symbol       : Ticker symbol (e.g. 'AAPL', 'TSLA').
    api_key      : FMP API key. Omit to read from .env / environment.
    lookback_days: Days of history to fetch (default 300, min 60).

    Returns
    -------
    dict with keys:
        symbol, price, classification, strength,
        signals_triggered, reason, indicators
    """
    resolved_key = _resolve_api_key(api_key)

    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Ticker symbol cannot be empty.")
    if lookback_days < 60:
        raise ValueError("lookback_days must be at least 60.")

    df  = fetch_ohlcv(symbol, resolved_key, lookback_days)
    ind = compute_indicators(df)

    for k in ("rsi14", "rsi5", "ma20", "bb_upper", "bb_lower"):
        if pd.isna(ind.get(k)):
            raise ValueError(
                f"Could not compute '{k}' — not enough historical data. "
                f"Try increasing lookback_days (current: {lookback_days})."
            )

    signals       = detect_signals(ind)
    classification, strength = classify(
        signals["ob_count"], signals["os_count"]
    )

    # Signals triggered for display
    if classification == "OVERBOUGHT":
        triggered = signals["ob_signals"]
    elif classification == "OVERSOLD":
        triggered = signals["os_signals"]
    else:
        # Show whichever side had more (informational)
        triggered = (
            signals["ob_signals"] if signals["ob_count"] >= signals["os_count"]
            else signals["os_signals"]
        )

    reason = build_reason(
        classification, strength,
        signals["ob_signals"], signals["os_signals"],
        ind,
    )

    return {
        "symbol":            symbol,
        "price":             round(ind["price"], 2),
        "classification":    classification,
        "strength":          strength,
        "signals_triggered": triggered,
        "reason":            reason,
        "indicators":        {k: round(v, 2) for k, v in ind.items()
                              if not isinstance(v, float) or not np.isnan(v)},
    }


def format_result(result: dict) -> str:
    """Format the result dict into the clean 3-line output."""
    return (
        f"\n"
        f"State    : {result['classification']}\n"
        f"Strength : {result['strength']}\n"
        f"Reason   : {result['reason']}\n"
    )


# =============================================================================
# CLI Entry-Point
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overbought_oversold_tool",
        description=(
            "Multi-indicator OVERBOUGHT / OVERSOLD / NEUTRAL stock classifier.\n"
            "Requires 3+ confirming signals — no forced binary result."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python overbought_oversold_tool.py AAPL
  python overbought_oversold_tool.py TSLA --lookback 300
  python overbought_oversold_tool.py SPY  --api-key YOUR_KEY
        """,
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g. AAPL)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="FMP API key (optional if FMP_API_KEY is in .env or environment)",
    )
    parser.add_argument(
        "--lookback", "-l",
        type=int,
        default=300,
        metavar="DAYS",
        help="Trading days of history to fetch (default: 300)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        result = analyze_overbought_oversold(
            symbol       = args.ticker,
            api_key      = args.api_key,
            lookback_days= args.lookback,
        )
        print(format_result(result))

    except (ValueError, ConnectionError, TimeoutError) as exc:
        print(f"\n  Error: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Cancelled.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()