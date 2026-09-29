"""Generate the Sentinel sample dataset.

Writes 180 days of daily e-commerce metrics in long format — one row per day
per region/channel/device/product_category combination — with weekly
seasonality, a slow growth trend, multiplicative noise, and three injected
anomalies in the last 40 days.

Run from the repository root:

    python data/generate.py

Outputs:
    data/sample.csv             the dataset
    data/anomalies_truth.json   ground truth for the injected anomalies

The dataset carries the nine documented columns only. ``conversion_rate`` and
AOV are derived in memory to produce ``orders`` and ``revenue``; the detection
pipeline is expected to recompute them from the raw columns.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parent
SAMPLE_PATH: Path = DATA_DIR / "sample.csv"
TRUTH_PATH: Path = DATA_DIR / "anomalies_truth.json"

# Deterministic generation: same seed, same dataset, every run.
RNG_SEED: int = 42
RNG: np.random.Generator = np.random.default_rng(RNG_SEED)

# Calendar. The window is anchored on its last day; 2026-09-27 is a Sunday, so
# the 180 days run from 2026-04-01.
DAYS: int = 180
END_DATE: date = date(2026, 9, 27)
START_DATE: date = END_DATE - timedelta(days=DAYS - 1)
DATES: pd.DatetimeIndex = pd.date_range(START_DATE, periods=DAYS, freq="D")

# Segment dimensions.
REGIONS: list[str] = ["north_america", "europe", "apac"]
CHANNELS: list[str] = ["organic", "paid", "email", "social"]
DEVICES: list[str] = ["mobile", "desktop", "tablet"]
CATEGORIES: list[str] = ["apparel", "electronics", "home", "grocery"]

# Traffic mix. Shares sum to 1.0 per dimension.
REGION_SHARE: dict[str, float] = {"north_america": 0.45, "europe": 0.35, "apac": 0.20}
CHANNEL_SHARE: dict[str, float] = {"organic": 0.30, "paid": 0.35, "email": 0.15, "social": 0.20}
DEVICE_SHARE: dict[str, float] = {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10}
CATEGORY_SHARE: dict[str, float] = {"apparel": 0.30, "electronics": 0.25, "home": 0.30, "grocery": 0.15}

BASE_DAILY_SESSIONS: float = 120_000.0
TREND_GROWTH: float = 0.18  # total session growth across the window
SESSIONS_NOISE_SIGMA: float = 0.08
CONVERSION_NOISE_SIGMA: float = 0.05
AOV_NOISE_SIGMA: float = 0.10
REFUND_NOISE_SIGMA: float = 0.20

# Weekly seasonality, Monday-first. Traffic rises into the weekend; conversion
# falls as it does.
WEEKDAY_SESSIONS_FACTOR: list[float] = [0.92, 0.95, 0.98, 1.02, 1.15, 1.30, 1.10]
WEEKDAY_CONVERSION_FACTOR: list[float] = [1.00, 1.00, 0.99, 0.99, 0.95, 0.90, 0.96]

# Conversion rate starts at a store-wide baseline and is adjusted by a relative
# factor per channel, device, and category. Keeping the factors centred on 1.0
# (instead of stacking absolute probabilities) holds conversion in a realistic
# 1.5%-8% band.
BASE_CONVERSION: float = 0.038
CHANNEL_CONVERSION_FACTOR: dict[str, float] = {"organic": 1.15, "email": 1.35, "social": 0.70, "paid": 0.85}
DEVICE_CONVERSION_FACTOR: dict[str, float] = {"mobile": 0.85, "desktop": 1.30, "tablet": 0.95}
CATEGORY_CONVERSION_FACTOR: dict[str, float] = {"apparel": 1.00, "electronics": 0.75, "home": 0.90, "grocery": 1.40}

CATEGORY_AOV: dict[str, float] = {"apparel": 65.0, "electronics": 180.0, "home": 90.0, "grocery": 45.0}
REGION_AOV_FACTOR: dict[str, float] = {"north_america": 1.05, "europe": 1.00, "apac": 0.85}

CATEGORY_REFUND_RATE: dict[str, float] = {"apparel": 0.070, "electronics": 0.050, "home": 0.040, "grocery": 0.020}

# Injected anomalies. All three start inside the last 40 days and run through
# the end of the window, each on a different onset.
ANOMALY_WINDOW_DAYS: int = 40
ANOMALY_START_INDEX: int = DAYS - ANOMALY_WINDOW_DAYS
ANOMALY_A_ONSET_INDEX: int = ANOMALY_START_INDEX + 2
ANOMALY_B_ONSET_INDEX: int = ANOMALY_START_INDEX + 16
ANOMALY_C_ONSET_INDEX: int = ANOMALY_START_INDEX + 28


def onset_date(index: int) -> date:
    """Return the calendar date at ``index`` in the generated window."""
    return DATES[index].date()


ANOMALY_A_ONSET: date = onset_date(ANOMALY_A_ONSET_INDEX)
ANOMALY_B_ONSET: date = onset_date(ANOMALY_B_ONSET_INDEX)
ANOMALY_C_ONSET: date = onset_date(ANOMALY_C_ONSET_INDEX)

# (a) Checkout breaks on mobile organic traffic: conversion collapses, so
# orders and revenue fall while sessions stay flat.
ANOMALY_A_SEGMENT: dict[str, str] = {"device": "mobile", "channel": "organic"}
ANOMALY_A_CONVERSION_FACTOR: float = 0.40

# (b) One bad product batch: refunds spike in electronics.
ANOMALY_B_SEGMENT: dict[str, str] = {"product_category": "electronics"}
ANOMALY_B_REFUND_FACTOR: float = 6.0

# (c) Paid traffic floods in but almost never converts.
ANOMALY_C_SEGMENT: dict[str, str] = {"channel": "paid"}
ANOMALY_C_SESSIONS_FACTOR: float = 2.20
ANOMALY_C_CONVERSION_FACTOR: float = 0.45


def anomaly_mask(dates: pd.DatetimeIndex, onset: date) -> np.ndarray:
    """True where the injected anomalies apply, i.e. from ``onset`` onward."""
    return dates.to_numpy() >= np.datetime64(onset)


def session_factor(dates: pd.DatetimeIndex, channel: str) -> np.ndarray:
    """Traffic multiplier: weekly seasonality, growth trend, noise, surge."""
    factor = np.array(WEEKDAY_SESSIONS_FACTOR)[dates.dayofweek.to_numpy()]
    trend = 1.0 + TREND_GROWTH * np.arange(len(dates)) / (len(dates) - 1)
    factor = factor * trend
    if channel == ANOMALY_C_SEGMENT["channel"]:
        factor = factor * np.where(anomaly_mask(dates, ANOMALY_C_ONSET), ANOMALY_C_SESSIONS_FACTOR, 1.0)
    # The lognormal is centred so its mean is 1.0 and the intended traffic level
    # is preserved.
    noise = RNG.lognormal(mean=-SESSIONS_NOISE_SIGMA**2 / 2, sigma=SESSIONS_NOISE_SIGMA, size=len(dates))
    return factor * noise


def conversion_rate(channel: str, device: str, category: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Per-session conversion probability, including the injected collapses."""
    rate = (
        BASE_CONVERSION
        * CHANNEL_CONVERSION_FACTOR[channel]
        * DEVICE_CONVERSION_FACTOR[device]
        * CATEGORY_CONVERSION_FACTOR[category]
    )
    rate = rate * np.array(WEEKDAY_CONVERSION_FACTOR)[dates.dayofweek.to_numpy()]
    if channel == ANOMALY_C_SEGMENT["channel"]:
        rate = rate * np.where(anomaly_mask(dates, ANOMALY_C_ONSET), ANOMALY_C_CONVERSION_FACTOR, 1.0)
    if device == ANOMALY_A_SEGMENT["device"] and channel == ANOMALY_A_SEGMENT["channel"]:
        rate = rate * np.where(anomaly_mask(dates, ANOMALY_A_ONSET), ANOMALY_A_CONVERSION_FACTOR, 1.0)
    noise = RNG.normal(loc=1.0, scale=CONVERSION_NOISE_SIGMA, size=len(dates))
    return rate * noise


def refund_rate(category: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Share of orders refunded, including the injected spike."""
    rate = np.full(len(dates), CATEGORY_REFUND_RATE[category])
    if category == ANOMALY_B_SEGMENT["product_category"]:
        rate = rate * np.where(anomaly_mask(dates, ANOMALY_B_ONSET), ANOMALY_B_REFUND_FACTOR, 1.0)
    noise = RNG.normal(loc=1.0, scale=REFUND_NOISE_SIGMA, size=len(dates))
    return rate * noise


def segment_frame(region: str, channel: str, device: str, category: str) -> pd.DataFrame:
    """Build every day of one segment combination as its own frame."""
    share = (
        REGION_SHARE[region]
        * CHANNEL_SHARE[channel]
        * DEVICE_SHARE[device]
        * CATEGORY_SHARE[category]
    )
    sessions = BASE_DAILY_SESSIONS * share * session_factor(DATES, channel)
    orders = sessions * conversion_rate(channel, device, category, DATES)
    aov = CATEGORY_AOV[category] * REGION_AOV_FACTOR[region] * RNG.normal(
        loc=1.0, scale=AOV_NOISE_SIGMA, size=len(DATES)
    )
    revenue = orders * aov
    refunds = orders * refund_rate(category, DATES)

    return pd.DataFrame(
        {
            "date": DATES,
            "region": region,
            "channel": channel,
            "device": device,
            "product_category": category,
            "sessions": np.round(sessions).astype(int),
            "orders": np.round(orders).astype(int),
            "revenue": np.round(revenue, 2),
            "refunds": np.round(refunds).astype(int),
        }
    )


def build_dataset() -> pd.DataFrame:
    """Concatenate every segment combination into the long-format dataset."""
    frames = [
        segment_frame(region, channel, device, category)
        for region in REGIONS
        for channel in CHANNELS
        for device in DEVICES
        for category in CATEGORIES
    ]
    dataset = pd.concat(frames, ignore_index=True)
    return dataset.sort_values(
        ["date", "region", "channel", "device", "product_category"]
    ).reset_index(drop=True)


def truth_entry(
    anomaly_id: str,
    metric: str,
    segment: dict[str, str],
    onset: date,
    description: str,
    secondary_metric: str | None = None,
) -> dict[str, object]:
    """One ground-truth record covering the anomaly through the end of the window."""
    entry: dict[str, object] = {
        "id": anomaly_id,
        "metric": metric,
        "segment": segment,
        "date_start": onset.isoformat(),
        "date_end": END_DATE.isoformat(),
        "description": description,
    }
    if secondary_metric is not None:
        entry["secondary_metric"] = secondary_metric
    return entry


def build_truth() -> dict[str, object]:
    """Ground truth for the three injected anomalies."""
    return {
        "dataset": SAMPLE_PATH.name,
        "seed": RNG_SEED,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "days": DAYS,
        "anomaly_window_days": ANOMALY_WINDOW_DAYS,
        "anomalies": [
            truth_entry(
                "a",
                "revenue",
                ANOMALY_A_SEGMENT,
                ANOMALY_A_ONSET,
                "Checkout breakage on mobile organic traffic: conversion collapses, "
                "so revenue falls with no drop in sessions.",
            ),
            truth_entry(
                "b",
                "refunds",
                ANOMALY_B_SEGMENT,
                ANOMALY_B_ONSET,
                "Bad electronics batch: refunds spike while revenue and sessions are unchanged.",
            ),
            truth_entry(
                "c",
                "sessions",
                ANOMALY_C_SEGMENT,
                ANOMALY_C_ONSET,
                "Bot flood on paid traffic: sessions surge while conversion collapses, "
                "leaving orders and revenue roughly flat.",
                secondary_metric="conversion_rate",
            ),
        ],
    }


def main() -> None:
    dataset = build_dataset()
    dataset.to_csv(SAMPLE_PATH, index=False)
    TRUTH_PATH.write_text(json.dumps(build_truth(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset):,} rows to {SAMPLE_PATH}")
    print(f"Wrote ground truth for {len(build_truth()['anomalies'])} anomalies to {TRUTH_PATH}")


if __name__ == "__main__":
    main()
