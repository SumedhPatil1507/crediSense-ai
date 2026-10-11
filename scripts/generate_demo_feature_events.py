"""Create a deterministic, synthetic Feast event file for local PIT demos."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for idx in range(1, 6):
        for month in range(4):
            event_time = base + timedelta(days=30 * month)
            rows.append({
                "applicant_id": f"demo-applicant-{idx:03d}",
                "pan_hash": f"synthetic-pan-hash-{idx:03d}",
                "event_timestamp": event_time,
                "created_timestamp": event_time + timedelta(minutes=1),
                "rolling_credit_utilization": round(min(0.15 * idx + 0.04 * month, 0.95), 3),
                "payment_velocity": round(max(0.98 - 0.06 * idx - 0.02 * month, 0.4), 3),
                "debt_to_income_ratio": round(0.12 * idx + 0.025 * month, 3),
            })
    target = Path(__file__).resolve().parents[1] / "data" / "credit_feature_events.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    print(f"Wrote {len(rows)} synthetic feature events to {target}")


if __name__ == "__main__":
    main()
