"""
Model validation & backtesting
==============================

Two complementary, leakage-free tests:

1. **Walk-forward hold-out.** Models are trained on the first 80 % of the
   history only. They then forecast 30 days ahead from *every* trading day in
   the remaining 20 %, and each forecast is scored against what actually
   happened. Errors are reported per horizon (day 1 ... day 30) and compared
   with a naive "no-change" forecast - the standard benchmark that any
   price model must beat to be useful.

   The distribution of these out-of-sample errors also yields calibrated
   80 % prediction intervals for the live forecast.

2. **Crisis stress tests.** For each historical shock, fresh models are
   trained only on data available the day before the event, then asked to
   forecast the next 30 trading days blind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from price_predictor import HORIZON, SEQ_LEN, HybridPricePredictorEngine, PricePredictorEngine

CRISIS_EVENTS = {
    "covid_crash": {
        "name": "COVID-19 demand collapse",
        "short": "COVID-19 · 2020",
        "description": "Lockdowns erased roughly a fifth of global oil demand within weeks, "
                       "while the OPEC+ alliance briefly collapsed into a price war.",
        "start": "2020-03-02",
    },
    "russia_ukraine": {
        "name": "Russia-Ukraine war",
        "short": "Russia-Ukraine · 2022",
        "description": "The invasion on 24 February 2022 triggered sanctions on the world's "
                       "second-largest crude exporter; WTI spiked above $120 within two weeks.",
        "start": "2022-02-24",
    },
    "opec_cuts_2023": {
        "name": "OPEC+ surprise production cut",
        "short": "OPEC+ cut · 2023",
        "description": "A surprise 1.16 million barrels-per-day cut announced on 2 April 2023 "
                       "gapped prices higher before demand worries pulled them back.",
        "start": "2023-04-03",
    },
}

MODEL_LABELS = {
    "lstm": "Price-only LSTM",
    "hybrid": "Risk-aware LSTM",
    "naive": "Naive (no change)",
}


def compute_metrics(actual, predicted, origin_price=None):
    """
    Standard regression metrics for a forecast path.

    ``directional_accuracy`` is the share of horizons where the forecast got
    the direction of the move *from the origin price* right - i.e. "did the
    model correctly call up vs down?". Undefined for flat forecasts.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]
    err = predicted - actual

    metrics = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
        "n_samples": int(n),
    }
    if origin_price is not None:
        pred_dir = np.sign(predicted - origin_price)
        true_dir = np.sign(actual - origin_price)
        valid = pred_dir != 0
        metrics["directional_accuracy"] = (
            float(np.mean(pred_dir[valid] == true_dir[valid]) * 100) if valid.any() else None
        )
    return metrics


class ModelBacktester:
    """Runs walk-forward and crisis backtests and stores their results."""

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, train_kwargs=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.train_kwargs = train_kwargs or {}
        self.walkforward: dict = {}
        self.intervals: dict = {}
        self.test_info: dict = {}
        self.crisis: dict = {}

    # ------------------------------------------------------------------
    # 1. Walk-forward hold-out
    # ------------------------------------------------------------------
    def run_walkforward(self, prices, dates, risk, split_ratio=0.80):
        prices = np.asarray(prices, dtype=float)
        risk = np.asarray(risk, dtype=float)
        h = self.horizon
        split = int(len(prices) * split_ratio)

        lstm = PricePredictorEngine().fit(prices[:split], **self.train_kwargs)
        hybrid = HybridPricePredictorEngine().fit(prices[:split], risk[:split], **self.train_kwargs)

        # Every origin whose whole 30-day target lies after the training data.
        origins = np.arange(split - 1, len(prices) - h)
        actual = prices[origins[:, None] + np.arange(1, h + 1)[None, :]]
        origin_prices = prices[origins]

        paths = {
            "lstm": lstm.predict_origins(prices, origins),
            "hybrid": hybrid.predict_origins(prices, origins, risk),
            "naive": np.repeat(origin_prices[:, None], h, axis=1),
        }

        for name, pred in paths.items():
            abs_pct = np.abs(pred - actual) / actual * 100
            entry = {
                "mape_by_h": abs_pct.mean(axis=0),
                "rmse_by_h": np.sqrt(((pred - actual) ** 2).mean(axis=0)),
                "mape_30": float(abs_pct[:, -1].mean()),
                "mape_7": float(abs_pct[:, 6].mean()),
                "mape_1": float(abs_pct[:, 0].mean()),
                "mape_all": float(abs_pct.mean()),
                "rmse_30": float(np.sqrt(((pred[:, -1] - actual[:, -1]) ** 2).mean())),
                "paths": pred,
            }
            if name != "naive":
                true_dir = np.sign(actual[:, -1] - origin_prices)
                pred_dir = np.sign(pred[:, -1] - origin_prices)
                entry["dir_acc_30"] = float(np.mean(true_dir == pred_dir) * 100)
                # Out-of-sample log-error quantiles by horizon -> 80 % interval.
                log_err = np.log(actual / pred)
                self.intervals[name] = {
                    "lo": np.quantile(log_err, 0.10, axis=0),
                    "hi": np.quantile(log_err, 0.90, axis=0),
                    "coverage": 80,
                }
            self.walkforward[name] = entry

        for name in ("lstm", "hybrid"):
            naive = self.walkforward["naive"]["mape_30"]
            self.walkforward[name]["skill_30"] = (1 - self.walkforward[name]["mape_30"] / naive) * 100

        # How often did the risk-aware model's own learned 80 % band contain the outcome?
        q = hybrid.predict_quantiles(prices, origins, risk)
        inside = (actual >= q[:, 0, :]) & (actual <= q[:, 2, :])
        self.walkforward["hybrid"]["coverage_by_h"] = inside.mean(axis=0) * 100
        self.walkforward["hybrid"]["coverage"] = float(inside.mean() * 100)

        dates = pd.DatetimeIndex(dates)
        self.test_info = {
            "train_start": dates[0],
            "train_end": dates[split - 1],
            "test_start": dates[split],
            "test_end": dates[-1],
            "n_forecasts": int(len(origins)),
            "origin_dates": dates[origins],
            "actual": actual,
            "origin_prices": origin_prices,
            "dates": dates,
        }
        return self.walkforward

    # ------------------------------------------------------------------
    # 2. Crisis stress tests
    # ------------------------------------------------------------------
    def run_crisis_tests(self, prices, dates, risk, events=CRISIS_EVENTS, context_days=60):
        prices = np.asarray(prices, dtype=float)
        risk = np.asarray(risk, dtype=float)
        dates = pd.DatetimeIndex(dates)
        h = self.horizon

        for key, event in events.items():
            origin = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(event["start"])))) - 1
            if origin < 500 or origin + h >= len(prices):
                self.crisis[key] = {"event": event, "available": False}
                continue

            history = prices[: origin + 1]
            lstm = PricePredictorEngine().fit(history, **self.train_kwargs)
            hybrid = HybridPricePredictorEngine().fit(history, risk[: origin + 1], **self.train_kwargs)

            actual = prices[origin + 1: origin + 1 + h]
            origin_price = prices[origin]
            forecasts = {
                "lstm": lstm.forecast(history),
                "hybrid": hybrid.forecast(history, risk_score=risk[origin]),
                "naive": np.full(h, origin_price),
            }
            true_move = (actual[-1] / origin_price - 1) * 100
            results = {}
            for name, path in forecasts.items():
                m = compute_metrics(actual, path, origin_price if name != "naive" else None)
                pred_move = (path[-1] / origin_price - 1) * 100
                m["predicted_move"] = float(pred_move)
                # A move under 1% is not a directional call.
                m["direction_correct"] = (
                    None if name == "naive" or abs(pred_move) < 1.0
                    else bool(np.sign(pred_move) == np.sign(true_move))
                )
                results[name] = {"path": path, "metrics": m}

            ctx_start = max(0, origin - context_days)
            self.crisis[key] = {
                "event": event,
                "available": True,
                "origin_date": dates[origin],
                "origin_price": float(origin_price),
                "forecast_dates": dates[origin + 1: origin + 1 + h],
                "actual": actual,
                "actual_move": float(true_move),
                "context_dates": dates[ctx_start: origin + 1],
                "context_prices": prices[ctx_start: origin + 1],
                "results": results,
            }
        return self.crisis

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def headline_metrics(self):
        """Compact numbers for KPI cards and the assistant."""
        if not self.walkforward:
            return {}
        lstm, hybrid, naive = (self.walkforward[k] for k in ("lstm", "hybrid", "naive"))
        return {
            "lstm_mape_30": lstm["mape_30"],
            "hybrid_mape_30": hybrid["mape_30"],
            "naive_mape_30": naive["mape_30"],
            "lstm_dir_30": lstm["dir_acc_30"],
            "hybrid_dir_30": hybrid["dir_acc_30"],
            "lstm_skill_30": lstm["skill_30"],
            "hybrid_skill_30": hybrid["skill_30"],
            "hybrid_coverage": hybrid.get("coverage"),
            "n_forecasts": self.test_info.get("n_forecasts", 0),
        }

    def walkforward_table(self):
        rows = []
        for key in ("lstm", "hybrid", "naive"):
            r = self.walkforward.get(key)
            if not r:
                continue
            rows.append({
                "Model": MODEL_LABELS[key],
                "MAPE day 1 (%)": r["mape_1"],
                "MAPE day 7 (%)": r["mape_7"],
                "MAPE day 30 (%)": r["mape_30"],
                "RMSE day 30 ($)": r["rmse_30"],
                "Direction day 30 (%)": r.get("dir_acc_30"),
                "Skill vs naive (%)": r.get("skill_30"),
            })
        return pd.DataFrame(rows)

    def get_summary_table(self):
        rows = []
        for key, res in self.crisis.items():
            if not res.get("available"):
                continue
            for model in ("lstm", "hybrid", "naive"):
                m = res["results"][model]["metrics"]
                rows.append({
                    "Event": res["event"]["short"],
                    "Model": MODEL_LABELS[model],
                    "MAPE (%)": m["mape"],
                    "RMSE ($)": m["rmse"],
                    "30-day move called": "-" if m["direction_correct"] is None
                    else ("Correct" if m["direction_correct"] else "Missed"),
                })
        return pd.DataFrame(rows)
