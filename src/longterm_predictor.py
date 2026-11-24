"""
Long-term price outlook - direct multi-horizon gradient boosting
================================================================

Where the LSTM reads day-to-day price action, this model looks at the
slow-moving structure of the market. Daily prices are averaged into monthly
observations and described by engineered features: momentum, volatility,
mean-reversion gaps, drawdown and seasonality.

Instead of predicting one month and feeding the prediction back in (which
compounds errors and creates artificial oscillations), a separate XGBoost
model is trained for each anchor horizon - 1, 2, 3, 6, 9, 12, 18, 24, 30 and
36 months ahead - to predict the cumulative log-return directly. The monthly
path is interpolated between anchors.

Oil prices are close to a random walk, so each horizon's prediction is
shrunk toward the naive "no change" forecast by a weight fitted on
out-of-sample cross-validation predictions (forecast combination). Where the
trees add no skill, that horizon falls back to the naive forecast instead of
adding noise.

Uncertainty bands are the 10th-90th percentiles of each horizon's
out-of-sample errors in time-series cross-validation - empirical, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor

HORIZONS = (1, 2, 3, 6, 9, 12, 18, 24, 30, 36)

FEATURE_LABELS = {
    "ret_1m": "1-month momentum",
    "ret_3m": "3-month momentum",
    "ret_6m": "6-month momentum",
    "ret_12m": "12-month momentum",
    "vol_6m": "6-month volatility",
    "vol_12m": "12-month volatility",
    "gap_ma12": "Price vs 12-month average",
    "gap_ma36": "Price vs 36-month average",
    "drawdown_24m": "Distance from 24-month high",
    "month_sin": "Seasonality (sine)",
    "month_cos": "Seasonality (cosine)",
}

FEATURE_HELP = {
    "Momentum": "Log-return over the last N months - captures trend persistence.",
    "Volatility": "Standard deviation of monthly returns - high values signal unstable regimes.",
    "Price vs average": "How stretched the price is relative to its 12/36-month mean - "
                        "the model's mean-reversion signal.",
    "Distance from 24-month high": "Drawdown from the recent peak - large drawdowns often precede recoveries.",
    "Seasonality": "Month of year encoded on a circle, so December sits next to January.",
}


class LongTermForecaster:
    """Monthly direct multi-horizon model for a 36-month structural outlook."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models: dict[int, object] = {}
        self.weights: dict[int, float] = {}
        self.residual_q: dict[int, dict[float, float]] = {}
        self.last_price: float | None = None
        self.feature_names: list[str] = []
        self.training_metrics: dict = {}
        self.monthly: pd.Series | None = None
        self.backend = "XGBoost" if XGBOOST_AVAILABLE else "Gradient Boosting (scikit-learn)"

    # ------------------------------------------------------------------
    def _new_model(self):
        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2.0,
                random_state=self.seed,
                n_jobs=4,
            )
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=self.seed
        )

    @staticmethod
    def to_monthly(df: pd.DataFrame) -> pd.Series:
        return df["Price"].resample("ME").mean().dropna()

    @staticmethod
    def engineer_features(monthly: pd.Series) -> pd.DataFrame:
        """Features at month t use only information available at month t."""
        log_p = np.log(monthly)
        r1 = log_p.diff()
        f = pd.DataFrame(index=monthly.index)
        f["ret_1m"] = r1
        f["ret_3m"] = log_p.diff(3)
        f["ret_6m"] = log_p.diff(6)
        f["ret_12m"] = log_p.diff(12)
        f["vol_6m"] = r1.rolling(6).std()
        f["vol_12m"] = r1.rolling(12).std()
        f["gap_ma12"] = log_p - log_p.rolling(12).mean()
        f["gap_ma36"] = log_p - log_p.rolling(36).mean()
        f["drawdown_24m"] = log_p - log_p.rolling(24).max()
        month = monthly.index.month
        f["month_sin"] = np.sin(2 * np.pi * month / 12)
        f["month_cos"] = np.cos(2 * np.pi * month / 12)
        return f

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        monthly = self.to_monthly(df)
        log_p = np.log(monthly)
        feats = self.engineer_features(monthly)
        self.feature_names = list(feats.columns)
        valid_rows = feats.dropna()

        per_h = {}
        for h in HORIZONS:
            target = (log_p.shift(-h) - log_p).rename("target")
            data = valid_rows.join(target).dropna()
            X, y = data[self.feature_names].values, data["target"].values

            oos_pred, oos_true = [], []
            for tr, te in TimeSeriesSplit(n_splits=5).split(X):
                # Leave a gap of h months so no training target overlaps the test window.
                tr = tr[tr < te[0] - h]
                if len(tr) < 48:
                    continue
                model = self._new_model()
                model.fit(X[tr], y[tr])
                oos_pred.extend(model.predict(X[te]))
                oos_true.extend(y[te])
            oos_pred, oos_true = np.asarray(oos_pred), np.asarray(oos_true)

            # Least-squares shrinkage toward the naive (zero-return) forecast.
            denom = float(np.sum(oos_pred ** 2))
            weight = float(np.clip(np.sum(oos_pred * oos_true) / denom, 0.0, 1.0)) if denom > 0 else 0.0
            blended = weight * oos_pred
            res = oos_true - blended if len(oos_true) else np.array([0.0])

            final = self._new_model()
            final.fit(X, y)
            self.models[h] = final
            self.weights[h] = weight
            self.residual_q[h] = {q: float(np.quantile(res, q)) for q in (0.10, 0.25, 0.75, 0.90)}
            rmse = float(np.sqrt(np.mean(res ** 2)))
            rmse_naive = float(np.sqrt(np.mean(oos_true ** 2))) if len(oos_true) else np.nan
            nonzero = blended != 0
            per_h[h] = {
                "rmse": rmse,
                "rmse_naive": rmse_naive,
                "skill": (1 - rmse / rmse_naive) * 100 if rmse_naive else 0.0,
                "weight": weight,
                "dir_acc": float(np.mean(np.sign(blended[nonzero]) == np.sign(oos_true[nonzero])) * 100)
                if nonzero.any() else np.nan,
                "n": int(len(X)),
            }

        self.monthly = monthly
        self.last_price = float(df["Price"].iloc[-1])
        self.training_metrics = {
            "per_horizon": per_h,
            "cv_rmse_12m": per_h[12]["rmse"],
            "cv_rmse_naive_12m": per_h[12]["rmse_naive"],
            "skill_12m": per_h[12]["skill"],
            "skill_36m": per_h[36]["skill"],
            "dir_acc_12m": per_h[12]["dir_acc"],
            "avg_weight": float(np.mean(list(self.weights.values()))),
            "n_samples": per_h[1]["n"],
            "n_features": len(self.feature_names),
            "n_models": len(HORIZONS),
            "first_month": valid_rows.index[0],
            "last_month": valid_rows.index[-1],
        }
        return self

    def predict_longterm(self, df: pd.DataFrame = None, months: int = 36) -> pd.DataFrame:
        """
        Monthly path anchored at today's price, interpolated between horizon
        anchors, with empirical 50 % (Lower50/Upper50) and 80 % (Lower/Upper) bands.
        """
        if not self.models:
            raise RuntimeError("Call train() before predict_longterm()")
        latest = self.engineer_features(self.monthly).iloc[[-1]][self.feature_names].values
        anchors = np.array([0] + list(HORIZONS), dtype=float)
        shrunk = np.array([self.weights.get(h, 1.0) * float(self.models[h].predict(latest)[0])
                           for h in HORIZONS])
        # Light smoothing across neighbouring horizons so independently fitted
        # weights don't create kinks in the path.
        padded = np.concatenate([[shrunk[0]], shrunk, [shrunk[-1]]])
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        mean_r = np.concatenate([[0.0], smoothed])

        steps = np.arange(1, months + 1)
        path_r = np.interp(steps, anchors, mean_r)

        def band(q):
            return np.interp(steps, anchors, np.array([0.0] + [self.residual_q[h][q] for h in HORIZONS]))

        base = self.last_price if self.last_price is not None else float(self.monthly.iloc[-1])
        dates = [self.monthly.index[-1] + pd.offsets.MonthEnd(int(s)) for s in steps]
        return pd.DataFrame({
            "Price": base * np.exp(path_r),
            "Lower": base * np.exp(path_r + band(0.10)),
            "Lower50": base * np.exp(path_r + band(0.25)),
            "Upper50": base * np.exp(path_r + band(0.75)),
            "Upper": base * np.exp(path_r + band(0.90)),
        }, index=pd.DatetimeIndex(dates, name="Date"))

    # ------------------------------------------------------------------
    def get_feature_importance(self, top_n: int = 11):
        """Gain importance averaged across the horizon models."""
        if not self.models:
            return []
        total = np.zeros(len(self.feature_names))
        for model in self.models.values():
            imp = np.asarray(getattr(model, "feature_importances_", np.zeros(len(total))), dtype=float)
            total += imp / imp.sum() if imp.sum() else imp
        total /= len(self.models)
        pairs = sorted(zip(self.feature_names, total), key=lambda p: p[1], reverse=True)
        return [(FEATURE_LABELS.get(name, name), float(score)) for name, score in pairs[:top_n]]

    @staticmethod
    def get_trend_analysis(predictions: pd.DataFrame, current_price: float) -> dict:
        prices = predictions["Price"]
        end_price = float(prices.iloc[-1])
        change = (end_price / current_price - 1) * 100
        if change > 15:
            trend, tone = "Strong uptrend", "up"
        elif change > 5:
            trend, tone = "Moderate uptrend", "up"
        elif change > -5:
            trend, tone = "Range-bound", "neutral"
        elif change > -15:
            trend, tone = "Moderate downtrend", "down"
        else:
            trend, tone = "Strong downtrend", "down"

        yearly = prices.groupby(prices.index.year).mean()
        return {
            "trend": trend,
            "tone": tone,
            "total_change_pct": change,
            "end_price": end_price,
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "avg_price": float(prices.mean()),
            "price_12m": float(prices.iloc[min(11, len(prices) - 1)]),
            "price_24m": float(prices.iloc[min(23, len(prices) - 1)]),
            "yearly": yearly,
            "months": int(len(prices)),
        }


if __name__ == "__main__":
    from market_data import load_market_data

    data = load_market_data()
    forecaster = LongTermForecaster().train(data["wti"])
    for h, m in forecaster.training_metrics["per_horizon"].items():
        print(f"h={h:2d}m  rmse {m['rmse']:.3f}  naive {m['rmse_naive']:.3f}  skill {m['skill']:+.1f}%  "
              f"weight {m['weight']:.2f}  dir {m['dir_acc']:.0f}%  n={m['n']}")
    outlook = forecaster.predict_longterm(months=36)
    print(outlook.iloc[[0, 5, 11, 23, 35]].round(2))
    for name, score in forecaster.get_feature_importance():
        print(f"  {name:32s} {score:.3f}")
