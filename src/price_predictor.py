"""
Short-term price forecasting - LSTM sequence models (PyTorch)
=============================================================

Two models share one architecture:

* ``PricePredictorEngine``        price-only LSTM (1 input feature)
* ``HybridPricePredictorEngine``  risk-aware LSTM (price + market-risk index)

Design choices
--------------
* **Scale-free inputs and targets.** Each 30-day input window is expressed
  relative to its last price (p_t / p_origin - 1) and the model predicts the
  next 30 days relative to that same origin. Oil has traded between $20 and
  $120 in the training period; working in relative terms lets one network
  learn from every regime instead of memorising price levels.
* **Direct multi-horizon output.** A single forward pass emits all 30 future
  days, avoiding the error compounding of recursive one-step forecasts.
* **Honest training loop.** Mini-batches, gradient clipping, a time-ordered
  validation split (with a gap equal to the horizon so targets never overlap)
  and early stopping on validation loss, with a fixed seed for repeatability.
* **Risk channel.** The hybrid model is trained on a real historical risk
  signal - the percentile of the CBOE Oil Volatility Index (OVX) - so the
  relationship between risk and future prices is learned from data. At
  inference time that channel can be driven by the NLP news-risk score.
* **Quantile output.** Market stress predicts the *spread* of future prices
  far better than their direction, so the risk-aware model is trained with a
  pinball (quantile) loss and emits the 10th, 50th and 90th percentile paths.
  Raising the risk input visibly widens the learned range of outcomes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30        # trading days of history fed to the model
HORIZON = 30        # trading days forecast in one pass
RET_SCALE = 10.0    # relative returns are multiplied by this for better conditioning

# Small recurrent nets train faster on a handful of threads than on all cores.
torch.set_num_threads(max(1, min(6, torch.get_num_threads())))

# Streamlit's file watcher inspects every imported module's __path__;
# torch.classes raises on that lookup and floods the console with warnings.
try:
    torch.classes.__path__ = []
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(prices, origins, seq_len=SEQ_LEN, horizon=HORIZON, risk=None, with_targets=True):
    """
    Builds model inputs for each forecast origin (index of the last known day).

    Returns ``X`` of shape (n, seq_len, n_features) and, optionally, ``y`` of
    shape (n, horizon): the scaled relative change from the origin price.
    """
    prices = np.asarray(prices, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    base = prices[origins][:, None]

    window_idx = origins[:, None] + np.arange(-seq_len + 1, 1)[None, :]
    features = [((prices[window_idx] / base) - 1.0)[..., None] * RET_SCALE]
    if risk is not None:
        features.append(np.asarray(risk, dtype=np.float64)[window_idx][..., None])
    X = np.concatenate(features, axis=-1).astype(np.float32)

    if not with_targets:
        return X, None
    target_idx = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    y = (((prices[target_idx] / base) - 1.0) * RET_SCALE).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class SequenceLSTM(nn.Module):
    """Stacked LSTM encoder followed by a linear multi-horizon head."""

    def __init__(self, n_features=1, hidden_size=64, num_layers=2, horizon=HORIZON, dropout=0.2,
                 n_outputs=1):
        super().__init__()
        self.horizon = horizon
        self.n_outputs = n_outputs
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, horizon * n_outputs))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        if self.n_outputs == 1:
            return out                                       # (batch, horizon)
        return out.view(-1, self.n_outputs, self.horizon)    # (batch, quantile, horizon)


class PinballLoss(nn.Module):
    """Quantile regression loss averaged over quantiles and horizons."""

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1))

    def forward(self, pred, target):
        diff = target.unsqueeze(1) - pred
        return torch.maximum(self.q * diff, (self.q - 1) * diff).mean()


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
class PricePredictorEngine:
    """Price-only LSTM: learns 30-day price paths from 30 days of price history."""

    n_features = 1
    uses_risk = False
    quantiles = None           # None -> point forecast trained with MSE
    label = "Price-only LSTM"

    def __init__(self, sequence_length=SEQ_LEN, output_days=HORIZON, hidden_size=64,
                 num_layers=2, dropout=0.2, seed=42):
        self.seq_length = sequence_length
        self.output_days = output_days
        self.seed = seed
        torch.manual_seed(seed)
        n_outputs = len(self.quantiles) if self.quantiles else 1
        self.model = SequenceLSTM(self.n_features, hidden_size, num_layers, output_days, dropout, n_outputs)
        self.history = {"train": [], "val": []}
        self.train_info: dict = {}
        self.is_trained = False

    # -- helpers ------------------------------------------------------------
    def _risk_arg(self, risk, n):
        if not self.uses_risk:
            return None
        if risk is None:
            raise ValueError(f"{self.label} needs a risk series")
        risk = np.asarray(risk, dtype=np.float64)
        if len(risk) != n:
            raise ValueError("risk series must be the same length as prices")
        return risk

    def _origins(self, n):
        return np.arange(self.seq_length - 1, n - self.output_days)

    # -- training -----------------------------------------------------------
    def fit(self, prices, risk=None, epochs=40, batch_size=128, lr=1e-3,
            val_fraction=0.1, patience=6, verbose=False):
        """Trains on every available window of ``prices`` (oldest first)."""
        prices = np.asarray(prices, dtype=np.float64)
        risk = self._risk_arg(risk, len(prices))
        origins = self._origins(len(prices))
        if len(origins) < 200:
            raise ValueError("Not enough price history to train the LSTM")

        X, y = make_windows(prices, origins, self.seq_length, self.output_days, risk)

        # Time-ordered split; drop `horizon` windows between train and validation
        # so no validation target is visible inside a training target.
        n_val = max(50, int(len(X) * val_fraction))
        X_tr, y_tr = X[: -(n_val + self.output_days)], y[: -(n_val + self.output_days)]
        X_va, y_va = X[-n_val:], y[-n_val:]

        X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = PinballLoss(self.quantiles) if self.quantiles else nn.MSELoss()

        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(X_tr))
            running = 0.0
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self.model(X_tr_t[batch]), y_tr_t[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                running += loss.item() * len(batch)

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_va_t), y_va_t).item()
            train_loss = running / len(order)
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if verbose:
                print(f"  epoch {epoch + 1:02d}  train {train_loss:.4f}  val {val_loss:.4f}")

            if val_loss < best_val - 1e-6:
                best_val, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_trained = True
        self.train_info = {
            "epochs_run": len(self.history["val"]),
            "best_val_loss": round(best_val, 5),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
        }
        return self

    # -- inference ----------------------------------------------------------
    def _raw_paths(self, prices, origins, risk=None):
        """Model output converted to price paths; quantile models keep the quantile axis."""
        prices = np.asarray(prices, dtype=np.float64)
        risk = self._risk_arg(risk, len(prices))
        X, _ = make_windows(prices, origins, self.seq_length, self.output_days, risk, with_targets=False)
        self.model.eval()
        with torch.no_grad():
            rel = self.model(torch.from_numpy(X)).numpy().astype(np.float64) / RET_SCALE
        base = prices[np.asarray(origins)]
        if rel.ndim == 3:
            rel = np.sort(rel, axis=1)  # guard against quantile crossing
            return base[:, None, None] * (1.0 + rel)
        return base[:, None] * (1.0 + rel)

    def predict_origins(self, prices, origins, risk=None):
        """Central forecast paths (n_origins x horizon) for many origins at once."""
        paths = self._raw_paths(prices, origins, risk)
        if paths.ndim == 3:
            return paths[:, list(self.quantiles).index(0.5), :]
        return paths

    def _risk_window(self, n, risk_score):
        if not self.uses_risk:
            return None
        return np.full(n, float(np.clip(0.5 if risk_score is None else risk_score, 0, 1)))

    def forecast(self, prices, risk_score=None):
        """30-day central price path starting after the last element of ``prices``."""
        prices = np.asarray(prices, dtype=np.float64)
        return self.predict_origins(prices, [len(prices) - 1], self._risk_window(len(prices), risk_score))[0]


class HybridPricePredictorEngine(PricePredictorEngine):
    """
    Risk-aware quantile LSTM: each timestep carries [relative price, market risk].

    Trained on the OVX percentile series with a pinball loss, so the network
    learns how the *distribution* of future prices differs between calm and
    stressed markets. At inference the risk channel is set to the score you
    supply (e.g. the NLP news-risk index).
    """

    n_features = 2
    uses_risk = True
    quantiles = (0.1, 0.5, 0.9)
    label = "Risk-aware LSTM"

    def predict_quantiles(self, prices, origins, risk):
        """(n_origins x 3 x horizon) price paths for the 10th / 50th / 90th percentiles."""
        return self._raw_paths(prices, origins, risk)

    def forecast_quantiles(self, prices, risk_score=0.5):
        prices = np.asarray(prices, dtype=np.float64)
        q = self._raw_paths(prices, [len(prices) - 1], self._risk_window(len(prices), risk_score))[0]
        return {"lower": q[0], "median": q[1], "upper": q[2]}

    def risk_sensitivity(self, prices, levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Quantile forecast for each risk level - shows what the model learned."""
        return {float(level): self.forecast_quantiles(prices, level) for level in levels}


if __name__ == "__main__":
    from market_data import load_market_data

    data = load_market_data()
    prices = data["wti"]["Price"].values
    risk = data["risk_index"].values

    print("Training price-only LSTM...")
    std = PricePredictorEngine().fit(prices, verbose=True)
    print("30-day path:", np.round(std.forecast(prices)[[0, 6, 13, 29]], 2))

    print("\nTraining risk-aware LSTM...")
    hybrid = HybridPricePredictorEngine().fit(prices, risk, verbose=True)
    for level, q in hybrid.risk_sensitivity(prices).items():
        print(f"  risk {level:.2f}: day-30 median ${q['median'][-1]:.2f}  "
              f"80% range ${q['lower'][-1]:.2f}-${q['upper'][-1]:.2f}")
