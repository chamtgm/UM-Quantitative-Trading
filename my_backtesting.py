import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Type
import matplotlib.pyplot as plt


class Strategy:
    def __init__(self, data: pd.DataFrame, params: dict):
        self.data = data.copy()  # Avoid mutating original DataFrame
        self.params = params
        self.indicators = {}
        self.positions = []

    def init(self):
        """Initialize indicators and preprocess data."""
        # Feature engineering
        self.data["log_return"] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        self.data["volatility"] = self.data["log_return"].rolling(window=10).std()
        self.data["momentum"] = self.data["log_return"].rolling(window=10).mean()
        self.data["RSI"] = self.calculate_rsi(self.data["Close"], window=14)

        # Drop rows with NaNs
        self.data.dropna(inplace=True)

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(
            self.data[["log_return", "volatility", "momentum", "RSI"]]
        )

        # Dimensionality reduction
        pca = PCA(n_components=2)
        self.features_pca = pca.fit_transform(features)

        # Hidden Markov Model
        self.hmm_model = GaussianHMM(
            n_components=self.params.get("n_components", 3),
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        self.hmm_model.fit(self.features_pca)

        # Predict regimes
        self.data["regime"] = self.hmm_model.predict(self.features_pca)

    def next(self):
        """Generate trading signals based on HMM state probabilities."""
        state_probs = self.hmm_model.predict_proba(self.features_pca)
        bull_state = np.argmax(self.hmm_model.means_[:, 0])
        self.data["position"] = (state_probs[:, bull_state] > self.params.get("threshold", 0.7)).astype(int)

    @staticmethod
    def calculate_rsi(series, window):
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class Backtest:
    def __init__(self, data: pd.DataFrame, strategy: Type[Strategy], cash: float = 10000, commission: float = 0.0006):
        self.original_data = data
        self.strategy_class = strategy
        self.cash = cash
        self.commission = commission
        self.results = {}

    def run(self, **params):
        """Run the backtest."""
        # Copy data to avoid contamination
        data = self.original_data.copy()

        # Initialize and apply strategy
        strategy = self.strategy_class(data, params)
        strategy.init()
        strategy.next()
        data = strategy.data  # use updated data

        # Calculate log returns
        data["future_price"] = data["Close"].shift(-1)
        data["return"] = np.log(data["future_price"] / data["Close"])
        data["strategy_return"] = data["position"] * data["return"]

        # Apply trading fees
        data["trade_cost"] = data["position"].diff().abs() * self.commission
        data["strategy_return"] -= data["trade_cost"]
        data["strategy_return"].fillna(0, inplace=True)

        # Calculate portfolio value
        data["portfolio_value"] = self.cash * (1 + data["strategy_return"].cumsum())

        # Calculate metrics
        sharpe_ratio = data["strategy_return"].mean() / (data["strategy_return"].std() + 1e-6) * np.sqrt(252)
        cumulative_return = data["strategy_return"].cumsum()
        roll_max = cumulative_return.cummax()
        drawdown = roll_max - cumulative_return
        max_drawdown = (drawdown.max() / (roll_max.max() + 1e-6)) * 100
        trade_frequency = data["position"].diff().abs().sum() / len(data)

        # Store results
        self.results = {
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Trade Frequency (%)": round(trade_frequency * 100, 2),
            "Final Portfolio Value": round(data["portfolio_value"].iloc[-1], 2),
        }

        self.data = data  # save for plotting
        return self.results

    def plot(self):
        """Plot portfolio value over time."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data["portfolio_value"], label="Portfolio Value", color="purple")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("combined_data.csv")
    data.rename(columns={"close": "Close"}, inplace=True)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data.set_index("datetime", inplace=True)

    # Run backtest
    backtest = Backtest(data, Strategy)
    results = backtest.run(n_components=3, threshold=0.7)
    print("Backtest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Plot results
    backtest.plot()
