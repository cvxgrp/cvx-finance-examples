import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Blending forecasts

        This notebook considers the blending problem. We have a set of three expert
        volume predictors, which forecast the log-volume of traded Apple stock each day
        from 1982-2024. The objective is to blend these forecasts with convex weights
        to create a new volume predictor that performs better than any of the
        individual experts.
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import cvxpy as cp
    import yfinance as yf
    import matplotlib.pyplot as plt
    from tqdm import trange

    return cp, np, pd, plt, trange, yf


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Setup

        ### Problem statement

        We observe $x_1, \ldots, x_t\in \reals^d$ and seek a forecast
        $\hat x_{t+1}$ of $x_{t+1}$ of the volume of traded Apple stock over period $t+1$.
        We are given $K$ expert predictors $\hat x_{t+1}^1, \ldots, \hat x^{K}_{t+1}$.
        We blend them using weights $\pi_t^1, \ldots, \pi_t^K$
        with $\pi_t \geq 0$, $\mathbf{1}^T \pi_t=1$ to form the blended predictor

        $$
        \hat x_{t+1} = \sum_{k=1}^K \pi_t^k \hat x_{t+1}^k.
        $$

        We find the weights $\pi_t$ as the solution of the convex optimization problem

        $$
        \begin{array}{ll} \text{minimize} &
        \frac{1}{M}\sum_{\tau=t-M}^t |\hat x_\tau - x_\tau| \\
        \text{subject to} & \hat x_\tau = \sum_{k=1}^K \pi_k \hat x^k_\tau, \quad
        \pi \geq 0, \quad  \mathbf{1}^T \pi =1.
        \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Example

        We are given $K=3$ predictors: 5-day (fast), 21-day (medium), and 63-day (slow)
        moving medians of Apple's traded log-volume. We use $M=250$, $\textit{i.e.}$, each
        day we blend the predictions using weights that would have worked well over the
        previous 250 trading days.
        """
    )
    return


@app.cell
def __(np, pd, yf):
    start_date = "1981-09-29"
    end_date = "2024-02-22"

    apple = yf.Ticker("AAPL")
    data = apple.history(period="max")
    volume = data["Volume"].loc[start_date:end_date] / 1e6
    price = data["Close"].loc[start_date:end_date]
    volume = (price * volume)[200:]  # dollar volume
    log_volume = np.log(volume)

    fast = 5
    medium = 21
    slow = 63

    median = log_volume.rolling(5).median()
    median_diff = log_volume - median

    short_term = (
        log_volume.rolling(window=fast).median().shift(1)
    )  # shift to make causal
    medium_term = log_volume.rolling(window=medium).median().shift(1)
    long_term = log_volume.rolling(window=slow).median().shift(1)

    experts = pd.concat([short_term, medium_term, long_term], axis=1).dropna()
    experts.columns = ["fast", "medium", "slow"]
    log_volume = log_volume.loc[experts.index]
    median_diff = median_diff.loc[experts.index]
    return (
        apple,
        data,
        end_date,
        experts,
        fast,
        log_volume,
        long_term,
        median,
        median_diff,
        medium,
        medium_term,
        price,
        short_term,
        slow,
        start_date,
        volume,
    )


@app.cell
def __(np):
    def rolling_MAE(x, y, window):
        diff = x - y
        return diff.rolling(window=window).apply(lambda x: np.mean(np.abs(x)))

    return (rolling_MAE,)


@app.cell
def __(cp, np):
    def loss(x, x_hat):
        return cp.abs(x - x_hat)

    def solve_comination_problem(
        experts, y, pi_old, decay_weights=None, gamma_smooth=None
    ):
        """
        param experts: n x K matrix of forecasts; one expert for each column
        param X: M x n vector of observed quantities
        param loss: convex loss function
        """
        T, K = experts.shape

        pi = cp.Variable(K, nonneg=True)

        if decay_weights is None:
            decay_weights = np.ones(T) / T

        if gamma_smooth is None:
            objective = cp.Minimize(decay_weights @ loss(experts @ pi, y))
        else:
            objective = cp.Minimize(
                decay_weights @ loss(experts @ pi, y)
                + gamma_smooth * cp.sum_squares(pi - pi_old)
            )

        constraints = [cp.sum(pi) == 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver="CLARABEL")

        return pi.value

    return loss, solve_comination_problem


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Results
        ##### (The results may take a few minutes to compute.)
        """
    )
    return


@app.cell
def __(experts, log_volume, mo, np, pd, solve_comination_problem):
    M = 250
    gamma_smooth = None

    T = len(log_volume)
    halflife = 250
    decay_weights = None

    combination = pd.Series(index=log_volume.index, name=log_volume.name, dtype=float)
    pis = pd.DataFrame(index=log_volume.index, columns=experts.columns)

    K = experts.shape[1]
    pi_old = np.ones(K) / K
    total_iters = T - M
    for t in range(M, T):
        mo.output.replace(f"\rIteration {t-M}/{total_iters}")
        y_t = log_volume.iloc[t - M : t].values.flatten()  # not including t
        experts_t = experts.iloc[t - M : t].values
        pi = solve_comination_problem(
            experts_t, y_t, pi_old, decay_weights, gamma_smooth
        )
        pis.iloc[t] = pi

        combination.iloc[t] = experts.iloc[t] @ pi
    return (
        K,
        M,
        T,
        combination,
        decay_weights,
        experts_t,
        gamma_smooth,
        halflife,
        pi,
        pi_old,
        pis,
        t,
        total_iters,
        y_t,
    )


@app.cell
def __(combination, experts, log_volume, rolling_MAE):
    window_mae = 1

    short_mae = rolling_MAE(experts.iloc[:, 0], log_volume, window_mae)
    medium_mae = rolling_MAE(experts.iloc[:, 1], log_volume, window_mae)
    long_mae = rolling_MAE(experts.iloc[:, 2], log_volume, window_mae)
    combination_mae = rolling_MAE(combination, log_volume, window_mae)
    return combination_mae, long_mae, medium_mae, short_mae, window_mae


@app.cell
def __(combination_mae, long_mae, medium_mae, pis, plt, short_mae):
    plt.rcParams.update({"font.size": 15})
    plt.rcParams["figure.figsize"] = (7.5, 5)

    test_times = combination_mae.dropna().index
    window = 250

    short_mae[test_times].rolling(window).median().plot()
    medium_mae[test_times].rolling(window).median().plot()
    long_mae[test_times].rolling(window).median().plot()
    combination_mae[test_times].rolling(window).median().plot()
    plt.legend(["fast", "medium", "slow", "combination"])
    plt.tight_layout()
    plt.xlabel(None)
    # plt.title("250-day rolling MAE");

    pis.loc[test_times].plot.area(stacked=True)
    plt.tight_layout()
    plt.xlabel(None)
    return test_times, window


@app.cell
def __(combination_mae, long_mae, medium_mae, mo, short_mae):
    with mo.redirect_stdout():
        print(f"Average short-term MAE: {short_mae.median():.2f}")
        print(f"Average medium-term MAE: {medium_mae.median():.2f}")
        print(f"Average long-term MAE: {long_mae.median():.2f}")
        print(f"Average combination MAE: {combination_mae.median():.2f}")
    return


@app.cell
def __(combination_mae, long_mae, medium_mae, mo, short_mae):
    with mo.redirect_stdout():
        print(f"90th percentile short-term MAE: {short_mae.quantile(0.9):.2f}")
        print(f"90th percentile medium-term MAE: {medium_mae.quantile(0.9):.2f}")
        print(f"90th percentile long-term MAE: {long_mae.quantile(0.9):.2f}")
        print(f"90th percentile combination MAE: {combination_mae.quantile(0.9):.2f}")
    return


@app.cell
def __(combination_mae, long_mae, medium_mae, mo, short_mae):
    with mo.redirect_stdout():
        print(f"10th percentile short-term MAE: {short_mae.quantile(0.1):.2f}")
        print(f"10th percentile medium-term MAE: {medium_mae.quantile(0.1):.2f}")
        print(f"10th percentile long-term MAE: {long_mae.quantile(0.1):.2f}")
        print(f"10th percentile combination MAE: {combination_mae.quantile(0.1):.2f}")
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
