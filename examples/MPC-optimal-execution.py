import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Model predictive control: optimal execution

        This notebook considers the same problem as in the `optimal-execution`
        notebook, with one difference: before we assumed the trade volumes and price volatilities were known;
        now we forecast the volumes and volatilities, and update our forecasts each period
        $t=1,\ldots,T$. This gives rise to a model predictive control (MPC) problem.
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import cvxpy as cp
    import yfinance as yf
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (7.5, 5)
    return cp, np, plt, yf


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Example

        We consider the same example as in the `optimal-execution` notebook. At each
        time $t=1,\ldots,T$ ($T=10$ trading days) we forecast the volumes $v_{\tau}$,
        $\tau=t,\ldots,T$ of the upcoming trades as the 5-day moving median of observed
        trades. Similarly, we forecast the price volatility as the 21-day moving standard
        deviation of observed price differences.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"### Optimal execution schedules")
    return


@app.cell
def __(mo):
    Q_slider = mo.ui.slider(1, 51, label=r"$Q$", value=10)
    Q_slider
    return (Q_slider,)


@app.cell
def __(mo):
    pi_max_slider = mo.ui.slider(
        0.01, 0.5, step=0.01, label=r"$\pi^{\max}$", value=0.05
    )
    pi_max_slider
    return (pi_max_slider,)


@app.cell
def __(mo):
    gamma_slider = mo.ui.slider(
        steps=[5 * 10**i for i in range(-5, 6)], value=0.05, label=r"$\gamma$"
    )
    gamma_slider
    return (gamma_slider,)


@app.cell
def __(Q_slider, gamma_slider, np, pi_max_slider, yf):
    T = 10
    Q = Q_slider.value
    pi_max = pi_max_slider.value
    gamma = gamma_slider.value
    memory_volume = 5
    memory_sigma = 21

    start_date = "2024-02-08"
    end_date = "2024-02-22"

    apple = yf.Ticker("AAPL")
    data = apple.history(period="max")
    prices = data["Close"]
    volumes = np.round(data["Volume"].loc[:end_date] / 1e6).astype(int)

    # Predictions
    v_hat = volumes.rolling(memory_volume).median().shift(1).loc[start_date:end_date]
    sigma_hat = (
        prices.diff().rolling(memory_sigma).std().shift(1).loc[start_date:end_date]
    )
    sigma = np.round(prices.loc[:start_date][-63:].diff().std(), 1)  # 'true' sigma
    prices = prices.loc[start_date:end_date]
    volumes = volumes.loc[start_date:end_date]
    return (
        Q,
        T,
        apple,
        data,
        end_date,
        gamma,
        memory_sigma,
        memory_volume,
        pi_max,
        prices,
        sigma,
        sigma_hat,
        start_date,
        v_hat,
        volumes,
    )


@app.cell
def __(cp, np):
    def get_execution(Q, T, p1, sigma, pi_max, gamma, v):
        # price covariance matrix
        Sigma = sigma**2 * np.array(
            [list(range(i + 1)) + [i] * (T - i - 1) for i in range(T)]
        )

        # solve optimal purchase execution problem in CVXPY
        q = cp.Variable(T)

        risk = cp.quad_form(q, Sigma)
        transaction_cost = sigma * cp.power(v, -1 / 2) @ cp.power(q, 3 / 2)
        obj = cp.Minimize(transaction_cost + gamma * risk)

        pi = q / v
        cons = [q >= 0, cp.sum(q) == Q, pi <= pi_max]

        prob = cp.Problem(obj, cons)
        prob.solve(solver="CLARABEL")
        return q.value, prob

    return (get_execution,)


@app.cell
def __(Q, T, gamma, get_execution, mo, pi_max, prices, sigma_hat, v_hat):
    v_fixed = [v_hat.iloc[0]] * T
    sigma_fixed = sigma_hat.iloc[0]
    q_fixed, prob = get_execution(
        Q, T, prices.iloc[0], sigma_fixed, pi_max, gamma, v_fixed
    )

    if q_fixed is None:
        with mo.redirect_stdout():
            print(f"Problem status: {prob.status}")
    return prob, q_fixed, sigma_fixed, v_fixed


@app.cell
def __(Q, T, gamma, get_execution, pi_max, prices, sigma_hat, v_hat):
    q_mpc = []

    plans = {}

    for j in range(T):
        _T = T - j
        _Q = Q - sum(q_mpc)
        _sigma = sigma_hat.iloc[j]
        _v = [v_hat.iloc[j]] * _T
        _q_mpc, _ = get_execution(_Q, _T, prices.iloc[j], _sigma, pi_max, gamma, _v)
        q_mpc.append(_q_mpc[0])

        plans[j] = _q_mpc
    return j, plans, q_mpc


@app.cell
def __(np):
    def cost_of_schedule(q, p, v, sigma):
        purchase_cost = (q * p).sum()
        market_cost = sigma * np.power(v, -1 / 2) @ np.power(q, 3 / 2)

        return purchase_cost + market_cost

    return (cost_of_schedule,)


@app.cell
def __(plt):
    def plot_plans_and_executions(plans, t, q_fixed, q_mpc):
        fig, ax = plt.subplots()
        ax.plot(range(t), [q_fixed[0]] * t, label="Fixed Plan")
        ax.plot(range(t), q_mpc, label="MPC Plan")
        ax.plot(range(t), [plans[i][0] for i in range(t)], label="MPC Executions")
        ax.legend()
        plt.show()

    return (plot_plans_and_executions,)


@app.cell
def __(cost_of_schedule, mo, prices, q_fixed, q_mpc, sigma, volumes):
    fixed_cost = cost_of_schedule(q_fixed, prices.values, volumes, sigma)

    mpc_cost = cost_of_schedule(q_mpc, prices.values, volumes, sigma)

    with mo.redirect_stdout():
        print(f"Cost of fixed schedule: {fixed_cost:.0f} million USD")
        print(f"Cost of MPC schedule: {mpc_cost:.0f} million USD")
        print(
            f"Relative improvement: {(fixed_cost - mpc_cost) / fixed_cost / 0.01**2 :.0f} bps"
        )
    return fixed_cost, mpc_cost


@app.cell
def __(T, np, plt, q_fixed, q_mpc):
    # plot plans

    periods = np.arange(1, T + 1)
    plt.plot(periods, q_fixed, label="fixed schedule", marker="o")
    plt.plot(periods, q_mpc, label="MPC schedule", marker="o")
    plt.ylabel("Quantity")
    plt.legend()
    return (periods,)


@app.cell
def __(periods, plans, plt, q_mpc):
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)

    for t in range(8):
        ax = axs[t // 4, t % 4]
        ax.plot(periods[t:], plans[t], label="plan", marker="o", zorder=0)
        ax.plot(
            periods[: t + 1], q_mpc[: t + 1], label="executed", marker="X", zorder=1
        )
        ax.set_title(f"t={t+1}")
        ax.axvline(t + 1, color="black", linestyle="--", zorder=-1, alpha=0.75)

        if t % 4 == 0:
            ax.set_ylabel("Quantity")
        if t == 0:
            ax.legend()

        ax.set_xticks(periods[1::2])
        ax.set_xticklabels([int(x) for x in periods[1::2]])

    fig
    return ax, axs, fig, t


@app.cell
def __(periods, plt, v_hat):
    # plot volume predictions

    plt.plot(periods, v_hat, marker="o")
    plt.ylabel("Predicted volume")
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
