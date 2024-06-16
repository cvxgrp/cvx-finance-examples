import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Optimal execution
        This notebook illustrates the optimal purchase
        schedule problem.

        ## Setup

        ### Problem statement
        We
        want to purchase $Q$ shares over $T$ periods, and denote by $t\in\mathbf{R}^T$
        the purchase schedule. ($q\geq 0$ and $\mathbf{1}^Tt=Q$). We model the bid-ask
        midpoint price dynamics as a geometric Brownian motion with zero drift,
        $\textit{i.e.}$,

        $$
        p_t = p_{t-1} + \xi_t, \quad t=2,\ldots,T,
        $$

        with $p_1$ known and $\xi_t$ IID $\mathcal{N}(0,\sigma^2)$. The nominal cost is random variable $p^Tq$, with

        $$
        \mathbf{E} (p^Tq) = p_1 Q, \qquad
        \mathbf{var}(p^T q) = q^T\Sigma q,
        $$

        where $\Sigma_{kl}=\sigma^2 \min(k-1,l-1)$. We define the risk of the purchase
        schedule as $q^T\Sigma q$.

        ### Market impact
        The transaction (or market impact) cost is modeled using the squareroot model

        $$
        \sum_{t=1}^T \sigma \pi^{1/2}_t q_t = \sigma \sum_{t=1}^T
        q_t^{3/2}/ v_t^{1/2}.
        $$

        ### Optimal execution problem
        The optimal execution problem trades off risk and transaction cost by solving
        the following optimization problem

        $$
        \begin{array}{ll}
        \text{minimize} &  \sigma \sum_{t=1}^T \left({q_t^{3/2}/v_t^{1/2}}\right) + \gamma q^T\Sigma q \\
        \text{subject to} & q \geq 0, \quad \mathbf{1}^T q = Q, \quad q_t/v_t \leq
        \pi^\mathrm{max}, \quad t=1, \ldots, T,
        \end{array}
        $$

        where $\gamma$ is a risk aversion parameter and $\pi^\mathrm{max}$ is a
        participation rate limit.
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (7.5, 5)
    return cp, np, plt


@app.cell
def __(Q_slider, gamma_slider, np, pi_max_slider):
    T = 10
    Q = Q_slider.value
    pi_max = pi_max_slider.value
    gamma = gamma_slider.value
    p1 = 188
    sigma = 2.1
    v = np.array([41, 45, 42, 57, 55, 65, 50, 54, 42, 52])
    return Q, T, gamma, p1, pi_max, sigma, v


@app.cell
def __(Q_slider, mo):
    mo.md(
        rf"""
        ## Example

        We consider an example in which we want to purchase $Q={Q_slider.value}$ million shares over
        $T=10$ trading days (Feb 8--22, 2024). We have a participation rate limit of
        5%. The data was scraped from Yahoo Finance using the following script
        ```python
        import yfinance as yf

        start_date = "2024-02-08"
        end_date = "2024-02-22"

        apple = yf.Ticker("AAPL")
        data = apple.history(period="max")
        prices = data["Close"]

        p1 = prices.loc[start_date].round().astype(int)
        sigma = np.round(prices.loc[:start_date][-63:].diff().std(), 1)
        v = (np.round(data["Volume"].loc[:end_date] / 1e6).astype(int)).loc[start_date:end_date]
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"### Results")
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
def __(Q, T, cp, gamma, mo, np, pi_max, sigma, v):
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
    print(f"Cost of optimal schedule: {prob.solve()}")

    if q.value is None:
        with mo.redirect_stdout():
            print(f"Problem status: {prob.status}")
    return Sigma, cons, obj, pi, prob, q, risk, transaction_cost


@app.cell(hide_code=True)
def __(Q, T, mo, np, p1, plt, risk, transaction_cost):
    periods = np.arange(1, T + 1)
    plt.rcParams["figure.dpi"] = 100  # Resolution in dots per inch

    with mo.redirect_stdout():
        print(f"Volatility: {np.sqrt(risk.value):.1f}")
        print(f"Transaction cost: {transaction_cost.value:.1f}")
        print(f"\nNominal volatility: {np.sqrt(risk.value)/(Q*p1)/(0.01**2):.1f} bps")
        print(
            f"Nominal transaction cost: {transaction_cost.value/(Q*p1)/(0.01**2):.1f} bps"
        )
    return (periods,)


@app.cell
def __(periods, plt, v):
    plt.figure(figsize=(7.5, 5))
    plt.plot(periods, v, marker="o")
    plt.xlabel("Period")
    plt.ylabel("Volume (millions of shares)")
    plt.ylim(0, 70)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def __(periods, plt, q):
    plt.figure(figsize=(7.5, 5))
    plt.plot(periods, q.value, marker="o")
    plt.xlabel("Period")
    plt.ylabel(r"$q_t$ (millions of shares)")
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def __(periods, pi, pi_max, plt):
    plt.figure(figsize=(7.5, 5))
    plt.plot(periods, pi.value, marker="o")
    plt.plot(periods, [pi_max] * len(periods), "k--", label=r"$\pi_{\max}$")
    plt.xlabel("Period")
    plt.ylabel("Market participation")
    plt.ylim(-0.01, 0.06)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
