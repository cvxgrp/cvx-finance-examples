import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Maximum expected utility portfolio construction
        This notebook illustrates portfolio construction using expected utility maximization.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Setup
            We consider asset returns $r \in \reals^n$ and portfolio weights $w\in \reals^n$.
        The portfolio return is $r^Tw$, and so wealth grows by the factor $1+r^Tw$. The
        expected utility is $\mathbf{E} U(1+r^Tw)$, where $U$ is a concave increasing
        utility function. We want to choose portfolio weights $w \in \mathcal W$
        (a convex set) to maximize expected utility. Since, in general, $\mathbf{E} U(1+r^Tw)$ can't be expressed analytically,
        we use a sample based approximation as follows. We generate $N$ samples $r_1, \ldots, r_N$, with
        probabilities $\pi_1, \ldots, \pi_N$, and approximate expected utility as $\mathbf{E} U(1+r^Tw)
        \approx \sum_{i=1}^N \pi_i U(1+r_i^Tw)$. The sample based approximate expected
        utility maximization problem is then

        $$
        \begin{array}{ll}
            \text{maximize} & \sum_{i=1}^N \pi_i U(1+r_i^Tw) \\
            \text{subject to} & w \in \mathcal{W}
        \end{array}
        $$
        """
    )
    return


@app.cell
def __(N_slider, mo):
    mo.md(
        rf"""
        ### Problem statement
        In this example we optimize a portfolio of one underlying, one call, and one put,
        both at-the-money; the underlying has $1+r$ log-normal. We use a CRRA utility with relative risk aversion $\rho$,
        $\mathcal{{W}} = \{{w \mid \mathbf{{1}}^Tw=1\}}$, and a sample approximation with $N={N_slider.value}$ samples.
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import cvxpy as cp
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 18})
    plt.rcParams["figure.figsize"] = (7.5, 5)
    return cp, np, pd, plt


@app.cell
def __(cp, np):
    def u(x, rho):
        """
        Utility function"""
        if rho == 1:
            return cp.log(x)
        else:
            return (x ** (1 - rho) - 1) / (1 - rho)

    def call_payoff(S, K):
        return np.maximum(S - K, 0)

    def put_payoff(S, K):
        return np.maximum(K - S, 0)
    return call_payoff, put_payoff, u


@app.cell
def __(mo):
    N_slider = mo.ui.slider(
        steps=[10**i for i in range(2, 8)],
        value=10**4,
        label=r"$N$",
        show_value=True,
    )

    rho_risky = mo.ui.slider(
        start=0, stop=3, value=2, label=r"$\rho^{\text{risky}}$", show_value=True
    )
    rho_conservative = mo.ui.slider(
        start=3, stop=6, value=4, show_value=True, label=r"$\rho^{\text{conservative}}$"
    )

    mo.vstack([N_slider, rho_risky, rho_conservative])
    return N_slider, rho_conservative, rho_risky


@app.cell
def __(cp, np, u):
    def solve_utility_maximization(return_outcomes, probabilities, rho):
        """
        Solve the utility maximization problem

        param return_outcomes: a numpy array of shape (N, n) where n is the number
        of assets and N is the number of scenarios
        param probabilities: a numpy array of shape (N,) representing the
        probability of each scenario; the probabilities sum to 1
        param rho: the coefficient of relative risk aversion
        """
        assert (
            return_outcomes.shape[0] == probabilities.shape[0]
        ), "return_outcomes and probabilities must have the same number of scenarios"
        assert np.isclose(probabilities.sum(), 1), "probabilities must sum to 1"

        n = return_outcomes.shape[1]
        w = cp.Variable(n)

        objective = probabilities @ u(1 + (return_outcomes @ w), rho)

        constraints = [cp.sum(w) == 1]

        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        return w.value
    return solve_utility_maximization,


@app.cell
def __(N_slider, call_payoff, np, put_payoff):
    np.random.seed(0)
    N = N_slider.value
    mean = 0.05
    stddev = 0.1
    S0 = 100  # price of underlying asset today
    K = 100  # strike price

    payoff_outcomes_underlying = S0 * np.random.lognormal(mean, stddev, (N, 1))
    payoff_outcomes_underlying.sort(axis=0)
    payoff_outcomes_call = call_payoff(payoff_outcomes_underlying, K)
    payoff_outcomes_put = put_payoff(payoff_outcomes_underlying, K)

    call_premium = payoff_outcomes_call.mean() * 0.75
    put_premium = payoff_outcomes_put.mean() * 1.4

    payoff_outcomes = np.hstack(
        [payoff_outcomes_underlying, payoff_outcomes_call, payoff_outcomes_put]
    )
    return (
        K,
        N,
        S0,
        call_premium,
        mean,
        payoff_outcomes,
        payoff_outcomes_call,
        payoff_outcomes_put,
        payoff_outcomes_underlying,
        put_premium,
        stddev,
    )


@app.cell
def __(
    S0,
    call_premium,
    np,
    payoff_outcomes_call,
    payoff_outcomes_put,
    payoff_outcomes_underlying,
    put_premium,
):
    return_outcomes_call = payoff_outcomes_call / call_premium - 1
    return_outcomes_put = payoff_outcomes_put / put_premium - 1
    return_outcomes_underlying = payoff_outcomes_underlying / S0 - 1

    return_outcomes = np.hstack(
        [return_outcomes_underlying, return_outcomes_call, return_outcomes_put]
    )
    return (
        return_outcomes,
        return_outcomes_call,
        return_outcomes_put,
        return_outcomes_underlying,
    )


@app.cell
def __(
    N,
    np,
    return_outcomes,
    rho_conservative,
    rho_risky,
    solve_utility_maximization,
):
    probabilities = np.ones(N) / N
    w_conservative = solve_utility_maximization(
        return_outcomes, probabilities, rho_conservative.value
    )
    return_distribution_conservative = return_outcomes @ w_conservative

    w_risky = solve_utility_maximization(
        return_outcomes, probabilities, rho_risky.value
    )
    return_distribution_risky = return_outcomes @ w_risky
    return (
        probabilities,
        return_distribution_conservative,
        return_distribution_risky,
        w_conservative,
        w_risky,
    )


@app.cell
def __(
    return_distribution_conservative,
    return_distribution_risky,
    return_outcomes,
    rho_conservative,
    rho_risky,
    u,
    w_conservative,
    w_risky,
):
    # Evaluate the utility of the two portfolios
    utility_conservative = u(
        1 + return_distribution_conservative, rho_conservative.value
    ).mean()
    utility_risky = u(1 + return_distribution_risky, rho_risky.value).mean()

    utility_conservative_pf_on_risky_utility = u(
        1 + (return_outcomes @ w_conservative), 2
    ).mean()
    utility_risky_pf_on_conservative_utility = u(
        1 + (return_outcomes @ w_risky), 4
    ).mean()
    return (
        utility_conservative,
        utility_conservative_pf_on_risky_utility,
        utility_risky,
        utility_risky_pf_on_conservative_utility,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        (Results may take a few seconds, or even minutes, to compute when $N$ is large.)
        """
    )
    return


@app.cell
def __(
    np,
    payoff_outcomes_call,
    payoff_outcomes_put,
    payoff_outcomes_underlying,
    plt,
    return_outcomes,
    return_outcomes_call,
    return_outcomes_put,
    return_outcomes_underlying,
    rho_conservative,
    rho_risky,
    w_conservative,
    w_risky,
):
    # Plotting
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, figsize=(8, 15))

    # Plot payoffs
    _ax1.plot(
        payoff_outcomes_underlying,
        payoff_outcomes_underlying,
        label="Underlying",
        zorder=1,
        linewidth=3,
    )
    _ax1.plot(payoff_outcomes_underlying, payoff_outcomes_call, label="Call", zorder=1)
    _ax1.plot(payoff_outcomes_underlying, payoff_outcomes_put, label="Put", zorder=1)
    _ax1.set_xlabel("Underlying")
    _ax1.set_ylabel("Payoff")
    _ax1.legend()
    _ax1.set_title("Payoffs")

    # Plot returns
    _ax2.plot(
        return_outcomes_underlying, return_outcomes_underlying, label="Underlying"
    )
    _ax2.plot(return_outcomes_underlying, return_outcomes_call, label="Call")
    _ax2.plot(return_outcomes_underlying, return_outcomes_put, label="Put")

    _ax2_twin = _ax2.twinx()
    _ax2_twin.set_ylabel("Count")
    _ax2_twin.hist(
        return_outcomes_underlying,
        bins=100,
        alpha=0.5,
        color="gray",
        label="Underlying\ndistribution",
    )

    _ax2.set_xlabel("Underlying return")
    _ax2.set_ylabel("Asset return")

    _lines, _labels = _ax2.get_legend_handles_labels()
    _lines2, _labels2 = _ax2_twin.get_legend_handles_labels()
    _ax2_twin.legend(_lines + _lines2, _labels + _labels2, loc="upper left")
    _ax2.set_xlim(-0.3, 0.3)
    _ax2.set_title("Returns")

    # Replace bar chart with histograms
    _return_distribution_conservative = return_outcomes @ w_conservative
    _return_distribution_risky = return_outcomes @ w_risky

    _bins = np.histogram(
        np.hstack((_return_distribution_conservative, _return_distribution_risky)),
        bins=100,
    )[1]

    _ax3.hist(
        _return_distribution_conservative,
        bins=_bins,
        alpha=0.5,
        label=rf"$\rho={rho_conservative.value}$",
    )
    _ax3.hist(
        _return_distribution_risky,
        bins=_bins,
        alpha=0.5,
        label=rf"$\rho={rho_risky.value}$",
    )

    # Add vertical lines for the expected return
    _ax3.axvline(
        _return_distribution_conservative.mean(),
        color="blue",
        linestyle="dashed",
        linewidth=2,
        alpha=0.8,
    )
    _ax3.axvline(
        _return_distribution_risky.mean(),
        color="orange",
        linestyle="dashed",
        linewidth=2,
        alpha=0.8,
    )

    _ax3.set_xlabel("Portfolio return")
    _ax3.set_ylabel("Count")
    _ax3.set_xlim(-0.3, 0.3)

    # Add to legend that dashed lines are mean returns
    _ax3.legend(
        [
            rf"$\rho={rho_conservative.value}$",
            rf"$\rho={rho_risky.value}$",
            rf"mean $\rho={rho_conservative.value}$",
            rf"mean $\rho={rho_risky.value}$",
        ]
    )
    _ax3.set_title("Utility Evaluation")

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
