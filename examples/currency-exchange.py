import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Currency exchange

        This notebook illustrates an example of rebalancing a currency portfolio.
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

    return cp, np, pd, plt, yf


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Setup

        ### Problem statement
        We hold $c^{\text{init}}=(c^{\text{init}}_1,\ldots,c^{\text{init}}_n)$ of
        $n$ currencies, in USD under nominal exchange rates. In words,
        $c^{\text{init}}_j = x$ means we hold $x$ USD in currency $j$. We want to exchange
        these currencies to obtain (at least)
        $c^{\text{req}}=(c^{\text{req}}_1,\ldots,c^{\text{req}}_n)$, valued in USD. Let $X
        \in \reals^{n\times n}$ be the currency exchange matrix, $\textit{i.e.}$,
        $X_{ij}\geq 0$ is the amount
        of $j$ we exchange for $i$, in USD. Moreover, let $\Delta_{ij} \geq 0$ be the cost of exchanging one USD of
        currency $j$ for currency $i$, expressed as a fraction. This means that the
        exchange $X_{ij}$ costs us $X_{ij}\Delta_{ij}$ USD.

        We find the optimal currency exchange, $\textit{i.e.}$, the one that minimizes the
        cost, by solving

        $$
        \begin{array}{ll}
        \text{minimize} & \sum_{i,j=1}^n X_{ij}\Delta_{ij} \\
        \text{subject to} & X_{ij} \geq 0, \quad \mathbf{diag}(X)=0,\\
        & c_i^\text{init} + \sum_j X_{ij} - \sum_j X_{ji} \geq
        c^{\text{req}}_i, \quad i=1, \ldots, n.
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

        Consider four currencies, USD, EUR, CAD, and SEK, with
        initial and required holdings (in $\$ 10^6$)

        $$
        c^{\text{init}}=(1,1,1,1), \qquad
        c^{\text{req}}=(2.1, 1.5, 0.3, 0.1).
        $$

        ### Exchange rates
        The exchange rates ($\Delta$) in basis points (bps) are given by the matrix

        |        | USD | EUR | CAD | SEK |
        |--------|-----|-----|-----|-----|
        | **USD**| 0.0 | 0.1 | 4.4 | 4.8 |
        | **EUR**| 0.1 | 0.0 | 5.0 | 5.7 |
        | **CAD**| 2.8 | 6.9 | 0.0 | 8.5 |
        | **SEK**| 1.1 | 7.9 | 7.6 | 0.0 |

        Roughly speaking it is cheap to trade USD and EUR, and expensive to trade CAD and SEK.
        """
    )
    return


@app.cell
def __(np, pd):
    np.random.seed(0)
    n = 4

    # Initial and required holdings
    c_init = np.array([1] * n)
    c_req = np.array([2.1, 1.5, 0.2, 0.1])

    bps_5 = 0.01**2 * 5

    currencies = ["USD", "EUR", "CAD", "SEK"]
    Delta = pd.DataFrame(np.ones((n, n)) * bps_5)

    Delta.index = currencies
    Delta.columns = currencies

    # USD and EUR are cheap
    Delta.loc["USD", :] -= np.random.uniform(0, bps_5, 4)
    Delta.loc[:, "USD"] -= np.random.uniform(0, bps_5, 4)
    Delta.loc["EUR", :] -= np.random.uniform(0, 0.5 * bps_5, 4)
    Delta.loc[:, "EUR"] -= np.random.uniform(0, 0.5 * bps_5, 4)

    # CAD and SEK are expensive
    Delta.loc["CAD", :] += np.random.uniform(0, 0.5 * bps_5, 4)
    Delta.loc[:, "CAD"] += np.random.uniform(0, 0.5 * bps_5, 4)
    Delta.loc["SEK", :] += np.random.uniform(0, bps_5, 4)
    Delta.loc[:, "SEK"] += np.random.uniform(0, bps_5, 4)

    Delta = (Delta * 100**2).clip(0.1)  # convert to bps
    np.fill_diagonal(Delta.values, 0)
    return Delta, bps_5, c_init, c_req, currencies, n


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"## Results")
    return


@app.cell
def __(Delta, c_init, c_req, cp, currencies, mo, n, pd):
    X = cp.Variable((n, n), nonneg=True)

    objective = cp.sum(cp.multiply(X, Delta.values))
    constraints = [
        cp.diag(X) == 0,
        c_init + cp.sum(X, axis=1) - cp.sum(X, axis=0) >= c_req,
    ]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver="CLARABEL")

    X_df = pd.DataFrame(X.value)
    X_df.index = currencies
    X_df.columns = currencies

    # compute cost of greedy policy
    X_greedy = X_df.copy() * 0
    X_greedy.loc["USD", "CAD"] = 0.7
    X_greedy.loc["USD", "SEK"] = 0.4
    X_greedy.loc["EUR", "SEK"] = 0.5

    with mo.redirect_stdout():
        print(f"Optimal cost: {objective.value / c_init.sum():.2f} bps")
        print(
            f"Greedy cost: {(X_greedy.values * Delta.values).sum() / c_init.sum():.2f} bps"
        )
        print(
            f"\nFinal holdings {(c_init + cp.sum(X, axis=1) - cp.sum(X, axis=0)).value}"
        )
        print(f"\nOptimal exchange matrix\n{X_df.round(5)}")
    return X, X_df, X_greedy, constraints, objective, prob


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
