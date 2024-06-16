import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Option price bounds

        This notebook considers the problem of finding arbitrate-free price bounds on a
        collar option. Arbitrage-free pricing is based on a set of strict alternatives,
        originating from Farkas' lemma.
        """
    )
    return


@app.cell
def __():
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt

    return cp, np, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Setup

        ### Problem statement
        Consider $n$ assets with prices $p_1,\ldots,p_n$. At the end of the investment
        period there are $m$ possible outcomes, and $V_{ij}$ denotes the payoff of asset
        $j$ in outcome $i$. The first investment is risk-free, and so $V_{i1}=1$ for all
        $i$. Suppose the current prices $p_1,\ldots,p_{n-1}$ of the first $n-1$ assets are known, but the price
        $p_n$ of the $n$-th asset is unknown. The goal is to find the arbitrage-free
        price bounds on the $n$-th asset. The arbitrage-free price range is found be
        solving

        $$
        \begin{array}{ll}
        \text{minimize}/\text{maximize} & p_n \\
        \text{subject to} & V^T \pi = p, \quad \pi \geq 0, \quad \mathbf{1}^T \pi=1
        \end{array}
        $$

        with variables $p_n\in\reals$ and $\pi\in\reals^m$.
        """
    )
    return


@app.cell
def __(cap_slider, floor_slider, mo):
    mo.md(
        rf"""
        ## Example

        Consider $n=7$ assets: one risk-free asset with price 1 and payoff 1; an
        underlying asset with price 1 and uncertain payoff; four vanilla options with
        known market prices
        <table>
            <tr>
                <td>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Strike</th>
                            <th>Price</th>
                        </tr>
                        <tr>
                            <td>Call</td>
                            <td>1.1</td>
                            <td>0.06</td>
                        </tr>
                        <tr>
                            <td>Call</td>
                            <td>1.2</td>
                            <td>0.03</td>
                        </tr>
                        <tr>
                            <td>Put</td>
                            <td>0.8</td>
                            <td>0.02</td>
                        </tr>
                        <tr>
                            <td>Put</td>
                            <td>0.7</td>
                            <td>0.01</td>
                        </tr>
                    </table>
                    <p><strong>Option types and prices</strong></p>
                </td>
            </tr>
        </table>

        There are $m=200$ possible outcomes for the underlying asset,
        uniformly between 0.5 and 2. We seek price bounds on a collar option with floor {floor_slider.value} and cap {cap_slider.value}.

        The payoff of the collar option is

        $$
        \min(C,\max(F,S))-S,
        $$

        where $C$ is the cap, $F$ is the floor, and $S$ is the underlying asset price.
        """
    )

    return


@app.cell
def __(mo):
    floor_slider = mo.ui.slider(
        0.1, 1, 0.05, value=0.9, show_value=True, label=r"floor $(F)$"
    )
    cap_slider = mo.ui.slider(
        1.1, 2, 0.05, value=1.15, show_value=True, label=r"cap $(C)$"
    )
    mo.vstack([floor_slider, cap_slider])
    return cap_slider, floor_slider


@app.cell
def __(mo):
    mo.md(r"### Data")
    return


@app.cell
def __(cap_slider, floor_slider, np):
    # data
    m = 200
    risk_free = 1.0
    p_given = np.array([1.0, 1.0, 0.06, 0.03, 0.02, 0.01])
    F = floor_slider.value
    C = cap_slider.value

    # Set up the payoff matrix
    S = np.linspace(0.5, 2, m)
    P1 = np.maximum(0, S - 1.1)
    P2 = np.maximum(0, S - 1.2)
    C1 = np.maximum(0, 0.8 - S)
    C2 = np.maximum(0, 0.7 - S)
    Collar = np.clip(S, a_min=F, a_max=C) - 1
    return C, C1, C2, Collar, F, P1, P2, S, m, p_given, risk_free


@app.cell
def __(C1, C2, Collar, P1, P2, S, plt):
    plt.rcParams.update({"font.size": 17.5})
    plt.rcParams["figure.figsize"] = (7.5, 5)
    plt.rcParams["lines.linewidth"] = 2.5

    plt.plot(S, P1, label="put 1")
    plt.plot(S, P2, label="put 2")
    plt.plot(S, C1, label="call 1")
    plt.plot(S, C2, label="call 2")
    plt.plot(S, Collar, label="collar")
    plt.xlabel("stock Price")
    plt.ylabel("payoff")
    plt.legend()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"### Results")
    return


@app.cell
def __(C1, C2, Collar, P1, P2, S, cp, m, mo, np, p_given, risk_free):
    V = np.vstack((risk_free * np.ones(m), S, P1, P2, C1, C2, Collar)).T

    # Define the variables
    p = cp.Variable(7)
    y = cp.Variable(m)

    # The first six prices should match the given prices, and we use Farkas' lemma
    # to ensure no-arbitrage
    constraints = [p[:6] == p_given, V.T @ y == p, y >= 0]

    # Set up and solve the problems
    prob_min = cp.Problem(cp.Minimize(p[-1]), constraints)
    prob_max = cp.Problem(cp.Maximize(p[-1]), constraints)
    p_min = prob_min.solve(solver="CLARABEL")
    p_max = prob_max.solve(solver="CLARABEL")

    with mo.redirect_stdout():
        print(f"Lower Bound: {p_min:.3f}")
        print(f"Upper Bound: {p_max:.3f}")
    return V, constraints, p, p_max, p_min, prob_max, prob_min, y


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
