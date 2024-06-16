import marimo

__generated_with = "0.6.13"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Bond pricing

        This notebook walks through an example of bond pricing. We consider bonds,
        paying the holder an amount $c_t$ in periods $t=1,\ldots,T$. The price of the
        bond is

        $$
        p = \sum_{t=1}^T c_{t}\exp(-t(y_t + s)),
        $$

        where $y=(y_1, \ldots, y_T) \in \reals^T$ is the yield curve, and
        $s \geq 0$ the spread. We will assume $y=Ya$ where $Y$ are basis functions and
        $a$ are coefficients. For example, we can use principal component analysis to
        fit $Y$ to historical data. (Often the first three principal components are
        used to explain the yield curve.)
        """
    )
    return


@app.cell
def __():
    import cvxpy as cp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from nelson_siegel_svensson.calibrate import calibrate_ns_ols

    plt.rcParams.update({"font.size": 15})
    plt.rcParams["figure.figsize"] = (7.5, 5)
    return calibrate_ns_ols, cp, np, pd, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Setup

        ### Problem statement

        We are given market prices $p_i$ and
        ratings $r_i \in \{1,\ldots, K\}$ of $n$ bonds, $i=1,\ldots, n$, and the
        objective is to fit a yield curve $y\in \reals^T$ and spreads
        $s\in \reals^K$ to this data. We add constraint $0 \leq s_1 \leq \cdots \leq
        s_K$, to ensure that higher rated bonds have lower spreads. Using square error
        loss we fit $y$ and $s$ by solving the optimization problem

        $$
        \begin{array}{ll}
        \text{minimize} & \sum_{i=1}^n \left( p_i - \sum_{t=1}^T c_{i,t}
        \exp(-t(y_t + s_{r_i})) \right)^2\\
        \text{subject to} &
        0 \leq s_1 \leq \cdots \leq s_K
        \end{array}
        $$

        with variables $y$ and $s$. Since this is not a convex problem, we solve is
        sequentially by linearizing the exponential term and iteratively fit yields and
        spreads. This yield an sequence of convex problems, which we solve using.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Example
        We consider $n=1000$ simulated bonds, with a maturity of up to 30 years. We rate
        the bonds AAA, AA, A, BBB, or BB. We then use yield data from 1990 to 2024 to
        fit basis functions, and use the latest yields and spread to price the bonds.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"### Bond data")
    return


@app.cell
def __(np, plt):
    # Create 100 synthetic bonds:
    # 1. maturity between 0.5 and 30
    # 2. coupon rate between 0 and 10%, including true 0s
    # 3. coupon frequency between 1 and 4
    # 4. face value between 0 and 100

    bonds = []
    rng1 = np.random.default_rng(0)
    for _ in range(1000):
        cf = np.zeros(60)
        maturity = rng1.uniform(0.5, 30)
        cf[int(maturity * 2)] = 100
        coupon_rate = np.clip(rng1.uniform(-0.02, 0.02, 1000), 0, None)
        coupon_freq = rng1.integers(1, 5)

        for j in range(int(maturity * 2)):
            if j % coupon_freq == 0:
                cf[j] = coupon_rate[j] * 100

        cf[int(maturity * 2)] += 100
        bonds.append(cf)

    C = np.array(bonds)

    ratings = np.repeat([0, 1, 2, 3, 4], 200)

    s_nominal_unique = np.array([3.01e-03, 4.15e-03, 5.9e-03, 9.1e-03, 1.4e-02])
    s_nominal = np.repeat(s_nominal_unique, 200)

    plt.plot(s_nominal, label="spreads for 1000 bonds")

    plt.legend()
    plt.tight_layout()

    plt.gcf()
    return (
        C,
        bonds,
        cf,
        coupon_freq,
        coupon_rate,
        j,
        maturity,
        ratings,
        rng1,
        s_nominal,
        s_nominal_unique,
    )


@app.cell
def __(pd):
    yields_dicrete = pd.read_csv("data/daily-treasury-rates.csv", index_col=0) / 100
    yields_dicrete.dropna(axis=0, inplace=True)
    yields_dicrete.columns = [
        1 / 12,
        2 / 12,
        3 / 12,
        4 / 12,
        6 / 12,
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
    ]
    return (yields_dicrete,)


@app.cell
def __(calibrate_ns_ols, np, yields_dicrete):
    # fit the "continuous", "true" yield curves using the Nelson-Siegel model
    res = []
    for l in range(yields_dicrete.shape[0]):
        try:
            res.append(
                calibrate_ns_ols(
                    np.array(yields_dicrete.columns), yields_dicrete.values[l]
                )
            )

        except Exception:
            print(f"Failed at {l}")
            pass

    res = [r for (r, s) in res if s.success == True]
    return l, res


@app.cell(hide_code=True)
def __(np, pd, plt, res):
    yields = pd.DataFrame(
        [r(np.linspace(0.5, 30, 60)) for r in res], columns=np.linspace(0.5, 30, 60)
    )
    y_nominal = yields.values[0]
    plt.plot(y_nominal, label="nominal yield")
    plt.legend()
    plt.tight_layout()
    plt.gcf()
    return y_nominal, yields


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"### Principal component yield curve basis")
    return


@app.cell
def __(np, plt, yields):
    # PCA with 3 factors using SVD
    U, D, V = np.linalg.svd(yields, full_matrices=False)
    U = U[:, :3]
    D = np.diag(D[:3])
    V = V[:3, :]
    yields_pca = np.dot(U, np.dot(D, V))
    Y = V.T

    plt.plot(Y)
    plt.legend([f"Y_{j}" for j in range(3)])
    plt.tight_layout()
    plt.gcf()
    return D, U, V, Y, yields_pca


@app.cell
def __(C, np, s_nominal, y_nominal):
    rng2 = np.random.default_rng(1)
    p = []
    for i in range(C.shape[0]):
        p.append(
            np.sum(
                [
                    C[i, t] * np.exp(-((t + 1) / 2) * (y_nominal[t] + s_nominal[i]))
                    for t in range(C.shape[1])
                ]
            )
            + rng2.normal(0, 1)
        )
    p = np.array(p)
    return i, p, rng2


@app.cell
def __(C, U, Y, cp, np, p, ratings, s_nominal_unique):
    t = np.linspace(0.5, 30, 60).reshape(1, -1)

    a_init = U[0]

    y = cp.Variable(60, value=Y @ a_init)
    a = cp.Variable(3, value=a_init)
    s = cp.Variable(5, value=np.mean(s_nominal_unique) * np.ones(5))

    solutions = []

    for _ in range(2):
        discount = cp.exp(
            cp.multiply(-t, y.reshape((1, -1)) + s[ratings].reshape((-1, 1)))
        )
        p_current = cp.sum(cp.multiply(C, discount), axis=1)

        Delta_hat = p_current.grad[y].T @ (y - y.value) + p_current.grad[s].T @ (
            s - s.value
        )
        objective = cp.norm2(p - (p_current.value + Delta_hat))
        constraints = [cp.diff(s) >= 0, y == Y @ a]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        print(f"\nStatue: {problem.status}")
        print(f"Objective value: {problem.value}")

        solutions.append((y.value, s.value, a.value))
    return (
        Delta_hat,
        a,
        a_init,
        constraints,
        discount,
        objective,
        p_current,
        problem,
        s,
        solutions,
        t,
        y,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"### Bond pricing results")
    return


@app.cell
def __(np, plt, s_nominal, solutions, y_nominal):
    plt.figure()

    x_ax = np.arange(1, 61) / 2

    plt.plot(x_ax, y_nominal + s_nominal[0], label="actual")
    # plt.plot(x_ax, initialization, label='initialization')

    for iter, (y_, S_, a_) in enumerate(solutions):
        plt.plot(x_ax, y_ + S_[0], label=f"iteration {iter + 1}", linestyle="--")

    plt.ylabel("Yield")
    plt.xlabel("Maturity (years)")

    # plt.ylim(0.04, 0.06)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))

    plt.legend()
    plt.tight_layout()
    plt.gcf()
    return S_, a_, iter, x_ax, y_


@app.cell
def __(np, plt, s, s_nominal_unique):
    ratings_labels = ["AA", "A", "BBB", "BB"]
    x = np.arange(len(ratings_labels))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(
        x - width / 2, s_nominal_unique[1:] - s_nominal_unique[0], width, label="actual"
    )
    rects2 = ax.bar(x + width / 2, s.value[1:] - s.value[0], width, label="fitted")

    plt.ylabel("Spread")
    plt.xlabel("Rating")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))

    ax.set_xticks(x)
    ax.set_xticklabels(ratings_labels)

    ax.legend()
    plt.tight_layout()
    plt.gcf()
    return ax, fig, ratings_labels, rects1, rects2, width, x


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
