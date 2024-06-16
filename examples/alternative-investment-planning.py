import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Alternative investment planning

        This notebook considers the problem of making commitments to an alternative
        investment in each quarter $t=1,\ldots,T$. Over time the investor puts committed
        money into the investment in response to investment calls, and she receives
        money back from the investment in response to distributions. Examples of such
        investments are private equity funds, venture capital funds, infrastructure
        projects, etc.
        """
    )
    return


@app.cell
def __():
    import cvxpy as cp
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter, cp, np, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Setup

        ### Alternative investment dynamics
        We denote by $c_t$, $p_t$, $d_t \geq 0$ the commitments, capital calls,
        and distributions, respectively, in quarter $t$. Moreover, $n_t$ denotes the net
        asset value (NAV), and $r_t$ the investment return in quarter $t$. $u_t\geq 0$
        is the total uncalled previous commitments at the beginning of quarter $t$. The
        investment dynamics are modeled as

        $$
        n_{t+1} = n_t (1+r_t) + p_t - d_t, \qquad
        u_{t+1} = u_{t} - p_{t} + c_{t}
        $$

        with $n_0=0$, $u_0=0$. The calls and distributions are modeled as

        $$
        p_t = \gamma^\textrm{call} u_t, \qquad d_t = \gamma^\text{dist} n_t,
        $$

        where $\gamma^\text{call}, \gamma^\text{dist} \in (0,1)$ are call and
        distribution intensities or rates, respectively.

        ### Problem statement
        The objective is to choose commitments $c_1, \ldots, c_T$ to minimize

        $$
        \frac{1}{T+1} \sum_{t=1}^{T+1} \left(
        (n_t-n^{\text{des}})^2 + \lambda \frac{1}{T-1} (c_{t+1}-c_t)^2\right),
        $$

        where $n^{\text{des}}$ is the desired NAV and $\lambda > 0$ is a smoothing
        parameter. In words we penalize deviations from the desired NAV, and encourage a smooth commitment schedule.
        """
    )
    return


app._unparsable_cell(
    r"""
    mo.md(
        rf\"\"\"
        ## Example

        We consider an example with $T=32$ (eight years), $r_t=0.04$ (4% quarterly return),
        $\gamma^\text\{call\} = {gamma_call_slider.value}$, $\gamma^\text\{dist\} = \{gamma_dist_slider.value\}$, and planning parameters
        $\quad c^\text{max} ={c_max_slider.value}, \quad
        u^\text\{max\} = {u_max_slider.value} , \quad
        n^\text\{des\} = {n_des_slider.value}, \quad
        \lambda = \{lambda_slider.value\}$.
        \"\"\"
    )
    """,
    name="__",
)


@app.cell
def __(
    c_max_slider,
    gamma_call_slider,
    gamma_dist_slider,
    lambda_slider,
    mo,
    n_des_slider,
    u_max_slider,
):
    mo.md(
        rf"""
        ## Example

        We consider an example with $T=32$ (eight years), $r_t=0.04$ (4% quarterly return),
        $\gamma^{{\text{{call}}}} = {gamma_call_slider.value}$, $\gamma^{{\text{{dist}}}} = {gamma_dist_slider.value}$, and planning parameters
        $\quad c^\text{{max}} ={c_max_slider.value}, \quad
        u^\text{{max}} = {u_max_slider.value} , \quad
        n^\text{{des}} = {n_des_slider.value}, \quad
        \lambda = {lambda_slider.value}$.
        """
    )

    return


@app.cell
def __(mo):
    gamma_call_slider = mo.ui.slider(
        0.01, 0.99, step=0.01, label=r"$\gamma^{\text{call}}$", value=0.23
    )
    gamma_call_slider
    return (gamma_call_slider,)


@app.cell
def __(mo):
    gamma_dist_slider = mo.ui.slider(
        0.01, 0.99, step=0.01, label=r"$\gamma^{\text{dist}}$", value=0.15
    )
    gamma_dist_slider
    return (gamma_dist_slider,)


@app.cell
def __(mo):
    c_max_slider = mo.ui.slider(1, 10, step=1, label=r"$c^{\max}$", value=4)
    c_max_slider
    return (c_max_slider,)


@app.cell
def __(mo):
    u_max_slider = mo.ui.slider(1, 50, step=1, label=r"$u^{\max}$", value=10)
    u_max_slider
    return (u_max_slider,)


@app.cell
def __(mo):
    n_des_slider = mo.ui.slider(1, 100, step=1, label=r"$n^{\text{des}}$", value=15)
    n_des_slider
    return (n_des_slider,)


@app.cell
def __(mo):
    lambda_slider = mo.ui.slider(
        steps=[5 * 10**i for i in range(-5, 6)], label=r"$\lambda$", value=5
    )
    lambda_slider
    return (lambda_slider,)


@app.cell
def __(
    c_max_slider,
    gamma_call_slider,
    gamma_dist_slider,
    lambda_slider,
    n_des_slider,
    u_max_slider,
):
    gamma_call = gamma_call_slider.value
    gamma_dist = gamma_dist_slider.value
    c_max = c_max_slider.value
    u_max = u_max_slider.value
    n_des = n_des_slider.value
    lmbda = lambda_slider.value
    T = 32
    r = 0.04
    return T, c_max, gamma_call, gamma_dist, lmbda, n_des, r, u_max


@app.cell
def __(T, c_max, cp, gamma_call, gamma_dist, lmbda, n_des, r, u_max):
    n = cp.Variable(T + 1, nonneg=True)
    u = cp.Variable(T + 1, nonneg=True)
    p = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    c = cp.Variable(T, nonneg=True)

    tracking = cp.mean((n - n_des) ** 2)
    smoothing = lmbda * cp.mean(cp.diff(c) ** 2)

    constraints = [c <= c_max, u <= u_max, n[0] == 0, u[0] == 0]
    for t in range(T):
        constraints += [n[t + 1] == (1 + r) * n[t] + p[t] - d[t]]
        constraints += [u[t + 1] == u[t] - p[t] + c[t]]
        constraints += [p[t] == gamma_call * u[t], d[t] == gamma_dist * n[t]]

    prob = cp.Problem(cp.Minimize(tracking + smoothing), constraints)
    print(prob.solve())
    return c, constraints, d, n, p, prob, smoothing, t, tracking, u


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"### Results")
    return


@app.cell
def __(FuncFormatter, T, c, d, n, n_des, np, p, plt, u):
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["figure.figsize"] = (7.5, 5)

    times = np.arange(1, T + 2)

    plt.figure(figsize=(8, 4))
    plt.plot(times, n.value, label="n")
    plt.plot(times, u.value, label="u")
    plt.plot(times[:-1], p.value, label="p")
    plt.plot(times[:-1], d.value, label="d")
    plt.plot(times[:-1], c.value, label="c")
    plt.ylim(-0.5, n_des * 1.1)
    plt.xticks(np.arange(1, T + 2, 5))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}M"))
    plt.axhline(y=n_des, color="black", linestyle="--", label="n_des")
    plt.xlabel("Period")
    plt.legend()
    plt.gcf()
    return (times,)


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
