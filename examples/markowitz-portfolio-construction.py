import marimo

__generated_with = "0.6.13"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Markowitz portfolio construction

        In mean-variance (Markowitz) optimization, the objective is to maximize return,
        subject to constraints on risk and other variables, such as leverage, turnover,
        and position sizes.

        ### Problem formulation
        A practical formulation of the mean-variance optimization problem is

        $$
        \begin{array}{ll}
        \text{maximize} & \mu^Tw -
        \gamma^\text{hold}\phi^\text{hold}(w,c) -
        \gamma^\text{trade}\phi^\text{trade}(z)\\
        \text{subject to} &  \mathbf{1}^T w + c = 1, \quad z=w-w^\text{pre},\\
                            &  w^\text{min} \leq w \leq w^\text{max},
        \quad c^\text{min} \leq c \leq c^\text{max},
                    \quad L \leq L^\text{tar},\\
                            &  z^\text{min} \leq z \leq z^\text{max},
                    \quad T \leq T^\text{tar},\\
                            &  \| \Sigma^{1/2} w \|_2 \leq \sigma^\text{tar},
        \end{array}
        $$

        where $w$ is the vector of asset weights, $\mu$ is the vector of (estimated) expected
        returns, $\Sigma$ is the (estimated) covariance matrix, $c$ is the cash weight, $z$ is the
        vector of trades, $w^\text{pre}$ is the previous period's asset weights,
        $\gamma^\text{hold}$ and $\gamma^\text{trade}$ are penalty parameters for
        holding and trading, $\phi^\text{hold}$ and $\phi^\text{trade}$ are holding and
        trading cost functions, and $\sigma^\text{tar}$ is the risk limit. In [Markowitz
        Portfolio Construction at
        Seventy](https://web.stanford.edu/~boyd/papers/markowitz.html), we walk through details of the problem formulation and its solution.
        """
    )
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
