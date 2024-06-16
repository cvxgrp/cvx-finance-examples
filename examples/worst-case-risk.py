import marimo

__generated_with = "0.6.17"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Worst-case risk analysis

        We consider a worst-case portfolio risk analysis, under known portfolio weight
        vector and given bounds on the covariance matrix entries.

        ## Setup

        ### Problem Statement
        We hold $n$ assets with weights $w\in\mathbf{R}^n$, $\mathbf{1}^Tw=1$. We know
        that the asset covariance $\Sigma \in \mathcal{S}$, where

        $$
        \mathcal S = \left\{ \Sigma \geq 0 \mid
        L_{ij} \leq \Sigma_{ij} \leq U_{ij}, \quad i,j=1,\ldots,n   \right\}
        $$

        is a known set. We want to find the the worst-case variance consistent with
        our belief $\Sigma \in \mathcal S$:

        $$
        \sigma_{\text{wc}}^2 = \sup\left\{w^T \Sigma w \mid \Sigma \geq 0, ~
        \Sigma \in \mathcal S \right\}.
        $$
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import cvxpy as cp

    return cp, np, pd


@app.cell
def __(mo):
    sigma12 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="+", label=r"$\Sigma_{12}$"
    )

    sigma13 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="+", label=r"$\Sigma_{13}$"
    )

    sigma14 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="+/-", label=r"$\Sigma_{14}$"
    )

    sigma23 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="-", label=r"$\Sigma_{23}$"
    )

    sigma24 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="-", label=r"$\Sigma_{23}$"
    )

    sigma34 = mo.ui.dropdown(
        options=["+", "-", "+/-"], value="+", label=r"$\Sigma_{34}$"
    )
    return sigma12, sigma13, sigma14, sigma23, sigma24, sigma34


@app.cell
def __(np, sigma12, sigma13, sigma14, sigma23, sigma24, sigma34):
    Sigma_array = np.array(
        [
            [0.2, sigma12.value, sigma13.value, sigma14.value],
            [sigma12.value, 0.1, sigma23.value, sigma24.value],
            [sigma13.value, sigma23.value, 0.3, sigma34.value],
            [sigma14.value, sigma24.value, sigma34.value, 0.1],
        ]
    )
    return (Sigma_array,)


@app.cell
def __(Sigma_array, np):
    def array_to_latex(array):
        rows = array.shape[0]
        cols = array.shape[1]
        latex_array = "\\begin{array}{cccc}\n"
        for i in range(rows):
            row = " & ".join([r"\pm" if elem == "+/-" else elem for elem in array[i]])
            if i < rows - 1:
                row += " \\\\"
            latex_array += f"{row}\n"
        latex_array += "\\end{array}"
        return latex_array

    def latex_to_signs(latex_Sigma):
        sign_map = {"+": 1, "-": -1, r"\pm": 0, "+/-": 0}

        latex_Sigma = latex_Sigma.replace("\\begin{array}{cccc}", "").replace(
            "\\end{array}", ""
        )
        rows = latex_Sigma.strip().split("\\\\")

        signs = []

        for row in rows:
            row_signs = []
            elements = row.strip().split(" & ")
            for elem in elements:
                if elem in sign_map:
                    row_signs.append(sign_map[elem])
                else:
                    try:
                        row_signs.append(float(elem))
                    except ValueError:
                        row_signs.append(0)
            signs.append(row_signs)

        Signs = np.array(signs)
        np.fill_diagonal(Signs, 1)
        return Signs

    latex_Sigma = array_to_latex(Sigma_array)
    return array_to_latex, latex_Sigma, latex_to_signs


@app.cell
def __(mo):
    mo.md(r"## Example")
    return


@app.cell
def __(mo, sigma12, sigma13, sigma14, sigma23, sigma24, sigma34):
    mo.hstack([sigma12, sigma13, sigma14, sigma23, sigma24, sigma34])
    return


@app.cell
def __(latex_Sigma, mo, w):
    latex_output = f"""
    $$
    w=\\left[ \\begin{{array}}{{r}}
        {w[0]} \\\\ {w[1]} \\\\ {w[2]} \\\\ {w[3]}
        \\end{{array}}\\right], \\qquad
    \\Sigma = \\left[
    {latex_Sigma}
    \\right].
    $$
    """
    mo.md(
        rf"""
        We consider an example with

        {latex_output}

        $+$ means the corresponding entry in the covariance matrix is nonnegative, $-$ means nonpositive, and $\pm$ means
        the sign is unknown.
        """
    )
    return (latex_output,)


@app.cell
def __(mo):
    mo.md(r"### Results")
    return


@app.cell
def __(latex_Sigma, latex_to_signs, np):
    n = 4
    K = 3

    Signs = latex_to_signs(latex_Sigma)
    diag = np.array([0.2, 0.1, 0.3, 0.1])
    w = np.array([0.5, 0.25, -0.05, 0.3])
    return K, Signs, diag, n, w


@app.cell
def __(Signs, cp, diag, mo, n, np, pd, w):
    Sigma = cp.Variable((n, n), PSD=True)

    objective = cp.Maximize(cp.quad_form(w, Sigma))
    constraints = [cp.diag(Sigma) == diag]
    for i in range(n):
        for j in range(i):
            if Signs[i, j] != 0:
                constraints += [Signs[i, j] * Sigma[i, j] >= 0]

    prob = cp.Problem(objective, constraints)
    wc_risk = prob.solve()

    with mo.redirect_stdout():
        print(f"Worst-case risk: {wc_risk:.0%}")
        print(f"Worst-case volatility: {np.sqrt(wc_risk):.0%}")
        print(f"\nWorst-case covariance: \n{pd.DataFrame(Sigma.value).round(2)}")
        diag_risk = (diag * w**2).sum()
        print(f"\nRisk with diagonal covariance: {diag_risk:.0%}")
        print(f"Volatility with diagonal covariance: {np.sqrt(diag_risk):.0%}")
    return Sigma, constraints, diag_risk, i, j, objective, prob, wc_risk


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
