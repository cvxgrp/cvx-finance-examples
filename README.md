# Convex Optimization in Quantitative Finance

This code repository accompanies the [Convex Optimization in Quantitative
Finance](https://web.stanford.edu/~boyd/papers/cvx-finance.html) slides.

## Poetry

We manage dependencies through [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock). To activate the virtual environment
you can run

```bash
make activate
```

## Marimo

All of our examples are provided as [Marimo](https://marimo.io/) notebooks in
the [examples](examples) directory. To view a notebook you can run

```bash
marimo run examples/notebook.py
```

For example, to run the
[sparse inverse covariance estimation](examples/sparse-inverse-covariance.py)
notebook you can run

```bash
marimo run examples/sparse-inverse-covariance.py
```

This displays the notebook as an app. Analogously, you can also run

```bash
marimo edit examples/notebook.py
```

This opens the notebook in an editor, and you can view and make changes to the
notebook code.
