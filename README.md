# Binary classification of reordering

The project uses [UV](https://docs.astral.sh/uv/) as a package manager.

## Scripts

UV can be used to run scripts for filtering, classifying, and feature importance.

```bash
uv run filter
```

```bash
uv run classify
```

```bash
uv run feat_importance
```

## Optimising

### Feature Selection

Resulted in a drop in AUC:

```text
Full model AUC: 0.6620
Top features model AUC: 0.5000
```

## Non-Linear Transformations

Polynomial features had no effect.
