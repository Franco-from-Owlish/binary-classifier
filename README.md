# Binary classification of reordering

## Scripts

UV can be used to run scripts for filtering and classifying.

```bash
uv run filter
```

```bash
uv run classify
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
