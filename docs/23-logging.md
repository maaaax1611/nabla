# Logging / Metric Tracking

## Why not just `print()`

Every example so far has printed loss values straight to the console -
fine for watching a run live, but nothing plottable survives once the
terminal scrolls past it or the process ends. [`logging.py`](../src/nabla/logging.py)'s
`History` is deliberately the smallest thing that fixes that: an
in-memory record of named metrics against step numbers, exportable to
CSV - not a dependency on TensorBoard/wandb/etc., which would be a lot
of surface area for what a training loop this size actually needs.

## Different metrics, different steps

A training loop typically doesn't log everything at the same cadence -
train loss every step, but validation loss only every `eval_interval`
steps (recomputing val loss every step would be wasteful). `History`
handles this by keeping each metric's own list of `(step, value)`
pairs rather than one row per step with columns for every metric:

```python
def log(self, step, **metrics):
    for name, value in metrics.items():
        self.records.setdefault(name, []).append((step, float(value)))
```

`history.log(step, train_loss=..., lr=..., grad_norm=...)` logs three
metrics at once at the same step;
`history.log(step, val_loss=...)` a few lines later, only every
`eval_interval` steps, logs a fourth metric that simply has fewer
recorded points than the others - no padding or `NaN`-filling needed
for the steps it wasn't logged at.

## Export: long/tidy CSV, not wide

`to_csv` writes one row per `(step, metric, value)` triple:

```
step,metric,value
1,train_loss,5.14
1,lr,6e-05
5,train_loss,4.54
5,val_loss,4.36
```

rather than one column per metric - a wide format (`step, train_loss,
val_loss, lr, ...`) would need every row to have a value for every
column, which doesn't fit metrics logged at different steps without
inventing a placeholder for the gaps. The long format sidesteps that
entirely, and is what most plotting tools (pandas `pivot`, seaborn,
etc.) actually want as input anyway.

## Used in the Shakespeare example

[`examples/shakespeare_transformer.py`](../examples/shakespeare_transformer.py)
logs train loss, learning rate ([docs/20](20-lr-schedules.md)), and
gradient norm ([docs/21](21-gradient-clipping.md)) every step, and val
loss every `eval_interval` steps, then writes the whole run to
`.shakespeare_history.csv` at the end - one file with everything needed
to plot the full training curve alongside the LR schedule and gradient
norm over time, without re-running training.

## Testing

[`tests/test_logging.py`](../tests/test_logging.py) checks logging and
retrieving a single metric, logging multiple metrics at once, that
metrics logged at different steps stay independent, `last()`'s
most-recent-value lookup (and its `None` result for a metric never
logged), and that `to_csv` produces the expected long-format rows.
