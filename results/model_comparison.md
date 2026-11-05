| Model | Weather at forecast time | MAE (MW) | RMSE (MW) | MAPE | Skill vs TSO | Pinball (MW) | 90% cov. (calibrated) |
|---|---|---:|---:|---:|---:|---:|---:|
| TSO day-ahead forecast (SMARD) | n/a | 1,982 | 2,552 | 3.78% | reference | n/a | n/a |
| Seasonal naive | none | 2,631 | 3,892 | 5.02% | -32.7% | 826 | 90.0% |
| Ridge | archived NWP | 1,742 | 2,333 | 3.35% | +12.1% | 520 | 89.8% |
| LightGBM, no weather | none | 1,449 | 1,877 | 2.75% | +26.9% | 454 | 90.1% |
| LightGBM | archived NWP | 1,290 | 1,665 | 2.46% | +34.9% | 427 | 89.9% |
| Attention net (PyTorch) | archived NWP | 1,337 | 1,732 | 2.57% | +32.5% | 408 | 89.9% |
| Ensemble (LightGBM + attention) | archived NWP | 1,241 | 1,596 | 2.37% | +37.4% | 387 | 89.9% |
| LightGBM, oracle weather | hindsight ERA5 (upper bound) | 1,270 | 1,634 | 2.42% | +35.9% | 418 | 89.9% |
