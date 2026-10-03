# Investment Engine Research Core (release candidate)

This is a curated, history-free public research repository released under Apache-2.0. It contains a focused research core rather than a complete trading system.

The candidate contains reusable research components for combinatorial purged cross-validation of factor IC, factor neutralization and orthogonalization, factor validation, and point-in-time out-of-sample readiness/replay. It does **not** contain market data, production strategy configurations, brokerage connectivity, private credentials, live performance records, or a complete trading system. The IC validation routines are diagnostics, not a trained predictive model or proof of genuine out-of-sample investment performance.

## Reproduce the included tests

Use Python 3.12 (the version used for candidate verification), then install the four direct dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The tests create synthetic inputs and in-memory DuckDB tables; no private warehouse is needed. For an OOS report using your own *lawfully licensed* DuckDB database:

```bash
python scripts/oos_backtest_report.py --db /path/to/your/warehouse.db --output /path/to/oos_report.json
```

The report requires the `decision_evidence` and `stock_daily_core` schema exercised in `tests/test_oos_readiness.py`. It fails closed when replayable evidence is insufficient; passing the included tests is **not** evidence of profitable live trading, regulatory compliance, or robust performance on real market data.

## License

Copyright 2026 Investment Engine contributors.

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

## Scope and provenance

This candidate was copied from a single snapshot of the private repository's `main` branch, without its Git history. The allowlist consists only of `strategies/*.py`, `scripts/oos_backtest_report.py`, `tests/test_*.py`, and this minimal documentation/dependency file. The original private repository is not a publication target.

The factor-validation implementation notes an approach inspired by AKQuant's walk-forward documentation. That reference is documentation-level attribution only unless a separate source audit identifies copied code; any such code must retain the applicable upstream notices and license terms before distribution. The candidate contains no AKQuant source file or vendored dependency. The release snapshot is subject to the final file, source, secret, and license checks described in the release record before public distribution.
