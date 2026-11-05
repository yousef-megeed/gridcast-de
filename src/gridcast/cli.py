"""Command-line entry point: ``gridcast <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from gridcast.config import load_config

logger = logging.getLogger("gridcast")


def _fetch(cfg, args) -> None:
    from gridcast.data.pipeline import build_dataset

    _, report = build_dataset(cfg)
    print(json.dumps(report.to_dict(), indent=2))


def _tune(cfg, args) -> None:
    from gridcast import tune

    runner = {"ridge": tune.tune_ridge, "lightgbm": tune.tune_gbm, "attention_net": tune.tune_nn}
    print(json.dumps(runner[args.model](cfg), indent=2))


def _backtest(cfg, args) -> None:
    from gridcast.backtest import VARIANTS, run_backtest

    if args.model == "all":
        # A fresh process per model, because torch and LightGBM must not train side by side.
        for name in VARIANTS:
            subprocess.run([sys.executable, "-m", "gridcast.cli", "backtest", name], check=True)
        return
    run_backtest(cfg, args.model, tuple(args.seeds) if args.seeds else None)


def _evaluate(cfg, args) -> None:
    from gridcast.evaluate import render_table, run_evaluation

    report = run_evaluation(cfg)
    print(render_table(report))
    print(json.dumps({k: report[k] for k in ("selected_model", "selection_pinball_pre_test")}))


def _analyze(cfg, args) -> None:
    from gridcast.analysis import run_analysis

    run_analysis(cfg, part=args.part)


def _bundle(cfg, args) -> None:
    from gridcast.bundle import build_bundle, fit_member

    if args.member:
        fit_member(cfg, args.member, args.output)
    else:
        build_bundle(cfg, args.output, reuse=args.reuse)


def _serve(cfg, args) -> None:
    import uvicorn

    uvicorn.run("gridcast.serve.app:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="gridcast", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch", help="Download and validate all data").set_defaults(run=_fetch)

    tune = commands.add_parser("tune", help="Search hyperparameters on pre-test data")
    tune.add_argument("model", choices=["ridge", "lightgbm", "attention_net"])
    tune.set_defaults(run=_tune)

    backtest = commands.add_parser("backtest", help="Run the rolling-origin backtest")
    backtest.add_argument("model")
    backtest.add_argument("--seeds", type=int, nargs="*")
    backtest.set_defaults(run=_backtest)

    commands.add_parser("evaluate", help="Score predictions").set_defaults(run=_evaluate)

    analyze = commands.add_parser("analyze", help="Create figures and error analysis")
    analyze.add_argument("--part", choices=["all", "report", "shap", "attention"], default="all")
    analyze.set_defaults(run=_analyze)

    bundle = commands.add_parser("bundle", help="Train and package the deployment model")
    bundle.add_argument("--output", default="artifacts/model")
    bundle.add_argument("--member", default=None)
    bundle.add_argument("--reuse", action="store_true", help="Keep members already trained")
    bundle.set_defaults(run=_bundle)

    serve = commands.add_parser("serve", help="Run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(run=_serve)

    args = parser.parse_args(argv)
    args.run(load_config(args.config), args)


if __name__ == "__main__":
    main()
