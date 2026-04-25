import argparse
from src.scripts import list_preprocessed_data, train, infer, list_models, list_logs, inspect_log


def main():
    parser = argparse.ArgumentParser(description="LLimit Model Selection CLI")
    subparsers = parser.add_subparsers(dest="command")

    tr = subparsers.add_parser("train")
    tr.add_argument("--spec-file", type=str, help="Path to JSON file containing training specification. If not provided, training specification is read from stdin.")
    tr.set_defaults(func=train.run_train)

    # TODO: Maybe improve passing prompts to command
    inf = subparsers.add_parser("infer")
    inf.add_argument("--model", type=str, required=True, help="Type and name of the saved model to load (e.g. 'dense_network/model_name')")
    inf.add_argument("--models-to-score", type=str, nargs="+", required=True, help="List of model names to score")
    inf.add_argument("--prompts", type=str, nargs="+", required=True, help="List of prompts to evaluate")
    inf.add_argument("--batch-size", type=int, required=False, default=128, help="Batch size for inference")
    inf.add_argument("--output-path", type=str, help="Path to output JSON file (default: auto-generated in inference_outputs/)")
    inf.set_defaults(func=infer.run_infer)

    ls = subparsers.add_parser("list")
    list_subparsers = ls.add_subparsers(dest="command")

    ls_models = list_subparsers.add_parser("models")
    ls_models.add_argument("--list-checkpoints", action="store_true", help="List checkpoints separately")
    ls_models.set_defaults(func=list_models.run_list_models)

    ls_preprocessed_data = list_subparsers.add_parser("preprocessed")
    ls_preprocessed_data.set_defaults(func=list_preprocessed_data.run_list_preprocessed_data)

    ls_training_logs = list_subparsers.add_parser("logs")
    ls_training_logs.add_argument("--list-timestamps", action="store_true", help="List timestamps separately")
    ls_training_logs.add_argument("--json", action="store_true", help="Emit runs as a JSON array (unix-int timestamps, suitable for jq)")
    ls_training_logs.set_defaults(func=list_logs.run_list_logs)

    insp = subparsers.add_parser("inspect", help="Inspect a training run log, printing it as JSON")
    insp.add_argument("run_name", type=str, help="Base name of the training run to inspect")
    insp.add_argument("--timestamp", type=int, default=None, help="Unix timestamp of a specific version (defaults to latest)")
    insp.add_argument("--include-config", action=argparse.BooleanOptionalAction, default=True, help="Include model config (default: on)")
    insp.add_argument("--include-final-metrics", action=argparse.BooleanOptionalAction, default=True, help="Include final metrics (default: on)")
    insp.add_argument("--include-epoch-logs", action=argparse.BooleanOptionalAction, default=False, help="Include per-epoch logs (default: off)")
    insp.set_defaults(func=inspect_log.run_inspect_log)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
