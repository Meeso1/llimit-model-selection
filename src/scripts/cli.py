import argparse
from src.scripts import list_preprocessed_data, train, infer, list_models, list_logs


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
    ls_training_logs.set_defaults(func=list_logs.run_list_logs)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
