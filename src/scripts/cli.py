import argparse
import subprocess
from src.scripts import train, infer


def main():
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command")

    tr = subparsers.add_parser("train")
    tr.add_argument("--spec-file", type=str, help="Path to JSON file containing training specification. If not provided, training specification is read from stdin.")
    tr.set_defaults(func=train.run_train)

    # TODO: Maybe improve passing prompts to command
    inf = subparsers.add_parser("infer")
    inf.add_argument("--model-type", type=str, required=True, choices=["dense_network"], help="Type of model to use")
    inf.add_argument("--model-name", type=str, required=True, help="Name of the saved model to load")
    inf.add_argument("--models-to-score", type=str, nargs="+", required=True, help="List of model names to score")
    inf.add_argument("--prompts", type=str, nargs="+", required=True, help="List of prompts to evaluate")
    inf.add_argument("--batch-size", type=int, required=True, help="Batch size for inference")
    inf.add_argument("--output-path", type=str, help="Path to output JSON file (default: auto-generated in inference_outputs/)")
    inf.set_defaults(func=infer.run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
