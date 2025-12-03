import argparse
import subprocess
from src.scripts import train, infer


def main():
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command")

    tr = subparsers.add_parser("train")
    # TODO: Add arguments
    tr.set_defaults(func=train.run_train)

    inf = subparsers.add_parser("infer")
    # TODO: Add arguments
    inf.set_defaults(func=infer.run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
