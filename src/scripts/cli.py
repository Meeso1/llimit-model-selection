import argparse
import subprocess


def run_preprocess(args):
    # TODO: Implement
    pass


def run_train(args):
    # TODO: Implement
    pass


def run_infer(args):
    # TODO: Implement
    pass


def main():
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command")

    pp = subparsers.add_parser("preprocess")
    # TODO: Add arguments
    pp.set_defaults(func=run_preprocess)

    tr = subparsers.add_parser("train")
    # TODO: Add arguments
    tr.set_defaults(func=run_train)

    inf = subparsers.add_parser("infer")
    # TODO: Add arguments
    inf.set_defaults(func=run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
