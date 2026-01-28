from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/corpora")
    parser.add_argument("--output", type=str, default="data/runs/experts.pt")
    args = parser.parse_args()
    print({"status": "stub", "data": args.data, "output": args.output})


if __name__ == "__main__":
    main()
