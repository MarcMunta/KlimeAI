from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="data/runs/adapters.pt")
    args = parser.parse_args()
    print({"status": "stub", "model": args.model, "output": args.output})


if __name__ == "__main__":
    main()
