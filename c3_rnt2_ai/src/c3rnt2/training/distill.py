from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="")
    parser.add_argument("--student", type=str, default="")
    parser.add_argument("--output", type=str, default="data/runs/distill.pt")
    args = parser.parse_args()
    print({"status": "stub", "teacher": args.teacher, "student": args.student, "output": args.output})


if __name__ == "__main__":
    main()
