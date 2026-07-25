"""Thin wrapper that emits the doctor environment report as JSON."""

from edgeguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor", "--json"]))
