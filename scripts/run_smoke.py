"""Thin wrapper around the package smoke command."""

from edgeguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke"]))
