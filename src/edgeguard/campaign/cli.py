"""Command-line interface for the project-specific campaign layer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgeguard.campaign.contracts import PROFILES, topological_stages
from edgeguard.campaign.runner import campaign_plan, run_campaign, status_summary
from edgeguard.campaign.state import Campaign
from edgeguard.serialization import canonical_json


def _campaign(args: argparse.Namespace) -> Campaign:
    return Campaign(args.campaign_root, args.repository)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--campaign-root", type=Path, required=True)
    initialize.add_argument("--campaign-id")
    initialize.add_argument("--profile", choices=tuple(PROFILES), required=True)
    for name in ("plan", "status", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-root", type=Path, required=True)
        if name == "report":
            command.add_argument("--audience", choices=("assistant", "thesis"), required=True)
    for name in ("run", "resume"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--stop-after", choices=topological_stages())
        command.add_argument("--interrupt-stage", choices=topological_stages())
        command.add_argument("--mmseg-checkout", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = _campaign(args)
    result: Any
    if args.command == "init":
        result = campaign.initialize(campaign_id=args.campaign_id, profile=args.profile)
    elif args.command == "plan":
        result = campaign_plan(campaign)
    elif args.command == "status":
        result = status_summary(campaign)
    elif args.command in {"run", "resume"}:
        result = run_campaign(
            campaign,
            stop_after=args.stop_after,
            interrupt_stage=args.interrupt_stage,
            mmseg_checkout=args.mmseg_checkout,
        )
    else:
        from edgeguard.campaign.reporting import generate_report

        result = generate_report(campaign, audience=args.audience)
    print(canonical_json(result))
    return 0
