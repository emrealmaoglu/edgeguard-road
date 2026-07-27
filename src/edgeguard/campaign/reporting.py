"""Reporting entry point implemented by the campaign report milestone."""

from __future__ import annotations

from typing import Any

from edgeguard.campaign.state import Campaign


def generate_report(campaign: Campaign, *, audience: str) -> dict[str, Any]:
    """Reject reporting until the bounded report factory is installed."""
    del campaign, audience
    raise RuntimeError("campaign reporting milestone is not installed")
