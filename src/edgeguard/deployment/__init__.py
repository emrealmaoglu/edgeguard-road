"""Deployment readiness contracts without platform performance claims."""

from edgeguard.deployment.package import build_deployment_package, verify_deployment_package

__all__ = ["build_deployment_package", "verify_deployment_package"]
