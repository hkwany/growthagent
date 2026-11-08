"""Mock ad-platform tools (create campaign / get status).

Stateful, in-memory.  This models the "autonomous digital worker" surface
of Hellyeah AI: an agent that provisions and inspects ad campaigns.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import tool

_campaigns: Dict[str, Dict[str, Any]] = {}
_seq = [0]


def reset() -> None:
    """Clear all mock campaigns (used by tests)."""
    _campaigns.clear()
    _seq[0] = 0


@tool(
    name="create_campaign",
    description="Create a new ad campaign in the (mock) ad platform.",
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "Campaign objective, e.g. 'app_installs' or 'conversions'."},
            "budget": {"type": "number", "description": "Daily budget in USD."},
            "creative": {"type": "string", "description": "Ad creative / copy to run."},
        },
        "required": ["objective", "budget", "creative"],
    },
)
def create_campaign(objective: str, budget: float, creative: str) -> Dict[str, Any]:
    _seq[0] += 1
    cid = f"camp_{_seq[0]}"
    _campaigns[cid] = {
        "objective": objective,
        "budget": budget,
        "creative": creative,
        "status": "active",
        "created": True,
    }
    return {"campaign_id": cid, "status": "active"}


@tool(
    name="get_campaign_status",
    description="Get the current status and config of a campaign by id.",
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": {"type": "string", "description": "Campaign id returned by create_campaign."}
        },
        "required": ["campaign_id"],
    },
)
def get_campaign_status(campaign_id: str) -> Dict[str, Any]:
    if campaign_id not in _campaigns:
        return {"error": f"unknown campaign: {campaign_id}"}
    return {"campaign_id": campaign_id, **_campaigns[campaign_id]}
