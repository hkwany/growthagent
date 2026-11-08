"""Mock analytics tool: returns deterministic performance metrics.

Metrics are derived from a stable hash of the campaign id so the same
campaign always reports the same numbers — good for reproducible demos
and tests.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from .base import tool

_spend: Dict[str, float] = {}


def reset() -> None:
    _spend.clear()


def _metrics_for(campaign_id: str, spend: float) -> Dict[str, Any]:
    h = int(hashlib.sha256(campaign_id.encode()).hexdigest(), 16)
    ctr = 0.8 + (h % 40) / 100.0          # 0.8% .. 1.2%
    cpc = 0.40 + (h % 60) / 100.0         # $0.40 .. $0.99
    impressions = int(spend / cpc * 1000) if cpc > 0 else 0
    clicks = int(impressions * ctr / 100.0)
    cvr = 3.0 + (h % 30) / 10.0           # 3% .. 6%
    conversions = int(clicks * cvr / 100.0)
    roas = round(conversions * 5.0 / spend, 2) if spend > 0 else 0.0
    return {
        "campaign_id": campaign_id,
        "spend": round(spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr,
        "cpc": round(cpc, 2),
        "conversions": conversions,
        "roas": roas,
    }


@tool(
    name="get_metrics",
    description="Get performance metrics (spend, CTR, CPC, conversions, ROAS) for a campaign.",
    parameters={
        "type": "object",
        "properties": {
            "campaign_id": {"type": "string", "description": "Campaign id returned by create_campaign."},
            "spend": {"type": "number", "description": "Total spend to attribute to this campaign, in USD."},
        },
        "required": ["campaign_id"],
    },
)
def get_metrics(campaign_id: str, spend: float = 0.0) -> Dict[str, Any]:
    if spend <= 0:
        spend = _spend.get(campaign_id, 500.0)
    _spend[campaign_id] = spend
    return _metrics_for(campaign_id, spend)


def record_spend(campaign_id: str, spend: float) -> None:
    """Set the spend used for a campaign (used by the demo script)."""
    _spend[campaign_id] = spend
