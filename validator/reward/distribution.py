"""Pure weight-distribution math (no bittensor / network dependencies).

Separated from ``weight_setter`` so the allocation logic is unit-testable
without importing heavy chain dependencies.
"""

from typing import Iterable, Optional, Sequence, Tuple


# Fraction of total weight allocated to each non-reward tournament's
# preliminary leader.
LEADER_WEIGHT_FRACTION = 0.01

# UID that receives unallocated / burned weight.
DEFAULT_BURN_UID = 0

# (uid, share_of_remainder) — shares should sum to ~1.0 across recipients.
RewardRecipient = Tuple[int, float]


def compute_weight_distribution(
    leader_uids: Iterable[Optional[int]],
    reward_winner_uid: Optional[int] = None,
    burn_uid: int = DEFAULT_BURN_UID,
    leader_fraction: float = LEADER_WEIGHT_FRACTION,
    reward_recipients: Optional[Sequence[RewardRecipient]] = None,
) -> dict[int, float]:
    """Build the combined ``{uid: weight}`` distribution.

    Args:
        leader_uids: One entry per non-reward tournament. Each is the UID of
            that tournament's preliminary leader, or ``None`` to skip (leader
            unknown / not in the metagraph). The skipped 1% rolls into the
            remainder.
        reward_winner_uid: Legacy single reward-period winner UID. Used only
            when ``reward_recipients`` is omitted; equivalent to
            ``[(reward_winner_uid, 1.0)]``.
        burn_uid: UID that receives unallocated weight (default 0).
        leader_fraction: Fixed fraction per leader (default 1%).
        reward_recipients: Optional list of ``(uid, share)`` for the remainder.
            Shares are relative fractions of the remainder (normalized if they
            do not already sum to 1). Missing / empty → remainder burns.

    Returns:
        A normalized ``{uid: weight}`` map summing to 1.0. Duplicate UIDs
        (leader == winner == burn) are merged by addition.
    """
    weight_map: dict[int, float] = {}
    fixed_total = 0.0

    for uid in leader_uids:
        if uid is None:
            continue
        weight_map[uid] = weight_map.get(uid, 0.0) + leader_fraction
        fixed_total += leader_fraction

    remainder = max(0.0, 1.0 - fixed_total)

    recipients: list[RewardRecipient]
    if reward_recipients is not None:
        recipients = [(int(uid), float(share)) for uid, share in reward_recipients if share > 0]
    elif reward_winner_uid is not None:
        recipients = [(int(reward_winner_uid), 1.0)]
    else:
        recipients = []

    if not recipients:
        weight_map[burn_uid] = weight_map.get(burn_uid, 0.0) + remainder
    else:
        share_total = sum(s for _, s in recipients)
        if share_total <= 0:
            weight_map[burn_uid] = weight_map.get(burn_uid, 0.0) + remainder
        else:
            allocated = 0.0
            for uid, share in recipients:
                portion = remainder * (share / share_total)
                weight_map[uid] = weight_map.get(uid, 0.0) + portion
                allocated += portion
            # Floating-point leftovers burn.
            leftover = remainder - allocated
            if leftover > 1e-12:
                weight_map[burn_uid] = weight_map.get(burn_uid, 0.0) + leftover

    total = sum(weight_map.values())
    if total <= 0:
        return {burn_uid: 1.0}
    if abs(total - 1.0) > 1e-9:
        weight_map = {uid: w / total for uid, w in weight_map.items()}

    return weight_map
