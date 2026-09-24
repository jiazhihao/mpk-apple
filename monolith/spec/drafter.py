"""The Drafter contract (design §5.14).

A drafter is a :class:`Module` (it has weights, an oracle and IR emission) that proposes a block of ``gamma`` tokens,
scores them, chooses how many to verify from the chip profile's cost table, and consumes the target's committed
positions back into its own context. The verify/accept ops and the dynamic-T program are shared by every drafter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.ir import Graph, Value
from ..core.profile import Profile
from ..nn.module import Module


@dataclass
class DraftContext:
    """What the target exposes to a drafter: the feature-tap buffers of the tapped layers and the anchor token."""

    taps: List[Value] = field(default_factory=list)      # residual stream after each tapped target layer, [T, H]
    anchor: Optional[Value] = None                       # the last committed token id


@dataclass
class DraftBlock:
    """A drafter's proposal: ``gamma`` token ids, their confidences and the draft hidden states."""

    tokens: Value            # [gamma] i32
    confidences: Value       # [gamma] f32, acceptance probability per position (1.0 if the drafter has no head)
    hidden: Optional[Value]  # [gamma, H_draft] or None
    gamma: int


class Drafter(Module):
    gamma: int = 0

    def lower_draft(self, g: Graph, ctx: DraftContext, anchor: Value) -> DraftBlock:
        raise NotImplementedError

    def lower_select(self, g: Graph, block: DraftBlock, profile: Profile) -> Value:
        """Emit the SERIAL op choosing the verify length ``L`` (a [1] u32 Value written into StepState)."""
        raise NotImplementedError

    def lower_context_update(self, g: Graph, taps: List[Value], accepted: Value) -> None:
        """Feed the committed positions' target features back into the drafter's context."""
        raise NotImplementedError
