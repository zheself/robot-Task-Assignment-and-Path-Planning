"""A4b development-only feasibility-aware LNS foundations.

The package contains no neural model.  It isolates atomic-unit destroy
selection while keeping repair, acceptance, scheduling, verification and
budgets shared across the controlled ordinary-LNS family.
"""

from .alns import (
    AlnsConfig,
    RepairOutcome,
    SearchOutcome,
    repair_destroyed_state,
    run_search,
    update_operator_weight,
)
from .anytime import (
    InitializerOutcome,
    InitializerProvenance,
    adapt_solver_initializer,
    build_hybrid_load_balanced_initializer,
)
from .operators import DESTROY_OPERATORS, HANDCRAFTED_OPERATORS, select_destroy_set
from .trace import (
    AnytimeSnapshot,
    IncumbentEvent,
    SearchStep,
    SearchTrace,
    best_at_budget,
    best_at_iteration,
    canonical_hash,
    replay_trace,
)

__all__ = [
    "AlnsConfig",
    "AnytimeSnapshot",
    "DESTROY_OPERATORS",
    "HANDCRAFTED_OPERATORS",
    "IncumbentEvent",
    "InitializerOutcome",
    "InitializerProvenance",
    "RepairOutcome",
    "SearchOutcome",
    "SearchStep",
    "SearchTrace",
    "adapt_solver_initializer",
    "best_at_budget",
    "best_at_iteration",
    "build_hybrid_load_balanced_initializer",
    "canonical_hash",
    "repair_destroyed_state",
    "replay_trace",
    "run_search",
    "select_destroy_set",
    "update_operator_weight",
]
