"""A1 deterministic allocation baselines and small-instance MILP."""

from .common import SolverProtocol, SolverResult, load_solver_protocol
from .heuristics import (
    solve_deadline_aware_greedy,
    solve_deadline_aware_load_balanced,
    solve_greedy,
    solve_hybrid_load_balanced,
    solve_hungarian,
    solve_load_balanced,
)
from .milp import (
    solve_assignment_milp,
    solve_deadline_aware_assignment_milp,
    solve_hybrid_assignment_milp,
)
from .joint_search import solve_joint_assignment_sequence_reference
from .assignment_beam import solve_assignment_beam_sequence
from .lns import solve_beam_alns, solve_deterministic_lns, solve_order_aware_lns

__all__ = [
    "SolverProtocol",
    "SolverResult",
    "load_solver_protocol",
    "solve_greedy",
    "solve_load_balanced",
    "solve_hungarian",
    "solve_assignment_milp",
    "solve_deterministic_lns",
    "solve_deadline_aware_greedy",
    "solve_deadline_aware_load_balanced",
    "solve_deadline_aware_assignment_milp",
    "solve_order_aware_lns",
    "solve_hybrid_load_balanced",
    "solve_hybrid_assignment_milp",
    "solve_joint_assignment_sequence_reference",
    "solve_beam_alns",
    "solve_assignment_beam_sequence",
]
