# A2 joint/beam development decision

Evidence: `SIM_GEOMETRIC`  
Accessed benchmark splits: v3 `train`, `validation` only  
Decision: **DEVELOPMENT_GATE_FAILED — DO NOT CREATE V4 YET**

## Registered checks

| check | registered threshold | observed | result |
|---|---:|---:|---|
| validation candidate coverage | >= 95% | 47/48 = 97.9% | pass |
| maximum per-cell drop vs order-aware LNS | <= 1/12 | 0 | pass |
| cells with strict coverage gain vs order-aware LNS | >= 1 | 0 | **fail** |
| validation median runtime | <= 5 s | 1.845 s | pass |

Beam-ALNS tied order-aware LNS coverage in every validation cell. It improved
conditional mean proxy score in medium-precedence, medium-resource and
small-sparse, but was slightly worse in medium-balanced and approximately 3.6x
slower in overall validation median runtime. Conditional scores exclude failed
instances and are not a substitute for the failed registered coverage-gain
criterion.

On train, beam-ALNS improved medium-precedence coverage from 45/48 to 47/48 and
matched 100% coverage in the other cells. This is useful engineering evidence,
but train improvement alone cannot authorize a new frozen gate.

## Joint reference result

The bounded joint reference passed its separate protocol: all five valid hand
fixtures completed with verified proxy-optimal plans; among eight registered
small v3-train cases, four completed and four returned verified incumbents at
the 10-second limit. `optimal` is used only when enumeration was complete.

## Consequence

V4 is not generated in this batch. V2/v3 frozen and stress remain unread by the
development runner. The next A2 revision should improve assignment branching or
constructively certify benchmark feasibility using hand fixtures and v3
train/validation; any new budget/protocol must be registered before another
development run. No current evidence supports A3 entry.
