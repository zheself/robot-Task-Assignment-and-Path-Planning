# A1 W3 analytical edge-oracle card

Version: `analytical-edge-oracle-v1`  
Evidence: `SIM_GEOMETRIC` proxy only.

The oracle computes deterministic curve features, tool/capability compatibility,
a base-to-curve maximum-distance reach proxy, common time-window feasibility,
an optional spherical no-go proxy, approach travel time and a reach-ratio risk.
Every blocked robot–segment edge has stable reason codes.

This is not IK, motion planning, geometric robot collision checking, controller
timing, process physics, or factory validation. Its confidence defaults to
`0.25`; unknown kinematic models explicitly use the configured default reach
and emit `DEFAULT_REACH_USED`. A later WP-B implementation may replace estimates
behind the same interface without upgrading historical evidence labels.
