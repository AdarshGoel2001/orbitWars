# Long-Term Fixes

## Same-turn combat projection

`GameView._project_garrison()` currently uses a simple sequential combat proxy:
it sorts inbound fleets by ETA and applies them one at a time. This is good
enough for early model-plumbing work, but it can mispredict messy contested
arrivals.

The real Orbit Wars combat rules group fleets that arrive on the same turn by
owner, resolve the largest force against the second-largest force, then apply
survivors to the planet garrison. Sequential add/subtract can produce a
different owner or garrison when multiple owners arrive at the same ETA.

When this starts affecting training quality, update `_project_garrison()` and
`_will_fall_if_ignored()` to process arrivals grouped by `eta` and resolve each
same-turn group with game-faithful combat semantics.
