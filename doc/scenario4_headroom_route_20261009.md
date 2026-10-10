# Scenario 4: headroom-gated learned route, development only

The eight-seed second holdout is complete: the plain learned-route agent
scored 39 versus 35 for the reviewed agent, but lost 3 pops on two fields.
It therefore failed the *predeclared* worst-case gate. The current live
submission and `run_qualification.cmd` remain unchanged.

## Single predeclared candidate

`Scenario4HeadroomLearnedRouteAgent` retains the same frozen, auditable
six-feature capture model, physical beam search, wind estimate, and low-level
controller. It makes two fixed changes: halve the maximum first-leg risk
preference from 1.0 to 0.5 seconds, and only let the model rerank a leg when
its observation-derived approximate required total acceleration is at most
80% of currently available acceleration. The 20% headroom is a robustness
margin for disturbances and attitude/actuator limitations; it is not a claim
that the approximate acceleration calculation proves capture. The guard
affects candidate ranking only and uses neither seed nor simulator truth.

Screen the frozen candidate once on four already-used development fields:
`140200341` and `568300771` (plain model gained four and six pops), plus
`178420909` and `820431975` (plain model lost three pops each). The reviewed
agent's scores on these fields were 4/2/6/6, totaling 18. Require candidate
total at least 19, no field more than two below the reviewed agent, and the
headroom-skip branch exercised. If it fails, stop this candidate rather than
searching thresholds on these same fields. If it passes, freeze its source,
config and model before drawing new independent paired validation fields.

This is a development screen, not a final-round estimate or automatic
promotion. The exact four-flight result will be written under ignored local
`evaluation/results/s4_headroom_route_dev4_20261009.json`.

## Completed screen and decision

All four official simulator flights finished normally with unchanged agent
source (`25cc64f2ccdffdda99a7a3ef3bb9da221c66dd85dbbc6751e4c7118e127c906e`).
In the predeclared seed order, the candidate scored **4/8/3/4** against the
reviewed agent's **4/2/6/6**. Its total **19 versus 18** met the aggregate
threshold and the headroom-skip branch ran, but seed `178420909` was **3
versus 6**, breaching the no-worse-than-two-pops gate. It therefore **fails**
the development screen; no new holdout seeds were drawn and no submission or
live-launcher file is changed.

The guard kept the reviewed agent's first target on seed `178420909`, unlike
the plain learned agent, but chose a different route after that target and
still scored only three. Thus the first-leg acceleration-headroom check did
not solve the later route-selection error. On seed `140200341` it also erased
the plain learned agent's four-pop gain. Do not tune this threshold on these
four fields. Future work, if time permits, needs a model of *multi-leg route
outcome and actuator-limited tracking*, evaluated on newly reserved seeds;
these observed fields are now development data, not independent validation.
