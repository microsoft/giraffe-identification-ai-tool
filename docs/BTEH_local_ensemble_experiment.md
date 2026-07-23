# BTEH Local-Ensemble Experiment Provenance

This retrospective experiment is registered by the content hash stored in:

`experiments/full_local_ensemble/retrospective_registration.json`

The existing fixed probes had already informed selection of `selected-v1`, so
the experiment cannot replace production. It can only report retrospective
evidence and define a future-session protocol.

## Baseline correction

The first fixed-probe evaluator implementation masked the newly fitted
four-channel OOF weights to construct the `selected_v1` baseline. Because the
new OOF fit assigned zero weight to body MiewID, that made the apparent baseline
effectively ear-only rather than the frozen production fusion.

This was detected before final reporting because query-weighted reciprocal rank
did not reproduce the registered selected-v1 value of 0.473. The erroneous
baseline rows were preserved as:

`fixed_probe/probe_rankings_pre_baseline_correction.parquet`

The authoritative `selected_v1` rows in `fixed_probe/probe_rankings.parquet`
were replaced from the pre-existing, registration-anchored artifact:

`reports/calibrated_eval_projected/normalized_eval_rankings.parquet`

No candidate/local ranking was changed and no probe was rescored. Postprocess
now hard-fails unless selected-v1 query-weighted reciprocal rank reproduces
0.473 within the registered tolerance.

## Multiplicity interpretation

The frozen registration uses the shorthand `holm_max_t`. The implemented
analysis treats the sole primary contrast at unadjusted two-sided alpha 0.05;
Holm-adjusted p-values and max-|T| simultaneous intervals apply only to the
four registered secondary contrasts. New registrations use the explicit label
`primary_unadjusted_secondaries_holm_max_t`.

## Outcome

The local ensemble substantially improved retrospective retrieval accuracy, but
it failed operational gates for p95 latency and both-local coverage. Therefore
`selected-v1` remains production.
