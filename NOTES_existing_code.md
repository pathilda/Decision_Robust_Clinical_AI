# Existing code reused by the `dr` inference pipeline

Written for spec Section 12, step 1. The `dr/` package wraps this code and does not modify it.

## Functions reused

| Need | Existing function | Used in |
|---|---|---|
| Client file loading (first row per normalized ID, PT/OT/SLP filter, ID normalization keeping leading zeros) | `clinical_target_extraction.src.data.load_clients`, `DataConfig`, `ColumnConfig` | `dr.chunks`, `dr.icf` |
| Pre-treatment note text | `ClientNote.note_text` = `<SERVICE_PLANNING_TEXT>…</SERVICE_PLANNING_TEXT>` + `<ASSESSMENT_TEXT>…</ASSESSMENT_TEXT>` | `dr.icf` (`{NOTES}`) |
| ID normalization | `patient_assignment.cohort.normalize_id` | `dr.context`, `dr.evaluate` |
| Load profiles (`balanced`, `unbalanced`, `highly_unbalanced`; `initial_count`, `initial_complexity`, `capacity`) | `patient_assignment.proxy_inputs.profiles` | `dr.or_bridge.load_profiles` |
| Deterministic OR optimum (complexity range) | `patient_assignment.experiment_solver.nominal` | label, score pools; benchmark |
| Seeded optimal selection (random linear objective over the certified optimal set) | `patient_assignment.proxy_solver.select` | label, score pools; reference allocation |
| Stochastic SAA scenarios | `patient_assignment.proxy_policy_solver.scenarios` | distribution pools |
| Stochastic SAA optimum and seeded selection | `patient_assignment.proxy_policy_solver.solve` (with `optimum`, `seed`) | distribution pools |
| SAA objective | `patient_assignment.proxy_policy_solver.objective` | pool objective values |
| Feasibility / load evaluation | `patient_assignment.optimization.evaluate` (raises on membership/capacity violation), `experiment_solver.actual` | `dr.or_bridge.check_allocation` |
| Regret and agreement | `patient_assignment.proxy_solver.evaluate` (`proxy_regret`, `exact_assignment_pct`) | `dr.evaluate` |
| Exact agreement / QWK | `patient_assignment.proxy_inputs.agreement` | `dr.evaluate` |
| Gurobi session and settings | `optimization.SolverSession`, `proxy_solver.ProxySettings` | `dr.or_bridge.Solver` |

Existing seeding: selection seeds `sample_seed .. sample_seed + n - 1` (default `sample_seed` 42),
reference allocation seed 1001, Gurobi algorithm seed 0, absolute certification tolerance 1e-7.
SAA defaults: the first values of the existing grid (20 scenarios, scenario seed 42, temperature 1).
The scenario stream is keyed by (model, chunk), as in the existing code, so all rungs share
common random numbers.

## Every constraint of the existing OR model (`complexity-range` objective)

From `optimization.add_allocation` and `experiment_solver.nominal`:

1. `sum_j x_ij = 1` for every new client i (each client assigned to exactly one clinician).
2. `initial_count_j + sum_i x_ij <= capacity_j` for every clinician j. `capacity` is the **final**
   patient-count limit, so the new-client limit is `cap_j = capacity_j - initial_count_j`. The
   prompts state `cap_j` this way.
3. `x_ij ∈ {0, 1}`.

Objective: minimize `max_j L_j - min_j L_j`, where `L_j = initial_complexity_j + sum_i s_i x_ij`
(linearized with upper/lower epigraph variables that carry no extra restriction). No other
constraint exists. The existing `lexicographic` option (count range first) is **not** used,
because it is not the problem shown at rung 6. `dr` refuses any objective other than
`complexity-range`. If a constraint is ever added, add it to `dr/prompts.py::RESTRICTIONS`,
to `prompts/block_formulation.txt`, and to `dr/or_bridge.check_allocation`.

## Deviations from the specification that need a decision

1. **Load profiles were rescaled.** The existing profiles (`configs/proxy_profiles.yaml`) use the
   3-class scale (weights 1 to 3; initial complexities 10 to 30 for 10 clients). The existing
   validator rejects them against class scores 1.0 to 1.8. `config/load_profiles.yaml` keeps the
   identifiers, clinicians, counts and capacities and maps each complexity with
   `count + 0.4 * (old - count)`. Balanced: 14/14/14. Unbalanced: 12/14/16. Highly unbalanced:
   10/14/18. Capacity is 25 final, so each clinician can take 15 new clients.
2. **Notes are not dated.** The existing workbook has one row per client with `SP text` and
   `assessment text`, and no dated notes. The ICF prompt receives the existing source-labeled
   combined note. The template heading "(in date order)" is kept verbatim.
3. **The cohort block drops "The target client is one of them."** in allocation and tie-break
   prompts, which have no target client. Rating prompts keep it.
4. **De-identification gate.** `dr.chunks` requires `--deidentified-confirmed`, matching the
   existing inference commands. `dr.icf` checks it.
5. **Environmental factors are rendered with a magnitude word**, for example
   `barrier, moderate (2, known).`, matching the body-function format.

## ICF summary recomputation rules (`dr/icf_profile.recompute_summary`)

- `areas_of_need`: body functions with status known/queried and qualifier 1, 2, 3, 4 or 8.
- `max_severity_qualifier`: the highest of 1 to 4 among areas of need; 8 when only unspecified
  severities are present; 0 when there are no areas of need.
- `developmental_factors_present`: known if any factor is known, else queried if any is queried,
  else none.
- `activity_or_participation_limited`: yes if activity or participation is known/queried with
  qualifier 1 to 4 or 8.
- `caregiver_capacity_impacted`: yes if caregiver capacity is a known/queried barrier with
  magnitude other than 0.
- `environmental_or_personal_factors_impacting_care`: yes if home/community or service-access
  factors are a known/queried barrier with magnitude other than 0, or if the behavioural profile
  is known/queried.
- In all three yes/no fields: "no" if any contributing field is documented and none is
  impacting; otherwise "not_documented". Silence is never turned into "no".
- `documentation_gaps` is a judgment, so it is kept from the model.

The record stores both summaries: the model's under `parsed.summary` and the recomputed one under
`summary_recomputed`. Disagreements are flagged as `summary_recomputed_differs:<fields>`.
