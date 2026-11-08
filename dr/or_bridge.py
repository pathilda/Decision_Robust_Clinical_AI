"""Thin wrappers around the existing OR code (never modified; spec Section 13).

Reused, unchanged:
- patient_assignment.proxy_inputs.profiles           load-profile loading and validation
- patient_assignment.experiment_solver.nominal       certified complexity-range optimum
- patient_assignment.proxy_solver.select             seeded random-objective optimal selection
- patient_assignment.proxy_policy_solver.scenarios   SAA scenario sampling from probabilities
- patient_assignment.proxy_policy_solver.solve       certified SAA optimum and seeded selection
- patient_assignment.experiment_solver.actual        evaluated loads/range (existing evaluate)
- patient_assignment.proxy_solver.evaluate           proxy regret and assignment agreement
"""

from __future__ import annotations

from decimal import Decimal

from patient_assignment import experiment_solver as solver
from patient_assignment import proxy_policy_solver as policy_solver
from patient_assignment import proxy_solver
from patient_assignment.inputs import Clinician
from patient_assignment.optimization import SolverSession, evaluate
from patient_assignment.proxy_inputs import profiles as load_existing_profiles

from .common import CONFIG_DIR, PROFILE_IDS, classes_config, study_config

OBJECTIVE = "complexity-range"


def optimization_settings():
    opt = dict(study_config()["optimization"])
    if opt["objective"] != OBJECTIVE:
        raise ValueError(
            "Only the existing complexity-range objective is supported: it is the formulation "
            "shown to the LLM at rung 6, and the prompts list exactly its constraints"
        )
    return opt


def selection_seeds():
    opt = optimization_settings()
    return list(range(int(opt["sample_seed"]), int(opt["sample_seed"]) + int(opt["n_seeds"])))


def load_profiles():
    path = CONFIG_DIR / study_config()["load_profiles"]
    weights = list(classes_config()["scores"].values())
    result = load_existing_profiles(path, weights)
    if tuple(result) != PROFILE_IDS:
        raise ValueError(f"Load profiles must be exactly {PROFILE_IDS}")
    return result


def new_client_limit(clinician):
    return int(clinician["capacity"]) - int(clinician["initial_count"])


def instance(chunk_id, patients, profile_id, clinicians):
    return proxy_solver.canonical(
        {
            "chunk_id": str(chunk_id),
            "profile_id": profile_id,
            "patients": [str(p) for p in patients],
            "clinicians": clinicians,
        }
    )


def feasible_capacity(patients, clinicians):
    return len(patients) <= sum(new_client_limit(c) for c in clinicians)


# ----------------------------------------------------------------------------- feasibility
def check_allocation(pairs, client_ids, clinicians):
    """Check every constraint of the existing OR model for an LLM-proposed assignment.

    Constraints (existing complexity-range model): each client assigned to exactly one
    clinician; each clinician's final count (initial_count + new) <= capacity; binary x.
    Returns (feasible, violations, assignment-or-None).
    """
    violations = []
    clients = {str(c) for c in client_ids}
    names = {c["id"] for c in clinicians}
    seen = {}
    for pair in pairs:
        cid, jid = str(pair["client_id"]), str(pair["clinician_id"])
        if cid not in clients:
            violations.append(f"unknown_client:{cid}")
            continue
        if jid not in names:
            violations.append(f"unknown_clinician:{jid}")
            continue
        if cid in seen:
            violations.append(f"client_assigned_more_than_once:{cid}")
            continue
        seen[cid] = jid
    for cid in sorted(clients - set(seen)):
        violations.append(f"client_not_assigned:{cid}")
    for c in clinicians:
        taken = sum(1 for j in seen.values() if j == c["id"])
        if taken > new_client_limit(c):
            violations.append(f"capacity_exceeded:{c['id']}:{taken}>{new_client_limit(c)}")
    if violations:
        return False, violations, None
    # Cross-check with the existing evaluator (raises on membership/capacity violations).
    existing = [
        Clinician(c["id"], c["initial_count"], Decimal(str(c["initial_complexity"])))
        for c in clinicians
    ]
    evaluate(
        seen,
        sorted(clients),
        existing,
        {p: 0 for p in clients},
        {c["id"]: c["capacity"] for c in clinicians},
    )
    return True, [], seen


def loads(inst, weights, assignment):
    return solver.actual(inst, weights, assignment)


# ----------------------------------------------------------------------------- weights
def label_weights(classes):
    scores = classes_config()["scores"]
    return {p: scores[int(c)] for p, c in classes.items()}


def argmax_class(probabilities):
    ids = classes_config()["ids"]
    values = [probabilities[str(i)] for i in ids]
    return ids[max(range(len(ids)), key=lambda k: (values[k], -k))]


# ----------------------------------------------------------------------------- solving
class Solver:
    """One lazily started Gurobi session reused for every solve of a command."""

    def __init__(self):
        opt = optimization_settings()
        self.tolerance = float(opt["optimality_tolerance"])
        self.settings = proxy_solver.ProxySettings(
            time_limit=opt.get("time_limit"),
            threads=int(opt["threads"]),
            seed=int(opt["solver_seed"]),
        )
        self.settings.validate()
        self.opt = opt
        self._session = None

    def session(self):
        if self._session is None:
            self._session = SolverSession(self.settings).__enter__()
        return self._session

    def close(self):
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def parameters(self):
        return self.settings.parameters()

    # -- deterministic model (label, score) ---------------------------------------------
    def deterministic(self, inst, weights, seeds):
        optimum = solver.nominal(self.session(), inst, weights, OBJECTIVE, self.tolerance)
        solutions = []
        for seed in seeds:
            chosen = proxy_solver.select(
                self.session(), inst, weights, optimum, self.tolerance, seed
            )
            value = solver.actual(inst, weights, chosen["assignment"])["range"]
            solutions.append({"seed": seed, "assignment": chosen["assignment"], "objective": value})
        return {"policy": "deterministic", "optimum": optimum["g"], "solutions": solutions}

    # -- stochastic model (distribution) ------------------------------------------------
    def stochastic(self, inst, probabilities, stream_id, seeds):
        """probabilities: {patient: {"1": p1, ..., "5": p5}} (normalized)."""
        cfg = classes_config()
        ids = cfg["ids"]
        class_weights = [cfg["scores"][i] for i in ids]
        vectors = {p: [float(probabilities[p][str(i)]) for i in ids] for p in inst["patients"]}
        nominal = {p: cfg["scores"][argmax_class(probabilities[p])] for p in inst["patients"]}
        scenarios = policy_solver.scenarios(
            vectors,
            class_weights,
            stream_id,
            inst["chunk_id"],
            int(self.opt["scenario_seed"]),
            int(self.opt["scenario_count"]),
            float(self.opt["temperature"]),
        )
        problem = {"policy": "stochastic", "scenarios": scenarios}
        optimum = policy_solver.solve(
            self.session(), inst, nominal, problem, OBJECTIVE, self.tolerance
        )
        solutions = []
        for seed in seeds:
            chosen = policy_solver.solve(
                self.session(), inst, nominal, problem, OBJECTIVE, self.tolerance, optimum, seed
            )
            value = policy_solver.objective(inst, nominal, problem, chosen["assignment"])
            solutions.append({"seed": seed, "assignment": chosen["assignment"], "objective": value})
        return {
            "policy": "stochastic",
            "optimum": optimum["g"],
            "solutions": solutions,
            "scenario_count": scenarios["count"],
            "scenario_seed": scenarios["seed"],
            "scenario_fingerprint": scenarios["fingerprint"],
            "scenario_stream_id": stream_id,
            "n_distinct_scenarios": len(scenarios["vectors"]),
        }

    # -- evaluation (existing proxy regret / agreement) ---------------------------------
    def benchmark(self, inst, weights, benchmark_seed):
        optimum = solver.nominal(self.session(), inst, weights, OBJECTIVE, self.tolerance)
        reference = proxy_solver.select(
            self.session(), inst, weights, optimum, self.tolerance, benchmark_seed, "benchmark"
        )
        return optimum, reference["assignment"]


def evaluate_against(inst, benchmark_weights, benchmark_optimum, assignment, reference):
    return proxy_solver.evaluate(inst, benchmark_weights, benchmark_optimum, assignment, reference)


def canonical_assignment(assignment):
    return sorted((str(p), str(c)) for p, c in assignment.items())
