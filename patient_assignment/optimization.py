"""Two certified lexicographic stages. Gurobi is imported only on solver use."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

VERIFICATION_TOLERANCE = 1e-6


def exact_decimal_sum(*values):
    values = [v if isinstance(v, Decimal) else Decimal(v) for v in values]
    with localcontext() as context:
        context.prec = (
            max(len(v.as_tuple().digits) + abs(v.as_tuple().exponent) for v in values) + 20
        )
        return sum(values, Decimal(0))


class SolverDependencyError(RuntimeError):
    pass


def gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise SolverDependencyError(
            "Gurobi Python API unavailable; install with python -m pip install gurobipy (or .[optimization])"
        ) from exc
    return gp


@dataclass(frozen=True)
class SolverSettings:
    time_limit: float | None = None
    threads: int = 1
    seed: int = 0

    def validate(self):
        if self.threads < 1 or self.seed < 0 or self.seed > 2_000_000_000:
            raise ValueError("Threads must be positive; solver seed must be in [0, 2000000000]")
        if self.time_limit is not None and (
            not math.isfinite(self.time_limit) or self.time_limit <= 0
        ):
            raise ValueError("Time limit must be finite and positive")

    def parameters(self):
        result = {
            "Threads": self.threads,
            "Seed": self.seed,
            "MIPGap": 0.0,
            "MIPGapAbs": 0.0,
            "FeasibilityTol": 1e-9,
            "IntFeasTol": 1e-9,
            "OptimalityTol": 1e-9,
            "OutputFlag": 0,
        }
        if self.time_limit is not None:
            result["TimeLimit"] = self.time_limit
        return result


class SolverSession:
    """Quiet environment prevents startup license details from entering reports."""

    def __init__(self, settings):
        self.settings = settings
        self.gp = gurobi()
        self.env = None

    def __enter__(self):
        try:
            self.env = self.gp.Env(empty=True)
            self.env.setParam("OutputFlag", 0)
            self.env.start()
        except self.gp.GurobiError as exc:
            if self.env is not None:
                self.env.dispose()
            raise SolverDependencyError(
                f"Gurobi initialization/license failed (code {exc.errno}); check your existing license configuration locally"
            ) from exc
        return self

    def __exit__(self, *_):
        self.env.dispose()

    def model(self):
        model = self.gp.Model(env=self.env)
        for key, value in self.settings.parameters().items():
            model.setParam(key, value)
        return model


def stage_diagnostics(model, gp):
    # Never query solution attributes without an incumbent.
    has_incumbent = model.SolCount > 0
    names = {
        getattr(gp.GRB, name): name
        for name in (
            "LOADED",
            "OPTIMAL",
            "INFEASIBLE",
            "INF_OR_UNBD",
            "UNBOUNDED",
            "CUTOFF",
            "ITERATION_LIMIT",
            "NODE_LIMIT",
            "TIME_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "NUMERIC",
            "SUBOPTIMAL",
            "USER_OBJ_LIMIT",
        )
    }

    def finite(value):
        return float(value) if math.isfinite(value) else None

    return {
        "status": names.get(model.Status, str(model.Status)),
        "status_code": model.Status,
        "optimal": model.Status == gp.GRB.OPTIMAL,
        "has_incumbent": has_incumbent,
        "objective": finite(model.ObjVal) if has_incumbent else None,
        "bound": finite(model.ObjBound),
        "gap": finite(model.MIPGap) if has_incumbent else None,
        "runtime": float(model.Runtime),
    }


def add_allocation(model, gp, patients, clinicians, weights, capacity, prefix):
    x = model.addVars(patients, [c.name for c in clinicians], vtype=gp.GRB.BINARY, name=prefix)
    counts, loads = {}, {}
    for p in patients:
        model.addConstr(gp.quicksum(x[p, c.name] for c in clinicians) == 1)
    for c in clinicians:
        counts[c.name] = c.initial_count + gp.quicksum(x[p, c.name] for p in patients)
        loads[c.name] = float(c.initial_severity) + gp.quicksum(
            weights[p] * x[p, c.name] for p in patients
        )
        model.addConstr(counts[c.name] <= capacity)
    return x, counts, loads


def constrain_range(model, expressions, bound):
    for left in expressions:
        for right in expressions:
            if left != right:
                model.addConstr(expressions[left] - expressions[right] <= bound)


def extract_assignment(x, patients, clinicians):
    result = {}
    for p in patients:
        selected = [c.name for c in clinicians if x[p, c.name].X > 0.5]
        if len(selected) != 1:
            raise ValueError(f"Invalid solver one-hot assignment for {p!r}")
        result[p] = selected[0]
    return result


def evaluate(assignment, patients, clinicians, weights, capacity):
    if set(assignment) != set(patients):
        raise ValueError("Allocation membership does not equal requested chunk")
    names = {c.name for c in clinicians}
    if set(assignment.values()) - names:
        raise ValueError("Unknown assigned clinician")
    rows = []
    for c in clinicians:
        selected = [p for p in patients if assignment[p] == c.name]
        assigned_severity = sum(weights[p] for p in selected)
        final_count = c.initial_count + len(selected)
        if final_count > capacity:
            raise ValueError("Exported allocation violates capacity")
        rows.append(
            {
                "clinician_id": c.name,
                "initial_count": c.initial_count,
                "initial_severity": str(c.initial_severity),
                "assigned_count": len(selected),
                "assigned_severity": assigned_severity,
                "final_count": final_count,
                "final_severity": str(exact_decimal_sum(c.initial_severity, assigned_severity)),
                "capacity": capacity,
            }
        )
    count_range = max(r["final_count"] for r in rows) - min(r["final_count"] for r in rows)
    severity_range = exact_decimal_sum(
        max(Decimal(r["final_severity"]) for r in rows),
        min(Decimal(r["final_severity"]) for r in rows).copy_negate(),
    )
    return rows, count_range, severity_range


@dataclass
class Optimum:
    stages: dict = field(default_factory=dict)
    assignment: dict | None = None
    count_range: int | None = None
    severity_range: Decimal | None = None

    @property
    def certified(self):
        return bool(
            self.stages.get("stage1", {}).get("optimal")
            and self.stages.get("stage2", {}).get("optimal")
        )


def verify_optimal_set(assignment, patients, clinicians, weights, capacity, optimum):
    loads, dn, db = evaluate(assignment, patients, clinicians, weights, capacity)
    if dn != optimum.count_range or abs(db - optimum.severity_range) > Decimal(
        str(VERIFICATION_TOLERANCE)
    ):
        raise ValueError("Selected assignment does not attain both certified component optima")
    return loads, dn, db


def solve_lexicographic(session, patients, clinicians, weights, capacity):
    gp = session.gp
    result = Optimum()
    with session.model() as model:
        x, counts, loads = add_allocation(model, gp, patients, clinicians, weights, capacity, "x")
        dn = model.addVar(lb=0, name="count_range")
        constrain_range(model, counts, dn)
        model.setObjective(dn, gp.GRB.MINIMIZE)
        model.optimize()
        result.stages["stage1"] = stage_diagnostics(model, gp)
        if result.stages["stage1"]["has_incumbent"]:
            result.assignment = extract_assignment(x, patients, clinicians)
            _, result.count_range, result.severity_range = evaluate(
                result.assignment, patients, clinicians, weights, capacity
            )
        if not result.stages["stage1"]["optimal"]:
            return result
        if abs(result.stages["stage1"]["objective"] - result.count_range) > VERIFICATION_TOLERANCE:
            raise ValueError("Certified count objective differs from actual assignment range")
        constrain_range(model, counts, result.count_range)
        db = model.addVar(lb=0, name="severity_range")
        constrain_range(model, loads, db)
        model.setObjective(db, gp.GRB.MINIMIZE)
        model.optimize()
        result.stages["stage2"] = stage_diagnostics(model, gp)
        if result.stages["stage2"]["has_incumbent"]:
            result.assignment = extract_assignment(x, patients, clinicians)
            _, actual_dn, result.severity_range = evaluate(
                result.assignment, patients, clinicians, weights, capacity
            )
            if actual_dn != result.count_range:
                raise ValueError("Stage 2 violated the certified count optimum")
            if (
                result.certified
                and abs(float(result.severity_range) - result.stages["stage2"]["objective"])
                > VERIFICATION_TOLERANCE
            ):
                raise ValueError(
                    "Certified severity objective differs from actual assignment range"
                )
        return result
