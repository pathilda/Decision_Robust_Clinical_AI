"""Certified optimal sets, random linear selection, exact envelopes and minimax ties.

All ranges in exported data are recomputed from allocations. Integer objective levels
are retained by restrictions below the next level; continuous levels use the explicit
absolute optimality tolerance. No objective-weight scalarization is used.
"""

import random
from dataclasses import dataclass
from decimal import Decimal

from .inputs import Clinician
from .optimization import (
    SolverSettings,
    add_allocation,
    constrain_range,
    evaluate,
    extract_assignment,
    stage_diagnostics,
)


@dataclass(frozen=True)
class ExperimentSettings(SolverSettings):
    mip_gap: float = 0.0

    def parameters(self):
        return {**super().parameters(), "MIPGap": self.mip_gap}


class IncompleteSolve(ValueError):
    def __init__(self, stage, diagnostics):
        self.stage, self.diagnostics = stage, diagnostics
        super().__init__(
            f"Uncertified {stage}: {diagnostics['status']} (bound={diagnostics.get('bound')})"
        )


def operational(instance):
    cs = [
        Clinician(c["id"], c["initial_count"], Decimal(str(c["initial_complexity"])))
        for c in instance["clinicians"]
    ]
    return cs, {c["id"]: c["capacity"] for c in instance["clinicians"]}


def actual(instance, weights, assignment):
    cs, capacity = operational(instance)
    rows, dn, g = evaluate(assignment, instance["patients"], cs, weights, capacity)
    return {"loads": rows, "count_range": dn, "range": float(g)}


def stress(instance, vectors, assignment):
    return max((actual(instance, v["weights"], assignment)["range"] for v in vectors), default=None)


def allocation(model, session, instance, weights, prefix="x"):
    cs, capacity = operational(instance)
    return add_allocation(model, session.gp, instance["patients"], cs, weights, capacity, prefix)


def assignment(x, instance):
    return extract_assignment(x, instance["patients"], operational(instance)[0])


def load_expressions(session, instance, weights, x):
    return {
        c["id"]: c["initial_complexity"]
        + session.gp.quicksum(weights[p] * x[p, c["id"]] for p in instance["patients"])
        for c in instance["clinicians"]
    }


def range_variable(model, expressions, name):
    upper = model.addVar(lb=0, name=name + "_upper")
    lower = model.addVar(lb=0, name=name + "_lower")
    gap = model.addVar(lb=0, name=name + "_range")
    for expression in expressions.values():
        model.addConstr(upper >= expression)
        model.addConstr(lower <= expression)
    model.addConstr(gap == upper - lower)
    return gap


def optimize(model, session, tolerance):
    model.optimize()
    status = stage_diagnostics(model, session.gp)
    # OPTIMAL can mean merely "within configured MIPGap"; require absolute certification.
    status["certified"] = bool(
        status["optimal"]
        and status["has_incumbent"]
        and status["bound"] is not None
        and abs(status["objective"] - status["bound"]) <= tolerance
    )
    status["certification_tolerance"] = tolerance
    return status


def require(status, stage):
    if not status["certified"]:
        raise IncompleteSolve(stage, status)


def nominal(session, instance, weights, mode, tolerance):
    stages = []
    with session.model() as model:
        x, counts, loads = allocation(model, session, instance, weights)
        dn = None
        if mode == "lexicographic":
            d = range_variable(model, counts, "count")
            model.setObjective(d, session.gp.GRB.MINIMIZE)
            status = optimize(model, session, tolerance)
            stages.append({"stage": "count", **status})
            require(status, "count optimum")
            dn = actual(instance, weights, assignment(x, instance))["count_range"]
            constrain_range(model, counts, dn)
        g = range_variable(model, loads, "nominal")
        model.setObjective(g, session.gp.GRB.MINIMIZE)
        status = optimize(model, session, tolerance)
        stages.append({"stage": "complexity", **status})
        require(status, "nominal optimum")
        chosen = assignment(x, instance)
        result = actual(instance, weights, chosen)
        if abs(result["range"] - status["objective"]) > tolerance:
            raise ValueError("Nominal solver objective differs from evaluated range")
        return {
            "g": result["range"],
            "d": dn,
            "assignment": chosen,
            "stages": stages,
            "certified": True,
            "mode": mode,
        }


def restricted(model, session, instance, weights, optimum, tolerance, slack=0, prefix="x"):
    if not optimum["certified"]:
        raise ValueError("Original optimum is not certified")
    x, counts, loads = allocation(model, session, instance, weights, prefix)
    if optimum["d"] is not None:
        constrain_range(model, counts, optimum["d"])
    gap = range_variable(model, loads, prefix + "_nominal")
    model.addConstr(gap <= optimum["g"] + tolerance + slack)
    return x


def verify(instance, weights, chosen, optimum, tolerance, slack=0):
    values = actual(instance, weights, chosen)
    # Allow only a small feasibility margin beyond the explicitly saved tolerance.
    if values["range"] > optimum["g"] + tolerance + slack + 1e-8:
        raise ValueError("Selected assignment violates nominal range restriction")
    if optimum["d"] is not None and values["count_range"] > optimum["d"]:
        raise ValueError("Selected assignment violates count-optimal restriction")
    return values


def random_objective(session, instance, x, seed):
    rng = random.Random(seed)
    return session.gp.quicksum(
        rng.random() * x[p, c["id"]] for p in instance["patients"] for c in instance["clinicians"]
    )


def random_selection(session, instance, weights, optimum, tolerance, seed):
    with session.model() as model:
        x = restricted(model, session, instance, weights, optimum, tolerance)
        model.setObjective(random_objective(session, instance, x, seed), session.gp.GRB.MINIMIZE)
        status = optimize(model, session, tolerance)
        require(status, "random optimal selection")
        chosen = assignment(x, instance)
        verify(instance, weights, chosen, optimum, tolerance)
        return {"assignment": chosen, "stages": [status], "seed": seed}


def matched_selection(session, instance, weights, benchmark, optimum, benchmark_optimum, tolerance):
    gp = session.gp
    with session.model() as model:
        x = restricted(model, session, instance, weights, optimum, tolerance, prefix="candidate")
        y = restricted(
            model, session, instance, benchmark, benchmark_optimum, tolerance, prefix="benchmark"
        )
        same = model.addVars(
            instance["patients"],
            [c["id"] for c in instance["clinicians"]],
            vtype=gp.GRB.BINARY,
            name="same",
        )
        for key in same:
            model.addConstr(same[key] <= x[key])
            model.addConstr(same[key] <= y[key])
            model.addConstr(same[key] >= x[key] + y[key] - 1)
        model.setObjective(len(instance["patients"]) - gp.quicksum(same.values()), gp.GRB.MINIMIZE)
        status = optimize(model, session, tolerance)
        require(status, "joint matching")
        a, b = assignment(x, instance), assignment(y, instance)
        verify(instance, weights, a, optimum, tolerance)
        verify(instance, benchmark, b, benchmark_optimum, tolerance)
        return {"assignment": a, "benchmark_assignment": b, "stages": [status]}


def robust_selection(session, instance, weights, optimum, vectors, tolerance, slack, seeds):
    with session.model() as model:
        x = restricted(model, session, instance, weights, optimum, tolerance, slack)
        worst = model.addVar(lb=0, name="worst_scenario_range")
        for index, vector in enumerate(vectors):
            gap = range_variable(
                model,
                load_expressions(session, instance, vector["weights"], x),
                f"scenario_{index}",
            )
            model.addConstr(worst >= gap)
        model.setObjective(worst, session.gp.GRB.MINIMIZE)
        status = optimize(model, session, tolerance)
        require(status, "robust minimax optimum")
        best = stress(instance, vectors, assignment(x, instance))
        if abs(best - status["objective"]) > tolerance:
            raise ValueError("Robust objective differs from fixed-assignment stress evaluation")
        model.addConstr(worst <= best + tolerance)
        results = []
        for seed in seeds:
            model.reset()  # no incumbent from an earlier sample determines residual selection
            model.setObjective(
                random_objective(session, instance, x, seed), session.gp.GRB.MINIMIZE
            )
            tertiary = optimize(model, session, tolerance)
            require(tertiary, "robust random tertiary selection")
            chosen = assignment(x, instance)
            verify(instance, weights, chosen, optimum, tolerance, slack)
            if stress(instance, vectors, chosen) > best + tolerance + 1e-8:
                raise ValueError("Tertiary selection violates robust optimum")
            results.append(
                {
                    "assignment": chosen,
                    "stages": [status, tertiary],
                    "seed": seed,
                    "robust_optimum": best,
                }
            )
        return results


def regret_envelope(session, instance, weights, benchmark, optimum, benchmark_optimum, tolerance):
    """Maximize ordered load differences, never a free range epigraph."""
    statuses, witnesses, values = [], {}, {}
    names = [c["id"] for c in instance["clinicians"]]
    pairs = [(j, k) for j in names for k in names if j != k]
    with session.model() as model:
        x = restricted(model, session, instance, weights, optimum, tolerance)
        loads = load_expressions(session, instance, benchmark, x)
        gap = range_variable(model, loads, "benchmark")
        model.setObjective(gap, session.gp.GRB.MINIMIZE)
        status = optimize(model, session, tolerance)
        statuses.append({"stage": "minimum", **status})
        if status["has_incumbent"]:
            witnesses["minimum"] = assignment(x, instance)
            values["minimum"] = actual(instance, benchmark, witnesses["minimum"])["range"]
        maximum = -float("inf")
        for j, k in pairs:
            model.setObjective(loads[j] - loads[k], session.gp.GRB.MAXIMIZE)
            status = optimize(model, session, tolerance)
            statuses.append({"stage": f"maximum:{j}:{k}", **status})
            if status["has_incumbent"]:
                chosen = assignment(x, instance)
                g = actual(instance, benchmark, chosen)["range"]
                if g > maximum:
                    maximum, witnesses["maximum"] = g, chosen
        if not pairs and "minimum" in witnesses:
            maximum, witnesses["maximum"] = 0.0, witnesses["minimum"]
        if maximum != -float("inf"):
            values["maximum"] = maximum
    complete = all(s["certified"] for s in statuses)
    return {
        "complete": complete,
        "statuses": statuses,
        "witnesses": witnesses,
        "minimum_range": values.get("minimum"),
        "maximum_range": values.get("maximum"),
        "minimum_regret": values.get("minimum", benchmark_optimum["g"]) - benchmark_optimum["g"]
        if "minimum" in values
        else None,
        "maximum_regret": values.get("maximum", benchmark_optimum["g"]) - benchmark_optimum["g"]
        if "maximum" in values
        else None,
        "minimum_range_lower_bound": statuses[0]["bound"],
        "maximum_range_upper_bound": (
            max(s["bound"] for s in statuses[1:])
            if pairs and all(s["bound"] is not None for s in statuses[1:])
            else (0 if not pairs else None)
        ),
    }
