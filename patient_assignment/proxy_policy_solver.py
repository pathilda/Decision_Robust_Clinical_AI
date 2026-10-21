"""Exact common-assignment SAA and cardinality-budget discrete robust policies."""

import math
import random
from collections import Counter

from . import experiment_solver as s
from .experiment_inputs import digest
from .optimization import constrain_range
from .proxy_solver import canonical, stream_seed

ALGORITHM = "proxy-uncertainty-v1"


def scenarios(probabilities, weights, model, chunk, seed, count, temperature=1.0, role="training"):
    """Stable independent patient marginals; profile and selection seed are absent."""
    patients = sorted(probabilities)
    rng = random.Random(int(digest([ALGORITHM, role, model, chunk, patients, seed]), 16))
    distributions = {}
    for p in patients:
        logs = [math.log(v) / temperature if v else -math.inf for v in probabilities[p]]
        peak = max(logs)
        values = [math.exp(v - peak) for v in logs]
        distributions[p] = values
    vectors = [
        tuple(rng.choices(weights, distributions[p])[0] for p in patients) for _ in range(count)
    ]
    counts = Counter(vectors)
    return {
        "vectors": [dict(zip(patients, v)) for v in counts],
        "multiplicities": list(counts.values()),
        "count": count,
        "fingerprint": digest(vectors),
        "generator": ALGORITHM,
        "role": role,
        "seed": seed,
        "temperature": temperature,
    }


def support(nominal, probabilities, weights, threshold):
    return {
        p: sorted({nominal[p], *(w for w, q in zip(weights, probabilities[p]) if q >= threshold)})
        for p in sorted(nominal)
    }


def separate(instance, nominal, supports, budget, chosen):
    """Exact max range: max ordered pair, then largest positive patient gains."""
    best = s.actual(instance, nominal, chosen)["range"]
    witness = dict(nominal)
    for j in instance["clinicians"]:
        for k in instance["clinicians"]:
            if j["id"] == k["id"]:
                continue
            candidate = dict(nominal)
            gains = []
            for p in sorted(nominal):
                sign = int(chosen[p] == j["id"]) - int(chosen[p] == k["id"])
                value = max(supports[p], key=lambda v: sign * (v - nominal[p]))
                gain = sign * (value - nominal[p])
                if gain > 0:
                    gains.append((gain, p, value))
            for _, p, value in sorted(gains, key=lambda item: (-item[0], item[1]))[:budget]:
                candidate[p] = value
            value = s.actual(instance, candidate, chosen)["range"]
            if value > best:
                best, witness = value, candidate
    return best, witness


def objective(instance, nominal, problem, chosen):
    if problem["policy"] == "stochastic":
        data = problem["scenarios"]
        return (
            math.fsum(
                s.actual(instance, v, chosen)["range"] * n
                for v, n in zip(data["vectors"], data["multiplicities"])
            )
            / data["count"]
        )
    return separate(instance, nominal, problem["supports"], problem["budget"], chosen)[0]


def solve(session, instance, nominal, problem, mode, tolerance, optimum=None, seed=None):
    """Certify policy optimum or paired random optimum, separating tertiary solutions too."""
    instance = canonical(instance)
    stochastic = problem["policy"] == "stochastic"
    vectors = (
        problem["scenarios"]["vectors"]
        if stochastic
        else (optimum["cuts"] if optimum else [nominal])
    )
    vectors = [dict(v) for v in vectors]
    stages = []
    with session.model() as model:
        x, counts, _ = s.allocation(model, session, instance, nominal)
        count_optimum = optimum["d"] if optimum else None
        if mode == "lexicographic" and optimum is None:
            gap = s.range_variable(model, counts, "count")
            model.setObjective(gap, session.gp.GRB.MINIMIZE)
            status = s.optimize(model, session, tolerance)
            s.require(status, "count optimum")
            stages.append(status)
            count_optimum = s.actual(instance, nominal, s.assignment(x, instance))["count_range"]
        if count_optimum is not None:
            constrain_range(model, counts, count_optimum)
        t = model.addVar(lb=0, name="policy_objective")

        def cut(vector, index):
            gap = s.range_variable(
                model, s.load_expressions(session, instance, vector, x), f"scenario_{index}"
            )
            if not stochastic:
                model.addConstr(t >= gap)
            return gap

        gaps = [cut(v, i) for i, v in enumerate(vectors)]
        if stochastic:
            data = problem["scenarios"]
            model.addConstr(
                t
                == session.gp.quicksum(
                    g * n / data["count"] for g, n in zip(gaps, data["multiplicities"])
                )
            )
        if optimum is not None:
            model.addConstr(t <= optimum["g"])  # zero intended degradation
            model.setObjective(
                s.random_objective(session, instance, x, stream_seed(instance, seed)),
                session.gp.GRB.MINIMIZE,
            )
        else:
            model.setObjective(t, session.gp.GRB.MINIMIZE)
        iterations = 0
        while True:
            iterations += 1
            status = s.optimize(model, session, tolerance)
            stages.append(status)
            s.require(status, "policy selection" if optimum else "policy master")
            chosen = s.assignment(x, instance)
            value = objective(instance, nominal, problem, chosen)
            target = optimum["g"] if optimum else status["objective"]
            if value <= target + tolerance:
                if optimum is None and abs(value - status["bound"]) > tolerance:
                    raise ValueError(
                        "Original policy objective is not within tolerance of the global bound"
                    )
                if abs(value - target) > tolerance:
                    raise ValueError("Recomputed policy objective differs from certified optimum")
                if (
                    count_optimum is not None
                    and s.actual(instance, nominal, chosen)["count_range"] != count_optimum
                ):
                    raise ValueError("Policy assignment is not count-optimal")
                return {
                    "g": value,
                    "d": count_optimum,
                    "assignment": chosen,
                    "certified": True,
                    "stages": stages,
                    "cuts": vectors,
                    "separation_iterations": iterations if not stochastic else 0,
                    "termination_reason": "certified_global_gap"
                    if optimum is None
                    else "verified_optimal_selection",
                    "mode": mode,
                }
            if stochastic:
                raise ValueError("SAA objective verification failed")
            _, witness = separate(instance, nominal, problem["supports"], problem["budget"], chosen)
            if witness in vectors:
                raise ValueError("Robust separation stalled; cannot certify zero degradation")
            cut(witness, len(vectors))
            vectors.append(witness)


def holdout_metrics(instance, data, chosen, alpha):
    values = sorted(
        v
        for vector, n in zip(data["vectors"], data["multiplicities"])
        for v in [s.actual(instance, vector, chosen)["range"]] * n
    )
    n = len(values)
    mean = math.fsum(values) / n

    def quantile(q):
        pos = (n - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, n - 1)] - values[lo]) * (pos - lo)

    # Integral of empirical quantile over [alpha,1], including fractional boundary mass.
    tail = n * (1 - alpha)
    whole = int(tail)
    cvar = (sum(values[n - whole :]) + (tail - whole) * values[max(0, n - whole - 1)]) / tail
    return {
        "n_holdout_scenarios": n,
        "holdout_fingerprint": data["fingerprint"],
        "holdout_seed": data["seed"],
        "holdout_mean": mean,
        "holdout_std": math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n),
        "holdout_max": max(values),
        "holdout_median": quantile(0.5),
        "holdout_q90": quantile(0.9),
        "holdout_q95": quantile(0.95),
        "holdout_cvar": cvar,
        "cvar_alpha": alpha,
    }
