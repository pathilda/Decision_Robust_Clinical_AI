"""Zero-degradation restrictions and paired random linear objectives."""

from dataclasses import dataclass

from . import experiment_solver as solver
from .experiment_inputs import digest, number
from .optimization import SolverSettings


@dataclass(frozen=True)
class ProxySettings(SolverSettings):
    feasibility_tolerance: float = 1e-9
    integer_tolerance: float = 1e-9
    solver_optimality_tolerance: float = 1e-9

    def validate(self):
        super().validate()
        for name in ("feasibility_tolerance", "integer_tolerance", "solver_optimality_tolerance"):
            value = number(getattr(self, name), name, 1e-9)
            if value > 1e-3:
                raise ValueError(f"{name} must be <= 0.001")

    def parameters(self):
        return {
            **super().parameters(),
            "FeasibilityTol": self.feasibility_tolerance,
            "IntFeasTol": self.integer_tolerance,
            "OptimalityTol": self.solver_optimality_tolerance,
        }


def canonical(instance):
    return {k: instance[k] for k in ("chunk_id", "profile_id")} | {
        "patients": sorted(instance["patients"]),
        "clinicians": sorted(instance["clinicians"], key=lambda c: c["id"]),
    }


def stream_seed(instance, seed, purpose="candidate"):
    return int(digest(["proxy-random-v1", canonical(instance), seed, purpose]), 16)


def select(session, instance, weights, optimum, tolerance, seed, purpose="candidate"):
    instance = canonical(instance)
    derived = stream_seed(instance, seed, purpose)
    with session.model() as model:
        # Do not add certification tolerance as intended objective degradation.
        x = solver.restricted(model, session, instance, weights, optimum, 0)
        model.setObjective(
            solver.random_objective(session, instance, x, derived), session.gp.GRB.MINIMIZE
        )
        status = solver.optimize(model, session, tolerance)
        solver.require(status, "random optimal selection")
        chosen = solver.assignment(x, instance)
        values = solver.verify(instance, weights, chosen, optimum, tolerance)
        if abs(values["range"] - optimum["g"]) > tolerance:
            raise ValueError("Selection differs from certified nominal optimum")
        return {
            "assignment": chosen,
            "stages": [status],
            "seed": seed,
            "stream_seed": str(derived),
            "certified": True,
            "purpose": purpose,
        }


def evaluate(instance, benchmark_weights, proxy_optimum, chosen, reference):
    g = solver.actual(instance, benchmark_weights, chosen)["range"]
    pct = (
        100
        * sum(chosen[p] == reference[p] for p in instance["patients"])
        / len(instance["patients"])
    )
    return {
        "proxy_evaluated_range": g,
        "proxy_regret": g - proxy_optimum["g"],
        "exact_assignment_pct": pct,
    }
