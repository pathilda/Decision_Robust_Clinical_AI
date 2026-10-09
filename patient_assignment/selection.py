"""Joint closest-pair matching and independent random objectives on optimal sets."""

import hashlib
import json
import random
from decimal import Decimal

from .optimization import (
    VERIFICATION_TOLERANCE,
    add_allocation,
    constrain_range,
    evaluate,
    exact_decimal_sum,
    extract_assignment,
    stage_diagnostics,
    verify_optimal_set,
)

RANDOM_ALGORITHM = "SHA256(JSON [selection_seed, chunk_id, repeat_id, model_label]) -> Python MT19937; Uniform[0,1) over sorted patient/clinician IDs"


def coefficients(seed, chunk_id, repeat_id, label, patients, clinicians):
    key = json.dumps([seed, chunk_id, repeat_id, label], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest, 16))
    values = {(p, c): rng.random() for p in sorted(patients) for c in sorted(clinicians)}
    encoded = json.dumps(
        [[p, c, v] for (p, c), v in values.items()], ensure_ascii=False, separators=(",", ":")
    )
    return values, {
        "substream_sha256": digest,
        "coefficient_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def add_optimal_allocation(
    model, session, patients, clinicians, weights, capacity, optimum, prefix
):
    if not optimum.certified:
        raise ValueError("Selection requires certified component optima")
    x, counts, loads = add_allocation(
        model, session.gp, patients, clinicians, weights, capacity, prefix
    )
    constrain_range(model, counts, optimum.count_range)
    constrain_range(model, loads, float(optimum.severity_range))
    return x


def match_optima(
    session, patients, clinicians, weights_a, weights_b, capacity, optimum_a, optimum_b
):
    gp = session.gp
    with session.model() as model:
        xa = add_optimal_allocation(
            model, session, patients, clinicians, weights_a, capacity, optimum_a, "A"
        )
        xb = add_optimal_allocation(
            model, session, patients, clinicians, weights_b, capacity, optimum_b, "B"
        )
        y = model.addVars(patients, [c.name for c in clinicians], vtype=gp.GRB.BINARY, name="both")
        for p in patients:
            for c in clinicians:
                key = p, c.name
                model.addConstr(y[key] <= xa[key])
                model.addConstr(y[key] <= xb[key])
                model.addConstr(y[key] >= xa[key] + xb[key] - 1)
        model.setObjective(len(patients) - gp.quicksum(y.values()), gp.GRB.MINIMIZE)
        model.optimize()
        status = stage_diagnostics(model, gp)
        a = b = None
        if status["has_incumbent"]:
            a, b = (
                extract_assignment(xa, patients, clinicians),
                extract_assignment(xb, patients, clinicians),
            )
            verify_optimal_set(a, patients, clinicians, weights_a, capacity, optimum_a)
            verify_optimal_set(b, patients, clinicians, weights_b, capacity, optimum_b)
            changed = sum(a[p] != b[p] for p in patients)
            if abs(changed - status["objective"]) > VERIFICATION_TOLERANCE:
                raise ValueError("Matching objective disagrees with exported pair")
        return a, b, status


def random_optimum(session, patients, clinicians, weights, capacity, optimum, values):
    gp = session.gp
    with session.model() as model:
        x = add_optimal_allocation(
            model, session, patients, clinicians, weights, capacity, optimum, "random"
        )
        model.setObjective(
            gp.quicksum(value * x[key] for key, value in values.items()), gp.GRB.MINIMIZE
        )
        model.optimize()
        status = stage_diagnostics(model, gp)
        assignment = None
        if status["has_incumbent"]:
            assignment = extract_assignment(x, patients, clinicians)
            verify_optimal_set(assignment, patients, clinicians, weights, capacity, optimum)
        return assignment, status


def reference_regret(
    ai_assignment, patients, clinicians, reference_weights, capacity, reference_optimum
):
    _, dn, gap = evaluate(ai_assignment, patients, clinicians, reference_weights, capacity)
    if dn != reference_optimum.count_range:
        raise ValueError("Reference evaluation count optimum differs")
    raw = exact_decimal_sum(gap, reference_optimum.severity_range.copy_negate())
    if raw < -Decimal(str(VERIFICATION_TOLERANCE)):
        raise ValueError(
            "Negative reference regret beyond tolerance; incompatible allocation/inputs"
        )
    return {
        "G_reference": str(reference_optimum.severity_range),
        "AI_reference_gap": str(gap),
        "raw_regret": str(raw),
        "reported_regret": str(max(Decimal(0), raw)),
        "regret_verification_tolerance": VERIFICATION_TOLERANCE,
    }
