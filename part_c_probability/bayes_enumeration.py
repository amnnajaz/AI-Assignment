"""
Part C: Inference by enumeration on the 8-variable incident-response Bayesian network.

    Re -> Bu        At -> Ov        Bu,Ov -> Cr        Cr,At -> Al
    Cr -> Co        Al,Co -> Es

enumeration_ask() implements the textbook ENUMERATION-ASK algorithm: it sums
the full joint distribution (product of CPT entries) over all hidden variables.
The helper `marginals_for_report` prints the intermediate quantities that are
used in the hand-worked solution in the PDF report.

Run:  python bayes_enumeration.py
"""
from itertools import product

# variable order is a valid topological order
VARS = ["Re", "At", "Bu", "Ov", "Cr", "Al", "Co", "Es"]
PARENTS = {
    "Re": [], "At": [],
    "Bu": ["Re"], "Ov": ["At"],
    "Cr": ["Bu", "Ov"],
    "Al": ["Cr", "At"],
    "Co": ["Cr"],
    "Es": ["Al", "Co"],
}
# CPT[var][tuple of parent values] = P(var = True|parents)
CPT = {
    "Re": {(): 0.2},
    "At": {(): 0.1},
    "Bu": {(True,): 0.3, (False,): 0.05},
    "Ov": {(True,): 0.6, (False,): 0.1},
    "Cr": {(True, True): 0.95, (True, False): 0.6, (False, True): 0.4, (False, False): 0.03},
    "Al": {(True, True): 0.99, (True, False): 0.9, (False, True): 0.7, (False, False): 0.05},
    "Co": {(True,): 0.85, (False,): 0.1},
    "Es": {(True, True): 0.95, (True, False): 0.6, (False, True): 0.4, (False, False): 0.01},
}


def prob(var, value, assignment):
    """P(var=value | parents(var)) read from the CPT; parents are taken from `assignment`."""
    p_true = CPT[var][tuple(assignment[p] for p in PARENTS[var])]
    return p_true if value else 1.0 - p_true


def joint(assignment):
    """Full joint probability of a COMPLETE assignment (chain rule over the network)."""
    p = 1.0
    for v in VARS:
        p *= prob(v, assignment[v], assignment)
    return p


def enumerate_all(variables, evidence):
    """Recursive ENUMERATE-ALL: sum out every variable that is not in `evidence`."""
    if not variables:
        return 1.0
    first, rest = variables[0], variables[1:]
    if first in evidence:
        return prob(first, evidence[first], evidence) * enumerate_all(rest, evidence)
    total = 0.0
    for val in (True, False):
        total += prob(first, val, {**evidence, first: val}) * enumerate_all(rest, {**evidence, first: val})
    return total


def enumeration_ask(query, evidence=None):
    """Return the normalised distribution {True: P(query=True|e), False: ...}."""
    evidence = dict(evidence or {})
    dist = {}
    for val in (True, False):
        dist[val] = enumerate_all(VARS, {**evidence, query: val})
    alpha = 1.0 / sum(dist.values())
    return {k: v * alpha for k, v in dist.items()}, dist     # normalised, unnormalised


def brute_force(query, evidence):
    """Independent check: sum the 2^8 joint entries directly."""
    num = den = 0.0
    for vals in product((True, False), repeat=len(VARS)):
        a = dict(zip(VARS, vals))
        if all(a[k] == v for k, v in evidence.items()):
            p = joint(a)
            den += p
            if a[query]:
                num += p
    return num / den


QUERIES = {
    "(a) P(Cr)":              ("Cr", {}),
    "(b) P(Al)":              ("Al", {}),
    "(c) P(Cr | Re)":         ("Cr", {"Re": True}),
    "(d) P(Es)":              ("Es", {}),
    "(e) P(At | Al, ~Co)":    ("At", {"Al": True, "Co": False}),
    "(f) P(Re | Cr, ~Al)":    ("Re", {"Cr": True, "Al": False}),
}


def answers():
    out = {}
    for name, (q, ev) in QUERIES.items():
        norm, unnorm = enumeration_ask(q, ev)
        out[name] = dict(answer=norm[True], unnorm_true=unnorm[True], unnorm_false=unnorm[False],
                         check=brute_force(q, ev))
        assert abs(out[name]["answer"] - out[name]["check"]) < 1e-12
    return out


def intermediate():
    """Intermediate tables used in the hand derivation (all by enumeration)."""
    T, F = True, False
    P = lambda v, val, a: prob(v, val, a)
    r = {}
    # marginals of the root-side variables
    r["P(Bu)"] = 0.2 * 0.3 + 0.8 * 0.05
    r["P(Ov)"] = 0.1 * 0.6 + 0.9 * 0.1
    # P(Cr | Ov) = sum_Bu P(Bu) P(Cr|Bu,Ov)
    r["P(Cr|Ov)"] = {ov: sum((r["P(Bu)"] if bu else 1 - r["P(Bu)"]) * CPT["Cr"][(bu, ov)] for bu in (T, F))
                     for ov in (T, F)}
    # joint P(Cr=T, At)
    r["P(Cr,At)"] = {}
    for at in (T, F):
        pat = 0.1 if at else 0.9
        r["P(Cr,At)"][at] = sum(pat * prob("Ov", ov, {"At": at}) * r["P(Cr|Ov)"][ov] for ov in (T, F))
    r["P(Cr,~At)"] = None
    # P(Cr=T) and P(Cr=F, At)
    r["P(Cr)"] = sum(r["P(Cr,At)"].values())
    r["P(~Cr,At)"] = {at: (0.1 if at else 0.9) - r["P(Cr,At)"][at] for at in (T, F)}
    # P(Al, Co) table
    palco = {}
    for al, co in product((T, F), repeat=2):
        tot = 0.0
        for cr in (T, F):
            for at in (T, F):
                pcrat = r["P(Cr,At)"][at] if cr else r["P(~Cr,At)"][at]
                tot += pcrat * prob("Al", al, {"Cr": cr, "At": at}) * prob("Co", co, {"Cr": cr})
        palco[(al, co)] = tot
    r["P(Al,Co)"] = palco
    r["P(Al)"] = palco[(T, T)] + palco[(T, F)]
    r["P(Es)"] = sum(prob("Es", T, {"Al": al, "Co": co}) * palco[(al, co)] for al, co in palco)
    return r


if __name__ == "__main__":
    print("PART C - inference by enumeration")
    for name, d in answers().items():
        print(f"{name:22s} = {d['answer']:.4f}   (unnormalised T={d['unnorm_true']:.6f}, "
              f"F={d['unnorm_false']:.6f}; brute-force check {d['check']:.4f})")
    print("\nIntermediate quantities:")
    for k, v in intermediate().items():
        print(" ", k, "=", v)
