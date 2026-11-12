"""
AI-2002 Assignment 2 - Part A: Map colouring of Pakistan as a CSP.

Contains
    * the CSP formulation (variables, domains, binary "not-equal" constraints)
    * Task 1: plain depth-first backtracking search (alphabetical variable order)
    * Task 2: enhanced backtracking with MRV + Degree heuristic + Forward Checking
    * Task 3: counters (assignments, backtracks, recursive calls, constraint checks)
      and a benchmark that compares the two solvers..

Run:  python csp_solver.py
"""
import itertools
import random
import time
from copy import deepcopy

VARIABLES = ["AK", "BA", "GB", "ICT", "KP", "PB", "SD"]   
FULL_NAMES = {
    "BA": "Balochistan", "SD": "Sindh", "PB": "Punjab",
    "KP": "Khyber Pakhtunkhwa", "GB": "Gilgit-Baltistan",
    "AK": "Azad Kashmir", "ICT": "Islamabad Capital Territory",
}
COLORS = ["Red", "Green", "Blue"]          

NEIGHBORS = {                              
    "BA":  ["SD", "PB", "KP"],
    "SD":  ["BA", "PB"],
    "PB":  ["BA", "SD", "KP", "ICT", "AK"],
    "KP":  ["BA", "PB", "ICT", "GB", "AK"],
    "GB":  ["KP", "AK"],
    "AK":  ["GB", "KP", "PB"],
    "ICT": ["KP", "PB"],
}


def check_symmetry(neighbors=NEIGHBORS):
    """Sanity check: adjacency must be symmetric (A borders B <=> B borders A)."""
    for a, nbrs in neighbors.items():
        for b in nbrs:
            assert a in neighbors[b], f"Adjacency not symmetric: {a}-{b}"
    return True


class Stats:
    """Counters used for the performance analysis (Task 3)."""

    def __init__(self):
        self.assignments = 0        # every time a value is assigned to a variable
        self.backtracks = 0         # every time an assignment is undone after failure
        self.recursive_calls = 0    # calls to the recursive search function
        self.constraint_checks = 0  # individual "is this value legal?" tests
        self.pruned_values = 0      # (forward checking only) values removed from domains
        self.time_s = 0.0

    def as_dict(self):
        return dict(assignments=self.assignments, backtracks=self.backtracks,
                    recursive_calls=self.recursive_calls,
                    constraint_checks=self.constraint_checks,
                    pruned_values=self.pruned_values, time_ms=self.time_s * 1000)


def is_consistent(var, value, assignment, neighbors, stats):
    """True if giving `var` the colour `value` violates no constraint with
    the already-assigned neighbours."""
    for nb in neighbors[var]:
        stats.constraint_checks += 1
        if assignment.get(nb) == value:
            return False
    return True


#Task 1 - Basic backtracking search (no heuristics)

def basic_backtracking(variables=VARIABLES, colors=COLORS, neighbors=NEIGHBORS,
                       find_all=False, stats=None):
    """Plain DFS backtracking.
    * Variable choice : first unassigned variable in the given (alphabetical) order.
    * Value choice    : colours in the fixed order of `colors`.
    * No look-ahead   : conflicts are only discovered when we try to assign.

    Returns (first_solution_or_None, list_of_all_solutions, stats).
    If find_all is False the search stops at the first solution.
    """
    stats = stats or Stats()
    solutions = []

    def recurse(assignment):
        stats.recursive_calls += 1
        if len(assignment) == len(variables):
            solutions.append(dict(assignment))
            return not find_all                      # True => stop searching
        var = next(v for v in variables if v not in assignment)   # chronological order
        for value in colors:
            if is_consistent(var, value, assignment, neighbors, stats):
                assignment[var] = value
                stats.assignments += 1
                if recurse(assignment):
                    return True
                del assignment[var]                  
                stats.backtracks += 1
        return False

    t0 = time.perf_counter()
    recurse({})
    stats.time_s = time.perf_counter() - t0
    return (solutions[0] if solutions else None), solutions, stats


#Task 2 - Enhanced backtracking: MRV + Degree + Forward Checking

def select_unassigned_variable(assignment, domains, variables, neighbors):
    """MRV: choose the variable with the fewest legal values left.
    Ties are broken by the Degree heuristic: the variable involved in the most
    constraints with *other unassigned* variables. Remaining ties -> list order."""
    unassigned = [v for v in variables if v not in assignment]

    def key(v):
        degree = sum(1 for nb in neighbors[v] if nb not in assignment)
        return (len(domains[v]), -degree)           # min MRV, then max degree
    return min(unassigned, key=key)


def forward_check(var, value, domains, assignment, neighbors, stats):
    """After assigning var=value remove `value` from the domains of the
    unassigned neighbours. Returns (ok, removed) where `removed` is the list
    of (neighbour, value) pairs deleted (so they can be restored on backtrack),
    and ok is False if some neighbour's domain became empty (dead end)."""
    removed = []
    for nb in neighbors[var]:
        if nb in assignment:
            continue
        stats.constraint_checks += 1
        if value in domains[nb]:
            domains[nb].remove(value)
            removed.append((nb, value))
            stats.pruned_values += 1
            if not domains[nb]:
                return False, removed
    return True, removed


def heuristic_backtracking(variables=VARIABLES, colors=COLORS, neighbors=NEIGHBORS,
                           find_all=False, stats=None, use_mrv=True,
                           use_degree=True, use_fc=True):
    """Backtracking with MRV + Degree tie-break + Forward Checking.
    The use_* flags allow ablation studies (turn each heuristic on/off)."""
    stats = stats or Stats()
    solutions = []
    domains = {v: list(colors) for v in variables}

    def pick(assignment):
        unassigned = [v for v in variables if v not in assignment]
        if use_mrv and use_degree:
            return select_unassigned_variable(assignment, domains, variables, neighbors)
        if use_mrv:
            return min(unassigned, key=lambda v: len(domains[v]))
        if use_degree:
            return max(unassigned, key=lambda v: sum(1 for n in neighbors[v] if n not in assignment))
        return unassigned[0]

    def recurse(assignment):
        stats.recursive_calls += 1
        if len(assignment) == len(variables):
            solutions.append(dict(assignment))
            return not find_all
        var = pick(assignment)
        for value in list(domains[var]):
            if not use_fc and not is_consistent(var, value, assignment, neighbors, stats):
                continue
            assignment[var] = value
            stats.assignments += 1
            removed = []
            ok = True
            if use_fc:
                ok, removed = forward_check(var, value, domains, assignment, neighbors, stats)
            if ok and recurse(assignment):
                return True
            # undo
            for nb, val in removed:
                domains[nb].append(val)
            del assignment[var]
            stats.backtracks += 1
        return False

    t0 = time.perf_counter()
    recurse({})
    stats.time_s = time.perf_counter() - t0
    return (solutions[0] if solutions else None), solutions, stats


#Task 3 - Benchmarking

def verify(solution, neighbors=NEIGHBORS):
    """Independent checker: every region coloured, no adjacent pair equal."""
    if solution is None or set(solution) != set(neighbors):
        return False
    return all(solution[a] != solution[b] for a in neighbors for b in neighbors[a])


def run_benchmarks(verbose=True):
    """Run every experiment used in the report and return the results dict."""
    check_symmetry()
    res = {}

    # (a) first solution, alphabetical order, colours R,G,B
    sol_b, _, st_b = basic_backtracking()
    sol_h, _, st_h = heuristic_backtracking()
    res["first_basic"] = (sol_b, st_b.as_dict())
    res["first_heur"] = (sol_h, st_h.as_dict())
    assert verify(sol_b) and verify(sol_h)

    # (b) exhaustive search (all solutions) - reveals the full size of the search tree
    _, all_b, sb = basic_backtracking(find_all=True)
    _, all_h, sh = heuristic_backtracking(find_all=True)
    assert len(all_b) == len(all_h)
    res["all_basic"] = (len(all_b), sb.as_dict())
    res["all_heur"] = (len(all_h), sh.as_dict())

    # (c) robustness: all 6 colour orderings x all 7! = 5040 variable orderings (first solution)
    #     basic search is order-dependent, so this shows the average-case behaviour.
    totals = {"basic": [0, 0, 0], "heur": [0, 0, 0]}
    worst_basic = 0
    n = 0
    for cols in itertools.permutations(COLORS):
        for order in itertools.permutations(VARIABLES):
            _, _, s1 = basic_backtracking(variables=list(order), colors=list(cols))
            _, _, s2 = heuristic_backtracking(variables=list(order), colors=list(cols))
            for key, s in (("basic", s1), ("heur", s2)):
                totals[key][0] += s.assignments
                totals[key][1] += s.backtracks
                totals[key][2] += s.recursive_calls
            worst_basic = max(worst_basic, s1.backtracks)
            n += 1
    res["avg"] = {k: [x / n for x in v] for k, v in totals.items()}
    res["avg_n"] = n
    res["worst_basic_backtracks"] = worst_basic

    # (d) ablation (all solutions): which heuristic contributes what?
    abl = {}
    for name, flags in {
        "none (basic order, no FC)": dict(use_mrv=False, use_degree=False, use_fc=False),
        "FC only": dict(use_mrv=False, use_degree=False, use_fc=True),
        "MRV only": dict(use_mrv=True, use_degree=False, use_fc=False),
        "MRV + Degree": dict(use_mrv=True, use_degree=True, use_fc=False),
        "MRV + Degree + FC (full)": dict(use_mrv=True, use_degree=True, use_fc=True),
    }.items():
        _, sols, s = heuristic_backtracking(find_all=True, **flags)
        abl[name] = s.as_dict()
    res["ablation"] = abl

    if verbose:
        print_report(res)
    return res


def print_report(res):
    print("=" * 72)
    print("PART A - CSP map colouring of Pakistan")
    print("=" * 72)
    for label, key in (("Basic backtracking", "first_basic"), ("MRV+Degree+FC", "first_heur")):
        sol, st = res[key]
        print(f"\n{label} (first solution)")
        print("  solution :", sol)
        print("  stats    :", {k: round(v, 3) for k, v in st.items()})
    print("\nExhaustive search (all solutions)")
    for label, key in (("Basic", "all_basic"), ("Heuristic", "all_heur")):
        cnt, st = res[key]
        print(f"  {label:10s} solutions={cnt}  ", {k: round(v, 3) for k, v in st.items()})
    print(f"\nAverage over {res['avg_n']} (variable order x colour order) combos "
          "[assignments, backtracks, calls]")
    print("  basic     :", [round(x, 2) for x in res['avg']['basic']])
    print("  heuristic :", [round(x, 2) for x in res['avg']['heur']])
    print("  worst-case basic backtracks:", res["worst_basic_backtracks"])
    print("\nAblation (all solutions)")
    for k, v in res["ablation"].items():
        print(f"  {k:28s} assignments={v['assignments']:5d} backtracks={v['backtracks']:5d} "
              f"calls={v['recursive_calls']:5d}")


if __name__ == "__main__":
    run_benchmarks()
