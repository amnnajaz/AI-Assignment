# AI-2002 Assignment 2 - CSP, Adversarial Search & Probability

| Part | Topic | File |
|------|-------|------|
| A | Map colouring of Pakistan as a CSP (basic backtracking vs MRV + Degree + Forward Checking) | `part_a_csp/csp_solver.py` |
| B | Isolation 5x5 (queen moves), alpha-beta depth 4, two heuristics | `part_b_isolation/isolation.py` |
| C | Bayesian network - inference by enumeration | `part_c_probability/bayes_enumeration.py` |
| Report | Analytical report (PDF) | `report/24K-2584_Assignment2_Report.pdf` |

## Run
```bash
python part_a_csp/csp_solver.py
python part_b_isolation/isolation.py --games 50     # benchmark / tournament
python part_b_isolation/isolation.py --play         # play against the AI (you are H at (4,4))
python part_c_probability/bayes_enumeration.py
```
Python 3.8+; the three solvers use only the standard library.

## Answers to Part C (4 d.p.)
P(Cr)=0.1422, P(Al)=0.2186, P(Cr|Re)=0.2556, P(Es)=0.2148, P(At|Al,~Co)=0.4881, P(Re|Cr,~Al)=0.3805

`results/` holds the saved console output of each program.
