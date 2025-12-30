# QAP Solver (EA vs CCEA) — chr12a

Python implementation of a **Single-population Evolutionary Algorithm (EA)** and a **Cooperative Coevolutionary Algorithm (CCEA)** to solve the **Quadratic Assignment Problem (QAP)** on the **QAPLIB chr12a** instance (optimal cost = **9552**).
## What’s inside
- **EA**: evolves a full permutation (single population).
- **CCEA**: splits the permutation into multiple subpopulations that co-evolve and are evaluated cooperatively.

## Results (chr12a)
Best observed: **EA = 9562**, **CCEA = 10688**.
## Requirements
Python 3 + `numpy` (and `matplotlib` if you plot).

## Run
```bash
python single_implementation.py
python co_implementation.py
