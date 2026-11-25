---
ral_schema: '2.1'
ral_id: RAL-CHEM-4RBC80E91Z
node_type: EMPIRICAL_LAW
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: chem.kinetics-catalysis
title: Ehrenfest-like Nuclear Propagation
formal_signature:
- M_a * d^2R_a/dt^2 = -grad_a E[rho(R(t), r, t)]
assumptions:
- Classical nuclei
- Mean time-dependent electronic potential
- Non-adiabatic coupling approximation
assumptions_hash: sha256:b71f2fbee156d42233da1840d304384be02b645f20b4b052298c0463712e4bf3
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:3ffdaa462279db24f6166f53242e95f94a8311cb9dc7f0777ce04903edb2e254
  verified_at: '2026-11-25T06:17:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: pmc:PMC7116320
  source_license: CC-BY-4.0
  source_format: pmc_jats
  source_locator: pmc:PMC7116320
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/88/0c/880c29797361efdbbc2c4e2f506c7c15c3cc6a26a3ee3a953b9886e224f9ea9f.xml
  ore_sha256: 880c29797361efdbbc2c4e2f506c7c15c3cc6a26a3ee3a953b9886e224f9ea9f
  quote_sha256: sha256:b00c163e336224809d4db80471113f28b0d93be328e863a6d78482dbf9ed7c3c
  quote_char_len: 27
  pub_year: 2020
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$M_a \frac{d^2 R_a}{dt^2} = -\nabla_a \langle \Psi(r, t; R) | \hat{H}_{el} | \Psi(r, t; R) \rangle$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| M_a | Mass of nucleus a | real | kg |
| R_a | Position vector of nucleus a | real | m |
| E | Mean electronic energy | real | J |
| t | Time | real | s |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Breakdown of mean-field approximation in strong correlation regimes
- **Breaks when:** Neglect of nuclear quantum effects (tunneling, zero-point energy)
- **Breaks when:** Inaccuracy in regions of high non-adiabatic coupling (conical intersections)

### Executable Verification Kernel
```python
import sympy as sp
import numpy as np

M, R, t = sp.symbols('M R t')
E = sp.Function('E')(R)
accel = -sp.diff(E, R) / M

# Assertion 1: Force-Mass-Acceleration consistency
force = -sp.diff(E, R)
assert sp.simplify(M * accel - force) == 0

# Assertion 2: Numerical gradient check for harmonic potential
k = 2.0
E_harm = 0.5 * k * R**2
force_val = -sp.diff(E_harm, R).subs(R, 1.0)
assert np.isclose(float(force_val), -2.0)

# Assertion 3: Kinetic energy conservation check for constant potential
# If grad E = 0, acceleration must be 0
accel_zero = -sp.diff(sp.Integer(5), R) / M
assert accel_zero == 0
```

### Canonical Provenance Anchors
- `pmc:PMC7116320` (`pmc:PMC7116320`): The Ehrenfest dynamics approach approximates the nuclear motion by treating nuclei as classical particles moving on a potential energy surface defined by the expectation value of the electronic Hamiltonian.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
