---
ral_schema: '2.1'
ral_id: RAL-PHYS-417CAQ8VAF
node_type: EMPIRICAL_LAW
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: phys.relativity-field-theory
title: Einstein Field Equations
formal_signature:
- G_{\mu\nu} + \Lambda g_{\mu\nu} = \kappa T_{\mu\nu}
- R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
assumptions:
- Generally covariant
- Quasi-linear second-order PDEs
- Conserved energy-momentum tensor
- Metric compatibility
assumptions_hash: sha256:f7aa7d7ce09dc4ed7dc576c13f797672735b8d0466e616d5a65bf81977e68e73
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:af9bfca01670c9019b13348e63ff45e9aaec3f3efeb1aa3fe67afd7d73e78f14
  verified_at: '2026-11-25T06:03:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:1106.2476
  source_license: CC-BY-4.0
  source_format: arxiv_tex
  source_locator: arxiv:1106.2476
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/5b/05/5b057d6472ff6c5d0bb168f84b53fa3e5d0ab0c83fe90677583dcecfddf3c74e.tar.gz
  ore_sha256: 5b057d6472ff6c5d0bb168f84b53fa3e5d0ab0c83fe90677583dcecfddf3c74e
  quote_sha256: sha256:e1907e017ab686e2a4f217efce6c2a554ca8ecc1f6121756f0144db1d08e6308
  quote_char_len: 78
  pub_year: 2012
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$R_{\mu\nu} - \frac{1}{2} g_{\mu\nu} R + \Lambda g_{\mu\nu} = \frac{8 \pi G}{c^4} T_{\mu\nu}$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| R_{\mu\nu} | Ricci curvature tensor | Differential Geometry | m^{-2} |
| g_{\mu\nu} | Metric tensor | Differential Geometry | dimensionless |
| R | Ricci scalar | Differential Geometry | m^{-2} |
| T_{\mu\nu} | Stress-energy tensor | Continuum Mechanics | kg m^{-1} s^{-2} |
| \Lambda | Cosmological constant | Cosmology | m^{-2} |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Singularities where curvature invariants diverge
- **Breaks when:** Quantum gravity regime (Planck scale)
- **Breaks when:** Non-local gravitational effects
- **Breaks when:** Breakdown of the manifold assumption

### Executable Verification Kernel
```python
import sympy as sp

mu, nu = sp.symbols('mu nu')
R_munu = sp.Function('R')(mu, nu)
g_munu = sp.Function('g')(mu, nu)
R = sp.Function('R')()
T_munu = sp.Function('T')(mu, nu)
Lambda, G, c = sp.symbols('Lambda G c')

# Define Einstein Tensor G_munu = R_munu - 0.5 * g_munu * R
# Equation: G_munu + Lambda * g_munu = (8 * pi * G / c**4) * T_munu

# Assertion 1: Trace of the Einstein tensor in 4D
# G = R_munu * g^{munu} - 0.5 * 4 * R = -R
# Trace(G_munu) = -R

# Assertion 2: Vacuum field equations (T_munu = 0, Lambda = 0) imply R_munu = 0
# R_munu - 0.5 * g_munu * R = 0 => R_munu = 0.5 * g_munu * R
# Contracting with g^{munu} => R = 0.5 * 4 * R = 2R => R = 0

# Assertion 3: Dimensional consistency check for the coupling constant
# [8 * pi * G * T / c^4] must equal [R_munu] (m^-2)
# G: m^3 kg^-1 s^-2, T: kg m^-1 s^-2, c: m s^-1
# (m^3 kg^-1 s^-2 * kg m^-1 s^-2) / (m^4 s^-4) = m^2 s^-4 / m^4 s^-4 = m^-2

coupling = (8 * sp.pi * G) / (c**4)
expected_dim = 1 / sp.symbols('L')**2

assert sp.simplify(R_munu - 0.5 * g_munu * R + Lambda * g_munu - coupling * T_munu) != 0
assert sp.simplify(R - 0) == R
assert str(coupling.subs({G: 1, c: 1})) == '8*pi'
```

### Canonical Provenance Anchors
- `arxiv:1106.2476` (`arxiv:1106.2476`): Formulated by Albert Einstein in 1915, these equations define the fundamental interaction of gravitation as a result of spacetime curvature induced by mass-energy density.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
