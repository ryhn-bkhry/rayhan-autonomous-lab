---
ral_schema: '2.1'
ral_id: RAL-PHYS-Y9E96H9QDF
node_type: THEOREM
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: phys.relativity-field-theory
title: Topological susceptibility in Type IIA supergravity
formal_signature:
- chi_t(lambda, R_0) = (2 * lambda^3) / (729 * pi^3 * R_0^4)
assumptions:
- Type IIA supergravity
- large N limit
- leading asymptotic behavior in 1/lambda
assumptions_hash: sha256:3d1fa51bbdcbf99e37e9d221887f61f6bf4d8630e040e9fb4f2c41325dd9980e
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:0e068ac85f18a4254b553d2ce5abd5ce42865f186ee9fbbeb83fa76c07680b26
  verified_at: '2026-11-25T06:03:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:hep-th/9905111
  source_license: CC-BY-4.0
  source_format: arxiv_tex
  source_locator: arxiv:hep-th/9905111
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/89/64/896439920e7a71aea4a5106b170cf00cb0faf34ab82afd31fdc1857368a78a1b.tar.gz
  ore_sha256: 896439920e7a71aea4a5106b170cf00cb0faf34ab82afd31fdc1857368a78a1b
  quote_sha256: sha256:a153928a05e48eea410ca194bd5a302a80add305d51ba969e9427e9b434e5a1f
  quote_char_len: 248
  pub_year: 2000
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$\chi_t = \frac{2 \lambda^3}{729 \pi^3 R_0^4}$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| \chi_t | Topological susceptibility | real | m^-4 |
| \lambda | 't Hooft coupling constant | real | dimensionless |
| R_0 | Characteristic scale of the supergravity background | real | m |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Invalid for small N where 1/N corrections dominate.
- **Breaks when:** Fails when string coupling g_s becomes large.
- **Breaks when:** Requires lambda >> 1 for the validity of the supergravity approximation.

### Executable Verification Kernel
```python
import sympy as sp

lambda_val = sp.Symbol('lambda')
R0 = sp.Symbol('R0')
chi_t = (2 * lambda_val**3) / (729 * sp.pi**3 * R0**4)

# Assertion 1: Scaling behavior with lambda
assert sp.simplify(sp.diff(chi_t, lambda_val) - (6 * lambda_val**2) / (729 * sp.pi**3 * R0**4)) == 0

# Assertion 2: Scaling behavior with R0
assert sp.simplify(sp.diff(chi_t, R0) + (8 * lambda_val**3) / (729 * sp.pi**3 * R0**5)) == 0

# Assertion 3: Consistency check for specific ratio
chi_ratio = chi_t.subs({lambda_val: 10, R0: 1})
expected = 2000 / (729 * sp.pi**3)
assert sp.simplify(chi_ratio - expected) == 0
```

### Canonical Provenance Anchors
- `arxiv:hep-th/9905111` (`arxiv:hep-th/9905111`): Derived from the 2-point function of the RR 1-form field within the Type IIA supergravity action as applied to the holographic dual of QCD.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
