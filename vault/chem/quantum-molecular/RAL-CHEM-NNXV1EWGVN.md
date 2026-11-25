---
ral_schema: '2.1'
ral_id: RAL-CHEM-NNXV1EWGVN
node_type: THEOREM
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: chem.quantum-molecular
title: Jacobi's Formula for Matrix Determinant Derivative
formal_signature:
- d/dk ln(det(H(k))) = Tr(H(k)^-1 * dH(k)/dk)
assumptions:
- H(k) is an invertible matrix
- H(k) is differentiable with respect to k
assumptions_hash: sha256:e620c1c1913d6b8630db2d0dfadeb693fea857dfa7c61ebcb245a02af65c29b4
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:f86f8b9034fb686d90d716b000baac86b167a8cc123681aef2d15d64ac530932
  verified_at: '2026-11-25T06:10:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:1802.07964
  source_license: CC-BY-4.0
  source_format: arxiv_tex
  source_locator: arxiv:1802.07964, Eq. (23)
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/a4/86/a4861294da539f778f32ce5a040d14e363d95439ec7ec731987d2356765e38d0.tar.gz
  ore_sha256: a4861294da539f778f32ce5a040d14e363d95439ec7ec731987d2356765e38d0
  quote_sha256: sha256:e47efc8c4aad2ad431ece11b2689fc01297ae1ed2a739daf9cb4dead8b4b3e70
  quote_char_len: 62
  pub_year: 2018
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$\frac{\partial}{\partial k} \ln(\det H(k)) = \text{Tr}\left( H(k)^{-1} \frac{\partial H(k)}{\partial k} \right)$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| H(k) | Hamiltonian matrix parameterized by k | complex-matrices | joule |
| k | Scalar parameter | real-scalars | dimensionless |
| Tr | Matrix trace operator | linear-algebra | dimensionless |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Singular matrix where det(H(k)) = 0
- **Breaks when:** Non-differentiable H(k) with respect to k
- **Breaks when:** Non-square Hamiltonian matrices

### Executable Verification Kernel
```python
import sympy as sp
k = sp.symbols('k')
a, b, c, d = sp.symbols('a b c d')
H = sp.Matrix([[a*k, b], [c, d*k]])
det_H = H.det()
log_det_H = sp.log(det_H)
lhs = sp.diff(log_det_H, k)
H_inv = H.inv()
dH_dk = sp.diff(H, k)
rhs = sp.trace(H_inv * dH_dk)
assert sp.simplify(lhs - rhs) == 0
assert H.det() != 0
assert sp.simplify(lhs) == (a*d*k + a*d*k) / (a*d*k**2 - b*c)
```

### Canonical Provenance Anchors
- `arxiv:1802.07964` (`arxiv:1802.07964, Eq. (23)`): This identity is a specific application of Jacobi's formula for the derivative of the determinant, widely utilized in quantum chemistry for calculating gradients of electronic energy surfaces.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
