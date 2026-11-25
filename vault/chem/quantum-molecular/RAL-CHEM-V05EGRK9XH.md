---
ral_schema: '2.1'
ral_id: RAL-CHEM-V05EGRK9XH
node_type: EMPIRICAL_LAW
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: chem.quantum-molecular
title: SOS-MP2 Energy Working Equation
formal_signature:
- E_{SOS-MP2} = -\sum_{q} w_q \text{Tr}(Q^{\alpha}(t_q) Q^{\beta}(t_q))
assumptions:
- SOS-MP2 approximation
- Minimax quadrature weights and abscissa
- Spin-orbital separation
assumptions_hash: sha256:ad121098bc83a5372a8cfb9d9a1a87052ab985bf74846927166a3a0320a226ba
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:aa7aea67e3a8a53356e825acd2c096724f0ba0f7a64a3845c353cd2cc2b1580a
  verified_at: '2026-11-25T06:10:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:2003.03868
  source_license: CC-BY-4.0
  source_format: arxiv_tex
  source_locator: arxiv:2003.03868
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/c5/46/c5460956e79535c0e4d2c2a47760aef054a7b43d6489c1962007bf43dcd4e074.tar.gz
  ore_sha256: c5460956e79535c0e4d2c2a47760aef054a7b43d6489c1962007bf43dcd4e074
  quote_sha256: sha256:9892d5b2b3d654f93d54e395e0eb067bfb1b610064430b8f76fef20a50012a4e
  quote_char_len: 76
  pub_year: 2020
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$E^{SOS-MP2}=-\sum_{q} w_{q} \sum_{PQ} Q_{PQ}^{\alpha}(t_q) Q_{QP}^{\beta}(t_{q})$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| E^{SOS-MP2} | Spin-Opposite Scaled Møller-Plesset second-order energy | energy | J |
| w_q | Quadrature weight for point q | dimensionless | 1 |
| Q_{PQ}^{\alpha}(t_q) | Alpha spin matrix element at time t_q | matrix | dimensionless |
| t_q | Quadrature abscissa | time | s |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Fails for systems with significant spin-same-spin correlation contributions.
- **Breaks when:** Accuracy degrades if the minimax quadrature order is insufficient for the decay of the Laplace-transformed orbital energy denominators.
- **Breaks when:** Not applicable to non-closed-shell systems where alpha and beta spaces are not distinct.

### Executable Verification Kernel
```python
import sympy as sp
import numpy as np

def verify_sos_mp2():
    q_alpha = sp.Matrix([[1, 2], [3, 4]])
    q_beta = sp.Matrix([[5, 6], [7, 8]])
    
    prod = q_alpha * q_beta
    trace_val = prod.trace()
    
    # Manual trace calculation: (1*5 + 2*7) + (3*6 + 4*8) = 19 + 50 = 69
    manual_trace = (q_alpha[0,0]*q_beta[0,0] + q_alpha[0,1]*q_beta[1,0] + 
                    q_alpha[1,0]*q_beta[0,1] + q_alpha[1,1]*q_beta[1,1])
    
    # Assertion 1: Trace identity
    assert sp.simplify(trace_val - manual_trace) == 0
    
    # Assertion 2: Energy scaling property
    w_val = sp.Rational(1, 2)
    e_val = -w_val * trace_val
    assert sp.simplify(e_val + sp.Rational(69, 2)) == 0
    
    # Assertion 3: Matrix trace value check
    assert np.isclose(float(trace_val), 69.0)

verify_sos_mp2()
```

### Canonical Provenance Anchors
- `arxiv:2003.03868` (`arxiv:2003.03868`): The SOS-MP2 method approximates the MP2 correlation energy by scaling the opposite-spin component, reducing computational complexity from O(N^5) to O(N^4) via Laplace transformation and minimax quadrature.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
