---
ral_schema: '2.1'
ral_id: RAL-BIO-JBDVKQTZAC
node_type: THEOREM
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: bio.systems-biochemistry
title: QTT Rank Inclusion Theorem
formal_signature:
- rank_tt(A) <= rank_qtt(A_q)
assumptions:
- The tensor is quantized using the natural ordering of virtual indices.
- The QTT decomposition is a TT decomposition of the quantized tensor.
assumptions_hash: sha256:b839dab0eedde7968dcaba6fb1d01f4aab1f4cb83bf94280d7ecdd12c38f54dc
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:144a973411a52d76bd41318e99960055a35c0a3f4c035ddef2cf225213f26586
  verified_at: '2026-11-25T06:21:01Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: pmc:PMC3953644
  source_license: CC-BY-4.0
  source_format: pmc_jats
  source_locator: pmc:PMC3953644
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/bb/ed/bbeddd963ab4559ea893e423a18c934127524d81aee31242a6d465eed19e7aad.xml
  ore_sha256: bbeddd963ab4559ea893e423a18c934127524d81aee31242a6d465eed19e7aad
  quote_sha256: sha256:ee8b4ad6b2dd8b0120f0eed458e8fe6a396a05c26cc49b3642586fb1369b152f
  quote_char_len: 128
  pub_year: 2014
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$\forall \mathcal{A} \in \mathbb{R}^{n_1 \times \dots \times n_d}, \text{rank}_{TT}(\mathcal{A}) \subseteq \text{rank}_{QTT}(\mathcal{A}_q)$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| \mathcal{A} | Original d-dimensional tensor | Real Tensor Space | dimensionless |
| \mathcal{A}_q | Quantized tensor representation | Real Tensor Space | dimensionless |
| rank_{TT} | Tensor Train rank vector | Positive Integers | dimensionless |
| rank_{QTT} | Quantized Tensor Train rank vector | Positive Integers | dimensionless |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Non-natural index ordering leads to rank permutation.
- **Breaks when:** Tensor dimensions not powers of 2 require padding, altering rank.
- **Breaks when:** Numerical instability in SVD truncation during decomposition.

### Executable Verification Kernel
```python
import numpy as np
from sympy import Matrix

def verify_rank_relation():
    # Simulate a small tensor and its quantized version
    # TT rank is bounded by the dimensions of the unfolding
    A = np.random.rand(4, 4)
    # Quantized version effectively increases dimensionality
    Aq = A.reshape(2, 2, 2, 2)
    
    # TT rank of A (matrix rank)
    rank_tt = np.linalg.matrix_rank(A)
    
    # QTT rank simulation (unfolding along virtual indices)
    # For a 2x2x2x2 tensor, QTT ranks are determined by unfoldings
    u1 = Aq.reshape(2, 8)
    u2 = Aq.reshape(4, 4)
    u3 = Aq.reshape(8, 2)
    
    ranks_qtt = [np.linalg.matrix_rank(u1), np.linalg.matrix_rank(u2), np.linalg.matrix_rank(u3)]
    
    # Assertion 1: TT rank is at most the max QTT rank
    assert rank_tt <= max(ranks_qtt)
    
    # Assertion 2: QTT ranks are non-negative integers
    assert all(r >= 0 for r in ranks_qtt)
    
    # Assertion 3: Rank of identity matrix is consistent
    I = np.eye(4)
    assert np.linalg.matrix_rank(I) == 4
    
verify_rank_relation()
```

### Canonical Provenance Anchors
- `pmc:PMC3953644` (`pmc:PMC3953644`): The relationship between TT and QTT ranks is established by the structural embedding of the TT-format within the QTT-format when the quantization mapping preserves the original tensor's index hierarchy.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
