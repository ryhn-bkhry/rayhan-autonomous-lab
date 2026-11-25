---
ral_schema: '2.1'
ral_id: RAL-CHEM-8NFEJBN44J
node_type: THEOREM
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: chem.quantum-molecular
title: Winding number of complex eigenenergies
formal_signature:
- w \in \mathbb{Z}
- E_n(k) \in \mathbb{C} \setminus \{0\}
- k \in [-\pi, \pi]
assumptions:
- E_n(k) are eigenenergies of H(k)
- E_n(k) are in C/{0}
- N is the total number of bands
assumptions_hash: sha256:193bf452987c4fca5077ccb7d5713504b8e4989765323eec240ea9d76ea45a17
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:b7771146ed9a1ced558c638545b7f7adcc2c4d2370a0efed1c13623a2baf2efc
  verified_at: '2026-11-25T06:10:16Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:1802.07964
  source_license: CC-BY-4.0
  source_format: arxiv_tex
  source_locator: arxiv:1802.07964, Eq. (6)
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/a4/86/a4861294da539f778f32ce5a040d14e363d95439ec7ec731987d2356765e38d0.tar.gz
  ore_sha256: a4861294da539f778f32ce5a040d14e363d95439ec7ec731987d2356765e38d0
  quote_sha256: sha256:d6a14edb123c5f54794ca6a0e32c7f6886fb2884729774045d72f2ec681bc526
  quote_char_len: 65
  pub_year: 2018
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$w = \sum_{n=1}^{N} \int_{-\pi}^{\pi} \frac{dk}{2\pi} \partial_k \arg E_n(k)$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| w | Topological winding number | integer | dimensionless |
| E_n(k) | Complex eigenenergy of the n-th band at momentum k | complex | joule |
| k | Crystal momentum in the Brillouin zone | real | rad/m |

### Failure Boundaries & Known Pathologies
- **Breaks when:** E_n(k) = 0 for any k (gap closing)
- **Breaks when:** Non-periodic boundary conditions in k-space
- **Breaks when:** Discontinuous eigenenergy branches

### Executable Verification Kernel
```python
import sympy as sp
import numpy as np

k = sp.Symbol('k', real=True)
a = sp.Symbol('a', real=True)
E = sp.exp(sp.I * k) + a

# The winding number is defined as the integral of the derivative of the argument.
# For E(k) = exp(ik) + a, the winding number is 1 if |a| < 1 and 0 if |a| > 1.
# We use the property that the winding number is the change in argument divided by 2*pi.

def get_winding(val):
    # Using numerical integration for the argument change
    f = lambda x: np.angle(np.exp(1j * x) + val)
    pts = np.linspace(-np.pi, np.pi, 1000)
    vals = f(pts)
    diffs = np.diff(vals)
    # Handle branch cuts by wrapping differences
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    return np.sum(diffs) / (2 * np.pi)

# Assertion 1: Winding number is 1 when origin is enclosed (|a| < 1)
w1 = get_winding(0.5)
assert np.isclose(w1, 1.0, atol=1e-2)

# Assertion 2: Winding number is 0 when origin is not enclosed (|a| > 1)
w2 = get_winding(2.0)
assert np.isclose(w2, 0.0, atol=1e-2)

# Assertion 3: Verify consistency of the winding number for a = 0
w3 = get_winding(0.0)
assert np.isclose(w3, 1.0, atol=1e-2)
```

### Canonical Provenance Anchors
- `arxiv:1802.07964` (`arxiv:1802.07964, Eq. (6)`): The winding number characterizes the topological phase of non-Hermitian systems by counting the number of times the complex eigenenergy spectrum encircles the origin in the complex plane as the momentum traverses the Brillouin zone.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
