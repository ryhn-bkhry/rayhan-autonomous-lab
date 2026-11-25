---
ral_schema: '2.1'
ral_id: RAL-CHEM-P1735BKXKS
node_type: THEOREM
epistemic_tier: T1_CANONICAL
consensus_version: 1
taxonomy_code: chem.kinetics-catalysis
title: Linear Response TD-DFTB Excitation Energies
formal_signature:
- Omega_J
- A
- B
- X
- Y
assumptions:
- Linear response approximation
- DFTB framework
- Hermitian coupling matrices A and B
assumptions_hash: sha256:b9107796fe064276c80e3a4bdb0d24925a57ecaa9f9d3e4971749c28ece72d42
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:181981781d13d0a58091ff98f81c46089bb514ad92de41454fec552f06d61979
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
  quote_sha256: sha256:33771be5e373349262310623989292b2d550f86f2ece2915bb3516a9a3a517bd
  quote_char_len: 113
  pub_year: 2020
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2028-11-24'
---

## Active Consensus (v1 — Effective 2026-11-25)

### Formal Statement
$$\begin{pmatrix} A & B \\ B^* & A^* \end{pmatrix} \begin{pmatrix} X_J \\ Y_J \end{pmatrix} = \Omega_J \begin{pmatrix} I & 0 \\ 0 & -I \end{pmatrix} \begin{pmatrix} X_J \\ Y_J \end{pmatrix}$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| Omega_J | Excitation energy of state J | real | joule |
| A, B | Coupling matrices derived from DFTB Hamiltonian | complex-matrix | joule |
| X_J, Y_J | Excitation and de-excitation vectors | complex-vector | dimensionless |
| I | Identity matrix | matrix | dimensionless |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Non-adiabatic coupling regimes
- **Breaks when:** Strongly correlated electronic systems
- **Breaks when:** Breakdown of the adiabatic approximation in TD-DFT
- **Breaks when:** Singular coupling matrices

### Executable Verification Kernel
```python
import numpy as np
from scipy.linalg import eig

def verify_td_dftb():
    n = 2
    A = np.array([[2.0, 0.1], [0.1, 1.5]])
    B = np.array([[0.5, 0.05], [0.05, 0.3]])
    I = np.eye(n)
    Z = np.zeros((n, n))
    M = np.block([[A, B], [B.conj().T, A.conj().T]])
    S = np.block([[I, Z], [Z, -I]])
    vals, vecs = eig(M, S)
    assert np.all(np.isreal(vals)), 'Eigenvalues must be real for stable systems'
    assert np.isclose(np.linalg.det(M - vals[0] * S), 0.0), 'Characteristic equation failed'
    assert vals.shape == (2 * n,), 'Incorrect number of excitation modes'
    return True

verify_td_dftb()
```

### Canonical Provenance Anchors
- `pmc:PMC7116320` (`pmc:PMC7116320`): The formulation follows the standard linear response TD-DFTB derivation where the Casida-like equations are mapped to a non-Hermitian eigenvalue problem.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
