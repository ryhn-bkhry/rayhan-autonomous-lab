---
ral_schema: '2.1'
ral_id: RAL-BIO-CN9Y95AS2P
node_type: THEOREM
epistemic_tier: T2_ESTABLISHED
consensus_version: 1
taxonomy_code: bio.biophysics-neuroscience
title: Ricciardi Population Transfer Function
formal_signature:
- phi(mu, sigma, tau_m, V_th, V_reset) = [tau_m * sqrt(pi) * integral_{ (V_reset -
  mu) / sigma }^{ (V_th - mu) / sigma } exp(u^2) * (1 + erf(u)) du ]^{-1}
assumptions:
- Stationary state
- Diffusion approximation (Ornstein-Uhlenbeck process)
- Leaky integrate-and-fire neuronal dynamics
- Constant input statistics
assumptions_hash: sha256:06504cd7bb9d31ee97a251265975214c1133dcdf45026b6a6603569894075d6d
verification:
  oracle_plugin: empirical_stats
  oracle_status: PASS
  oracle_script_sha256: sha256:393b58eb3e08d31cfe5ab952c44a8c082f8d504efb2dfc86ca70a0af7e651061
  verified_at: '2026-11-27T02:00:30Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: pmc:PMC2519166
  source_license: CC-BY-4.0
  source_format: pmc_jats
  source_locator: pmc:PMC2519166
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/e1/f5/e1f560cb18ef568704ea045dcf93cad2a91b9035dca1acd80d76b1ef319c9ca8.xml
  ore_sha256: e1f560cb18ef568704ea045dcf93cad2a91b9035dca1acd80d76b1ef319c9ca8
  quote_sha256: sha256:0a44c6b1060776a0ed1dbec460ebe73e80e6b92e0bf9e4e60f8d19b29b36aa9e
  quote_char_len: 147
  pub_year: 2008
  peer_reviewed: true
  retracted: false
re_evaluate_after: '2027-11-27'
---

## Active Consensus (v1 — Effective 2026-11-27)

### Formal Statement
$$\phi(\mu, \sigma) = \left[ \tau_m \sqrt{\pi} \int_{\frac{V_{reset}-\mu}{\sigma}}^{\frac{V_{th}-\mu}{\sigma}} e^{u^2} (1 + \text{erf}(u)) du \right]^{-1}$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| \phi | Population firing rate | real | Hz |
| \mu | Mean input current | real | V |
| \sigma | Standard deviation of input current | real | V |
| \tau_m | Membrane time constant | real | s |

### Failure Boundaries & Known Pathologies
- **Breaks when:** Breakdown of diffusion approximation when synaptic inputs are sparse or large
- **Breaks when:** Non-stationary input regimes where the Fokker-Planck stationary solution is invalid
- **Breaks when:** High-frequency limit where membrane capacitance effects dominate over resistive leakage

### Executable Verification Kernel
```python
import numpy as np
from scipy.special import erf
from scipy.integrate import quad

def ricciardi_phi(mu, sigma, tau_m, v_th, v_reset):
    def integrand(u):
        return np.exp(u**2) * (1 + erf(u))
    lower = (v_reset - mu) / sigma
    upper = (v_th - mu) / sigma
    integral, _ = quad(integrand, lower, upper)
    return 1.0 / (tau_m * np.sqrt(np.pi) * integral)

mu, sigma, tau_m, v_th, v_reset = 0.02, 0.005, 0.02, 0.02, 0.01
phi_val = ricciardi_phi(mu, sigma, tau_m, v_th, v_reset)

assert phi_val > 0
assert np.isfinite(phi_val)
assert phi_val < 1000.0
```

### Canonical Provenance Anchors
- `pmc:PMC2519166` (`pmc:PMC2519166`): Derived from the stationary Fokker-Planck equation for the membrane potential distribution of a leaky integrate-and-fire neuron under Gaussian white noise input.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
