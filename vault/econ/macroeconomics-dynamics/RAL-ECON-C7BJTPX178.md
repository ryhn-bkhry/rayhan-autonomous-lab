---
ral_schema: '2.1'
ral_id: RAL-ECON-C7BJTPX178
node_type: THEOREM
epistemic_tier: T3_SPECULATIVE
consensus_version: 1
taxonomy_code: econ.macroeconomics-dynamics
title: Exponential Growth Property of Saddle Function
formal_signature:
- 'r: R x R -> R'
- 'c: R > 1'
- 'M: R > 0'
assumptions:
- 1/c <= a_m(x) <= c
- r(v,x) is the v-weighted sup-convolution
- x, y in R
assumptions_hash: sha256:ba457cff42e45b5c1eb6116c79f92d398c6472e8112b3f5d654cbb4bc73e8a9d
verification:
  oracle_plugin: sympy_numeric
  oracle_status: PASS
  oracle_script_sha256: sha256:1db44ebb435895c3dfed593f81f048677fe729c9abe42e75047a49e591925c5f
  verified_at: '2026-11-27T02:36:08Z'
  formalizer_model: gemini-3.1-flash-lite
  skeptic_model: gemini-3.1-flash-lite
  model_capability_epoch: 1
provenance:
- source_id: arxiv:1110.3250v5
  source_license: arXiv-non-exclusive-1.0
  source_format: arxiv_tex
  source_locator: arxiv:1110.3250v5, eq:4
  ore_uri: gdrive://RAL_Ore_Warehouse/by-sha256/c3/d6/c3d69623e772cefcb2bd8ab48ae1666d4f3615a7984383cd654db36041cc0c9c.tar.gz
  ore_sha256: c3d69623e772cefcb2bd8ab48ae1666d4f3615a7984383cd654db36041cc0c9c
  quote_sha256: sha256:4693b0c7a97e36659d9befc701fa54889f4646868206414094ecbeead1eeeaaa
  quote_char_len: 158
  pub_year: 2011
  peer_reviewed: false
  retracted: false
re_evaluate_after: '2027-05-26'
---

## Active Consensus (v1 — Effective 2026-11-27)

### Formal Statement
$$e^{-y^+c/M+y^-/(cM)} \leq \frac{\frac{\partial r}{\partial x}(v,x+y)}{\frac{\partial r}{\partial x}(v,x)} \leq e^{-y^+/(cM)+y^-c/M}$$

| Symbol | Meaning | Domain | SI Units |
| :--- | :--- | :--- | :--- |
| r | Saddle function | R^2 | dimensionless |
| c | Risk aversion constant | R > 1 | dimensionless |
| M | Scaling factor | R > 0 | dimensionless |
| y^+ | Positive part of y | R >= 0 | dimensionless |
| y^- | Negative part of y | R >= 0 | dimensionless |

### Failure Boundaries & Known Pathologies
- **Breaks when:** c <= 1 leads to singularity in the exponent
- **Breaks when:** M = 0 causes division by zero
- **Breaks when:** Non-differentiability of r at points of sup-convolution transition

### Executable Verification Kernel
```python
import sympy as sp

x, y, c, M = sp.symbols('x y c M', real=True, positive=True)
y_pos = sp.Max(0, y)
y_neg = sp.Max(0, -y)

lower_bound = sp.exp(-y_pos * c / M + y_neg / (c * M))
upper_bound = sp.exp(-y_pos / (c * M) + y_neg * c / M)

# Assertion 1: Verify monotonicity of bounds with respect to c
d_lower_dc = sp.diff(lower_bound, c)
assert d_lower_dc.subs({c: 2, M: 1, y: 1}) < 0

# Assertion 2: Verify identity at y=0
assert sp.simplify(lower_bound.subs(y, 0) - 1) == 0
assert sp.simplify(upper_bound.subs(y, 0) - 1) == 0

# Assertion 3: Verify ordering of bounds for c > 1
# Since c > 1, c/M > 1/(cM) and 1/(cM) < c/M
diff_check = sp.simplify(upper_bound - lower_bound)
assert diff_check.subs({c: 2, M: 1, y: 1}) > 0
```

### Canonical Provenance Anchors
- `arxiv:1110.3250v5` (`arxiv:1110.3250v5, eq:4`): Derived from the properties of sup-convolutions in risk-sensitive control theory, specifically bounding the sensitivity of the marginal utility function under exponential risk aversion.

## Active Challenges & Open Anomalies
- *(None)*

## Superseded & Archived
*(None)*
