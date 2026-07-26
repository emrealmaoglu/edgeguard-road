# License Inventory

## Main project

Henüz public software license seçilmedi ve yeniden kullanım izni verilmedi.
A separate `LICENSE` decision is deferred to the human project owner.

## Direct Python dependencies

| Component | Purpose | License | Verification status |
| --- | --- | --- | --- |
| NumPy | CPU array contracts and dummy pipeline | BSD-3-Clause | Verify against the installed release metadata before distribution |
| Pillow | Legal single-image decoding and deterministic RGB preprocessing | HPND | Verify against the installed release metadata before distribution |
| PyYAML | YAML configuration loading | MIT | Verify against the installed release metadata before distribution |
| Pydantic | Configuration and record validation | MIT | Verify against the installed release metadata before distribution |
| pytest | Development tests | MIT | Development-only; verify installed release metadata |
| Ruff | Linting and formatting checks | MIT | Development-only; verify installed release metadata |
| mypy | Static type checking | MIT | Development-only; verify installed release metadata |
| types-Pillow | Pillow type information | Apache-2.0 | Development-only; verify installed release metadata |

Dataset, model, source-repository, and deployment-library licenses will be recorded
before those resources are downloaded or integrated. This table is an inventory, not
legal advice and not a license grant for EdgeGuard-Road.

## Approved external research references

| Component | Scope | License status | Distribution boundary |
| --- | --- | --- | --- |
| PIDNet source at `4c158cf24ce432f0a8cb43364fae38d93cee0dc3` | Fixed external inference spike checkout only; not vendored | MIT for source code | External ignored checkout; vendoring requires a later human decision |
| Two `samples/frankfurt_*_leftImg8bit.png` files in the fixed PIDNet checkout | Primary/fallback internal plumbing inputs only | **OPEN QUESTION**; the PIDNet source MIT license is not asserted to cover images | Never commit, redistribute, use as Cityscapes validation, or use for metrics |
| `PIDNet_S_Cityscapes_val.pt` | Official-repository-referenced checkpoint, approved only for non-commercial academic thesis research | **OPEN QUESTION**; the MIT source license is not asserted to cover weights | Never commit, redistribute, or include in the thesis delivery package |
