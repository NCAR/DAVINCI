# FABLE_PLAN — Aerosol Mode-Space Tuning: MERRA-2/GEOS-IT AOD → MODIS/VIIRS

> Tracked planning document (user-requested exception to the untracked-handoff convention).
> Status: **P0-P8 software complete; v1 `SYNTHETIC_READY` rejected; v2
> `passed_pending_user_review`; `SYNTHETIC_READY` remains unset; P9 blocked**. Original
> design written 2026-07-07; synthetic-first pre-flight revision and implementation completed
> 2026-07-10; frozen v1 acceptance executed 2026-07-10; frozen v2 recovery cycle completed
> 2026-07-11.
> Authors: Claude Fable 5 (original planning session with D. Fillmore); Codex (pre-flight review).

## Synthetic acceptance checkpoint (2026-07-10)

- The P0-P8 software path is implemented without adding real-data readers or accessing MERRA-2,
  GEOS-IT, MODIS, or VIIRS data. P9 remains blocked by the gate in §8.5.
- The frozen synthetic calibration policy is stored in
  `analyses/aerosol-tuning/configs/fable-synthetic-calibration.json`; acceptance validates its
  canonical SHA-256 (`de4b0074259ea4bca3819495ae8b2c6a0b1a994d2f0c6e5ebeef9d62b358fc09`),
  file SHA-256 (`45803af3babda5c51eebb49ef50e30db409a23db7500fab4b0f5f906e578dce0`),
  and current code/template identity
  (`15f05d85e4125144ccd60f4190cc944ab02d11ae32525e79c3d42d3938a0be1b`). The selected
  `fable-v1-all-band` policy and all calibration metrics were unchanged after the OSSE pre-flight
  fixes: calibration NRMSE is 0.1442 and null retained-energy/significant fractions are
  0.0273/0.0453.
- A post-freeze, development-only seed (`20260712`) passed every recovery threshold: correlation
  0.9945, origin slope 0.9937, NRMSE 0.1303, filter-target AOD RMSE ratio 0.1311, full-target
  AOD RMSE ratio 0.2660, and holdout AOD RMSE ratio 0.4952. This is not an acceptance seed.
- The first immutable execution root, `acceptance-1179-2358-11`, locked user-supplied seeds
  `1179`, `2358`, and `11` in that order (lock SHA-256
  `ede2a6bed778028e3c793550018a0901ee362ca2bda6cb945c590faff1380f2b`). A NumPy in-place
  broadcast defect stopped every seed before generation in 0.027-0.031 seconds, so this attempt
  exposed no scientific result. The failed root and record are preserved. The generator fix was
  regression-tested at the full 8-year/36x72 OSSE dimensions with development seed `20260712` in
  47.95 seconds and 3,251,428 KiB peak RSS before the calibration provenance was refreshed.
- The same seed order was locked under `acceptance-1179-2358-11-attempt-2` (lock SHA-256
  `11f1651d6d83ec2fcb93f0caec13ecbb4b3eb77e8e6ca5c2b7205af5306612a8`). This was the sole
  evaluative execution. Its original acceptance record SHA-256 is
  `b085fba4b341cbe1f38ba22a4b239e6d5a7d23969608a02bc3919d41b9b220a8`.

| Seed | Corr. | Slope | NRMSE | AOD ratio | Full-target ratio | Elapsed | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1179 | 0.9217 | 0.9228 | **0.4988 fail** | 0.5363 | 0.5860 | 83.28 s | 3.104 GiB |
| 2358 | 0.9108 | 0.8847 | **0.5154 fail** | 0.5575 | 0.6038 | 85.28 s | 3.694 GiB |
| 11 | 0.9160 | 0.9088 | **0.5060 fail** | 0.5459 | 0.5948 | 83.25 s | 3.721 GiB |

- Every fitting/evaluation pipeline, evidence checksum, resource limit, and exclusion diagnostic
  passed. Every seed failed only the frozen `field_nrmse <= 0.35` recovery requirement. Process
  peak RSS is the conservative lifetime `ru_maxrss`, not an isolated per-seed increment.
- Equal-seed means and 95% Student-t intervals are: correlation 0.9162 [0.9026, 0.9298], slope
  0.9054 [0.8575, 0.9534], NRMSE **0.5067 [0.4860, 0.5275]**, filter-target AOD RMSE ratio
  0.5466 [0.5202, 0.5730], and full-target AOD RMSE ratio 0.5949 [0.5728, 0.6169]. Mean excluded
  fraction is 0.6062, off-basis floor NRMSE is 0.2693, best-representable NRMSE is 0.5078,
  holdout AOD RMSE ratio is 0.8246, and clip fraction is zero.
- **Metric-semantics correction:** the historical field named `best_representable_nrmse` is the
  estimate-to-`delta_best_representable_true` error. That truth variable is the supported,
  policy-limited but **unfiltered** in-span correction, while the primary gate compares with the
  filtered observable-mode `delta_filter_target_true`. Therefore 0.5078 is not a best-achievable
  floor and its numerical proximity to 0.5067 does not establish a spatial-basis ceiling. The v1
  rejection is unchanged because the primary target and `field_nrmse` gate were correct; only the
  causal attribution is withdrawn. V2 retains the legacy name solely to read frozen v1 evidence.
- The original failed aggregate correctly rejected acceptance but omitted descriptive statistics.
  A reporting-only code correction now summarizes structurally complete failed gates; no seed or
  scientific pipeline was rerun. The original record remains byte-unchanged, and a read-only
  aggregate supplement beside the attempt root binds that record and has SHA-256
  `c6eecf6741c6858c7d68917b9b52219317831bef28bb44b2d5f40e62837f68da`.
- An independent final audit passed 110/110 identity, manifest, artifact, gate, and record checks;
  all 18 generated NetCDF inputs matched both byte and decoded scientific hashes. The evidence
  validator now also requires the acceptance artifact entry to equal its pipeline-manifest entry.
- Final repository validation reported 1,990 passed and 10 skipped in 146.33 seconds with
  1,816,400 KiB peak RSS; mypy passed 414 source files, and Black and isort are clean.
- `SYNTHETIC_READY` is **rejected for the frozen v1 policy**. These held-out results are not tuning
  inputs, and P9 real-data enablement remains unauthorized.

---

## V2 synthetic recovery checkpoint (2026-07-11)

**Disposition.** The versioned `fable-recovery-v2` implementation, development campaign,
preregistration, calibration, preflight, and one-time acceptance are complete using synthetic data
only. The immutable acceptance record has the literal status `passed_pending_user_review`.
`SYNTHETIC_READY` remains unset, no real data were read, no real-data products were generated, and
P9 remains blocked pending explicit user review of the passing evidence.

### Development and freeze

- V2-D0 and V2-D1 implemented the versioned seed protocol, joint seasonal bias/anomaly fit,
  overlap-constrained zero-sum sensor offsets, saved-fit validation, stagewise diagnostics, null
  policy, calibration/preflight/acceptance lifecycle, locks, attempt ledger, and focused unit and
  pipeline tests. All fitting and evaluation chains entered through
  `PipelineRunner.run_from_config()`.
- V2-D2 used only development seeds `8958027244578499926`, `7058240817492126009`, and
  `6541432702848222996`. The frozen sequential control failed NRMSE on all three seeds, while both
  joint candidates passed every unchanged per-seed and equal-seed gate:

| Development policy | Corr. | Slope | NRMSE | AOD ratio | Full-target ratio | Excluded | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| `v2-sequential-control` | 0.9156 | 0.9133 | **0.5254 fail** | 0.5671 | 0.6122 | 0.6057 | Diagnostic only |
| `v2-joint-seasonal` | 0.9612 | 0.9730 | 0.2794 | 0.2811 | 0.3859 | 0.6057 | Eligible |
| `v2-joint-seasonal-offset` | 0.9803 | 0.9661 | 0.2187 | 0.2267 | 0.3516 | 0.6057 | Eligible |

- The verified development report is
  `analyses/aerosol-tuning/synthetic/fable-v2-development-verified/development.json` (file SHA-256
  `72ef9ffac17468b676d2608a1004fffa3c4b2635939c4f183fc06ef214e52cb0`) and its generation lock
  SHA-256 is `8c4cc508073bd34c7220ba8469db15327573bceb9f500d497190c11548526021`.
  The read-only approval named exactly the two eligible policies; its file/record SHA-256 values are
  `9b3d902a345416211b20e14dd410d370289ad41e6f3d2fc76c72ef44c590c4fa` and
  `928ccd2f54b3d580fe107adc7d5156682c333e00390aa5d59b738f97aa9b5638`.
- Two development-only roots were abandoned before the verified report. `fable-v2-development`
  exposed repeated lazy graph evaluation and was stopped; `fable-v2-development-final` completed
  but was excluded after deep validation found a persisted boolean-attribute representation
  mismatch. Neither root entered approval, preregistration, calibration, preflight, or acceptance;
  `fable-v2-development-verified` is the sole development evidence bound by the freeze.
- The preregistration is read-only at
  `analyses/aerosol-tuning/synthetic/fable-v2-cycle/preregistration.json`. Its file/record SHA-256
  values are `28ce962ea6e342c8e2a1f5fc210a2414026f02590109c46320f2da169ef089c7` and
  `1af0653ef0edc3ac14adff6102287f1b856bdf9f7850958eec23aa68d9bed751`. It binds code
  `dc3174934649c32d858f80a680944a9ffe59f48f07e9c05529a9540d19bba99d`, environment
  `867f428df9e83ff38e7ee72b4134fc0458d88a90af5497e652b22b4fdd6529c7`, protocol
  `905894e197752618ee0564df5cff441cdd31232928e447abb74261ce34c4d3b2`, generator spec
  `8331f69b16bab24f8db36a236601f2310c14ddf4e8f1c39ab381cf511bc399a1`, and thresholds
  `343dbfe35fcb53dcae6b3b402bf93bae5e0431e81d451326e6b36f0c3ee5db1`. Its frozen test evidence
  records 2,105 passed and 10 skipped, mypy clean on 465 files, and Black, isort, and
  `git diff --check` passing.

### Calibration and preflight

V2-C ran both approved candidates on the three declared recovery seeds and the three declared
full-size null seeds. Every per-seed and mean gate, evidence check, and resource check passed. The
equal-seed calibration aggregates were:

| Candidate | Corr. | Slope | NRMSE | AOD ratio | Full-target ratio | Excluded | Null energy | Null significant | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v2-joint-seasonal` | 0.9600 | 0.9559 | 0.2827 | 0.2849 | 0.3935 | 0.4137 | 0.0525 | 0.0596 | Eligible |
| `v2-joint-seasonal-offset` | 0.9799 | 0.9539 | **0.2273** | **0.2360** | **0.3633** | 0.4137 | **0.0256** | 0.0557 | Selected |

The selected policy uses the joint seasonal fit plus overlap-connected, zero-sum relative sensor
offsets; all other frozen v1 policy settings and thresholds remain unchanged. Its recovery and null
seed results were:

| Calibration recovery seed | Corr. | Slope | NRMSE | AOD ratio | Full-target ratio | Excluded | Elapsed | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4720161833425845668 | 0.9786 | 0.9359 | 0.2356 | 0.2451 | 0.3698 | 0.4193 | 95.66 s | 3.361 GiB |
| 7615923448626770708 | 0.9826 | 0.9730 | 0.1997 | 0.2070 | 0.3453 | 0.4098 | 95.95 s | 3.361 GiB |
| 7027338798249911494 | 0.9785 | 0.9527 | 0.2467 | 0.2559 | 0.3748 | 0.4120 | 98.49 s | 3.361 GiB |

| Calibration null seed | Retained energy | Significant fraction | Elapsed | Peak RSS |
|---:|---:|---:|---:|---:|
| 6922119454902611484 | 0.0263 | 0.0441 | 85.34 s | 3.361 GiB |
| 8687442551985640685 | 0.0246 | 0.0775 | 85.99 s | 3.361 GiB |
| 1663300583890477700 | 0.0260 | 0.0454 | 86.53 s | 3.361 GiB |

The immutable calibration file/record SHA-256 values are
`178b08e07c6537baebf98c6647a7dcd5cefaac38a3ef792bfb119aadfccce598` and
`47dfcb4fb5d477b1359848cfe6cf63e352c15b816631a538213f9bd12b638a6b`. Recovery and null
generation-lock SHA-256 values are
`82e6dce296456f45b0c112d8a63ce6e6d0314bf1b9de3bded027da03f089b819` and
`8ebed237e0be561665be20022dd779953840020c9312627dd36658f9ed0b7e68`; the pre-generation attempt
claim SHA-256 is `71493275f6dc8b396837d51588e9226a5fcfd282c73c287e6110f6c811c4194c`.

V2-P then ran only `v2-joint-seasonal-offset` on preflight seed `736479105464814019` and passed:
correlation 0.9792, slope 0.9859, NRMSE 0.2177, AOD ratio 0.2244, full-target ratio 0.3501,
excluded fraction 0.6058, and learned-basis oracle NRMSE 0.0329. It completed in 100.26 seconds at
3.773 GiB peak RSS. The preflight file/record SHA-256 values are
`5adc95565a2f8f1dda54fd6ac35aaab6fea7609ba9d97ab8cda2fb3076ba099b` and
`8880e380393afeb80f6a6bb0ea3c12bb9f1290276588941d64c335b3b89de253`; its generation-lock and
attempt-claim SHA-256 values are
`0f87446b724c90eb085834088d8f03b33faa05e8a572f25785f20c85e4df35b2` and
`1127f441832b8b53956ab27127dcf77296f26fc1522a6f1da16c3ff5ff43c8e1`.

### Acceptance

Only after the frozen preflight passed did V2-A create the acceptance root and lock the exact
ordered tuple `(1969, 2010, 2013)`. This was the single acceptance attempt; every scientific,
evidence, exclusion, and resource gate passed:

| Seed | Corr. | Slope | NRMSE | AOD ratio | Full-target ratio | Excluded | Basis oracle | Elapsed | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1969 | 0.9799 | 0.9488 | 0.2202 | 0.2275 | 0.3501 | 0.6084 | 0.0334 | 100.06 s | 3.791 GiB |
| 2010 | 0.9807 | 0.9666 | 0.2151 | 0.2229 | 0.3493 | 0.6055 | 0.0337 | 101.65 s | 4.010 GiB |
| 2013 | 0.9777 | 0.9637 | 0.2183 | 0.2237 | 0.3486 | 0.6034 | 0.0329 | 100.01 s | 4.010 GiB |

Equal-seed means and 95% Student-t intervals are:

| Metric | Mean | 95% CI |
|---|---:|---:|
| Field correlation | 0.9794 | [0.9756, 0.9832] |
| Field origin slope | 0.9597 | [0.9360, 0.9834] |
| Field NRMSE | 0.2178 | [0.2114, 0.2243] |
| Filter-target AOD RMSE ratio | 0.2247 | [0.2186, 0.2308] |
| Full-target AOD RMSE ratio | 0.3493 | [0.3474, 0.3512] |
| Excluded fraction | 0.6058 | [0.5995, 0.6120] |

The acceptance file/record SHA-256 values are
`cdf1e49193e0d66d3d437a9040aaea58ac62155a4017054bcf1aad6ca09c4b3e` and
`97b01386c2f9beedb58b906878bd1634a74eda5dad030cba574ef6a805c88c26`. Its generation lock
SHA-256 is `ce62113c9b04b0ba113838917b3fd17932673687c342421d091c1edd97b69069`, and the attempt-claim
SHA-256 is `220b99ce14b9ad3644f2d1b7f5ec332ea50e50b9c07e1cc45c2ce050b80a2547`. Calibration,
preflight, and acceptance records and their development/preregistration inputs are read-only; each
record links backward by file and scientific-record hashes.

Acceptance evidence/diagnostic SHA-256 pairs, in seed order, are: seed `1969`,
`c428b09ca18d2c67c788e0ce72aa7caf2901f21bc8d4db055979577dabdd9267` /
`557a12a445d552d15737e7c4bb8c429176ea67df4d9fae67cb6f578aa33bd582`; seed `2010`,
`518bf19a89d2ad52dc4de1b1418710e894afda6421c4e1c0e436b3b068875612` /
`88df748e1dd678ebb6aba675b66630d67c7b08140724f76726f0eb633bc9556b`; and seed `2013`,
`d1d59933f2591c9062680238705e6ca29f04a534b40f0381c9cecda0f00b606e` /
`f1c5506290528f5f10049cd08f785a1ff32695ce5faffb63812bb093336c0f16`. The three phase claims
are stored under
`analyses/aerosol-tuning/synthetic/.fable-recovery-v2-attempts/1af0653ef0edc3ac14adff6102287f1b856bdf9f7850958eec23aa68d9bed751/`
as `calibration.json`, `preflight.json`, and `acceptance.json`.

**Post-freeze provenance audit.** Final review found that the frozen generic phase validators
verify the generation locks and all referenced evidence bytes, but do not themselves compare each
scenario/config/manifest path with that phase's locked root. This is an integrity-proof gap in the
reusable validator contract, not evidence that the completed run used a wrong root. Because those
modules are part of the frozen scientific code identity, they were not changed and the consumed
acceptance seeds were not rerun. A read-only post-hoc audit instead checked all 16 recorded runs
(12 calibration, one preflight, three acceptance): every identity path matched its canonical locked
bundle/run path, every fitting/evaluation YAML document exactly matched the expected synthetic-only
policy rendering, every run input/oracle link resolved into its locked bundle, and every manifest
artifact remained confined to the canonical run output. The audit record is
`analyses/aerosol-tuning/synthetic/fable-v2-posthoc-audit/root-binding-audit.json` (file SHA-256
`dda7139dcb9de650050b9add2b910e478e89a17099df2630672a82e044e55f55`, disposition `passed`). It
contains the exact audit source and binds the preregistration, calibration, preflight, acceptance,
and v2 code identities. This supplement closes root provenance for this completed cycle only; a
future cycle must add the root/config checks to the preregistered validators before exposing new
holdouts. A companion read-only integrity supplement independently recomputes the scaling, writer
config, writer code, scenario, and embedded-file provenance for all 32 corrected MMR files and
binds all five frozen test logs to exact paths and hashes. It passed in a separate fresh process at
`analyses/aerosol-tuning/synthetic/fable-v2-posthoc-audit/integrity-supplement.json` (file SHA-256
`e989dc2d4dc526a64688f8a234d356a9f2dcfd67dab4bd229de2b39423eaed21`). A final canonical
validation also runs in a new process so the reusable validator's same-process cache cannot conceal
an earlier transitive mutation.

### Review-only plots

Spatial recovery snapshots and temporal RMS maps, EOF comparisons, reconstructed PC time series,
and wavelet reconstructions/scalograms for modes 1-2 were rendered for all three acceptance seeds.
The 28 PNG/PDF outputs are listed with checksums in
`analyses/aerosol-tuning/plots/fable-v2-acceptance/diagnostic-record.json` (file SHA-256
`fdeedc64115bd8c0961c2c41df7dbbdceb73cab5d5179add0814952a733a9e82`); its combined diagnostic
NetCDF SHA-256 is `22a3dc86c3b374755bee38a69688eae87af2ae1560a7c8ce74d04c82a6503bd5`.
The 14 PDFs are also available under
`~/Library/Mobile Documents/com~apple~CloudDocs/Claude/FABLE_fable-v2-acceptance_diagnostics/`.

These plots are **review-only and non-preregistered**; they are not acceptance evidence. The
canonical frozen plot adapter rejected the valid three-file `basis_fit` NetCDF collection because
it required a single artifact file; its frozen module SHA-256 is
`18c5e7f15a1c2d213d9184beb441f6250dd8ed7ead4baf74d6f4f37543c245ef`. No immutable input or
source was changed. Plotting instead used the provenance-recorded runtime collection adapter at
`analyses/aerosol-tuning/plots/fable-v2-acceptance/runtime-adapter-record.json` (`input_mutation` is
false; file SHA-256 `dfdee8d599e3dcbb5902613f4dd8935cd95ec417329648f849bae99f007ea6ca`, adapter-source SHA-256
`f6f88188fc0ea4c5c6fe0b8271f93f4a77a998bb37da07082d9debfd49f025d6`). It only opens the
canonical chunk collections recorded by the manifests; the immutable acceptance record remains
byte-unchanged.

---

## V2 synthetic recovery frozen protocol (2026-07-10)

**State and scope.** This section records the reviewed development protocol that was subsequently
implemented and frozen for cycle `fable-recovery-v2`; outcomes are in the 2026-07-11 checkpoint
above. Frozen v1 configs, schemas, identities, records, execution roots, plots, and evidence remain
byte-for-byte historical inputs. V2 added versioned files without rewriting v1. P9 remains blocked,
and every completed v2 activity through acceptance was synthetic-only.

The acceptance challenge is unchanged: `SyntheticTuningSpec.synthetic_osse` keeps its eight-year
axis, grids, split dates, signal amplitudes, basis drift, off-basis term, errors, sensor offsets,
masks/MNAR behavior, support policy, truth variables, primary mask, resource limits, and all §8.1
and §8.5 thresholds. V2 changes the recovery method, not the test distribution or
`delta_filter_target_true`. Development ablations are additive diagnostics and cannot replace or
weaken the full-stress acceptance case.

### V2.1 Diagnosis and bounded method hypothesis

The corrected v1 audit is explanatory evidence only; no v2 parameter is selected from the three v1
acceptance realizations. It found:

| Diagnostic counterfactual | Frozen-v1 NRMSE range | Interpretation |
|---|---:|---|
| Full-grid least-squares target in the learned EOF span, with true bias | 0.0333-0.0334 | Learned spatial span is adequate for the primary target. |
| Oracle daily coefficients reconstructed with the current fitted bias | 0.4098-0.4163 | Sequential monthly bias fitting alone places the result above the gate. |
| Current projected/filtered coefficients reconstructed with true bias | 0.4335-0.4789 | Coefficient recovery is also impaired; errors interact. |
| Frozen v1 production result | 0.4988-0.5154 | Primary-gate result; still rejected. |

The current fitter averages raw `obs-model` innovations into a monthly field before removing the
20/60/150-day EOF anomaly. Finite-window and mask-dependent aliasing therefore contaminates the
bias estimate. A post-hoc source decomposition found that this in-span contamination was larger
than the relative sensor-offset contribution. Observable-mode resolution of 0.951-0.976 also shows
that changing `ridge=1` cannot by itself explain the much larger coefficient attenuation.

V2 therefore tests one production extension: a joint, identifiable seasonal-bias/anomaly fit with
an optional relative sensor-offset term. More EOFs, an innovation-derived basis, adaptive bases,
ridge grids, and threshold changes are not selectable v2 methods. They may appear only in
development diagnostics; changing that boundary requires a reviewed plan revision before any
calibration seed is generated.

For bias-fit observations only, define

```
d_s(t) = P_t B_perp[m(t)] + G_t (Z[m(t)] theta + a(t)) + X_s eta + e_s(t)
Z[m]   = [1, sin(2*pi*(m-0.5)/12), cos(2*pi*(m-0.5)/12)]
```

Here `P_t` selects observed cells, `G_t` is the frozen learned EOF design, `B_perp` is the part of
the 12-month spatial bias orthogonal to the pooled observable EOF span, `theta` is the constant and
annual bias inside that span, `a(t)` is the daily anomaly, and `eta` is an optional constant
relative sensor offset. Fit only the immutable `bias_fit` window by minimizing covariance-weighted
residual energy plus the existing `lambda_a=1` coefficient ridge and a support-aware spatial
Laplacian penalty. The Laplacian uses cyclic longitude, clipped latitude, edge weights normalized by
the median supported-cell fit precision, and fixed dimensionless strength `tau_b=1`. The sensor
offset solve is unregularized after applying its gauge.

The decomposition is identifiable by construction:

- `B_perp[m]` is area/precision-orthogonal to every pooled observable EOF direction for each month.
- `a(t)` is weighted-orthogonal to the three columns of `Z` for each mode over `bias_fit`.
- `sum_s eta_s = 0`; one sensor gives `eta=0`. Offset fitting requires a connected pairwise-overlap
  graph over the fit window and fails otherwise rather than inventing an absolute reference.
- Support is computed from masks/counts and frozen before any innovation value is fitted. A common
  absolute sensor offset remains scientifically unidentifiable and is reported as such.

Use deterministic block-coordinate GLS updates: daily `a`, global `theta`, 12 support-masked
Laplacian fields followed by exact EOF-orthogonal projection, then optional sensor offsets. The
objective must not increase; stop at relative decrease `< 1e-6` or 20 iterations, and treat
non-convergence as fatal. Emit the existing combined monthly `clim_bias`/support interface so
projection, wavelet filtering, scaling, and MMR writing do not change. Add
`clim_bias_perpendicular`, `clim_bias_mode_coefficient`, `sensor_offset` and uncertainty/overlap
counts, pooled observable rank/eigenvalues, objective history, convergence state, and complete
basis/grid/window/policy hashes to the saved projection-fit artifact.

The development menu is fixed to:

| ID | Bias/anomaly fit | Relative offsets | Eligibility |
|---|---|---|---|
| `v2-sequential-control` | Frozen v1 sequential monthly mean | none | Diagnostic only; v1 already rejected. |
| `v2-joint-seasonal` | Joint model above | none | Eligible for v2 calibration if all development gates pass. |
| `v2-joint-seasonal-offset` | Joint model above | overlap, zero-sum | Eligible for v2 calibration if all development gates pass. |

All three retain the selected v1 common-covariance, all-band wavelet, support, scaling, mode count,
and `ridge=1` policy. Development may report ridge-zero analytic and ridge 0.1/0.3 sensitivity, but
those results cannot enter candidate selection. If neither eligible joint candidate passes every
unchanged recovery gate on every development seed and its equal-seed mean, v2 stops before
calibration and this plan must be revised.

### V2.2 Seed roles and leakage boundary

Non-acceptance seeds are fixed without sampling results. For role string `r` and zero-based index
`i`, derive a seed as the low 63 bits of the little-endian integer represented by the first eight
bytes of `SHA-256("fable-v2\0" + r + "\0" + str(i))`. The exact values are:

| Role | Seeds | Permitted use |
|---|---|---|
| `development` | `8958027244578499926`, `7058240817492126009`, `6541432702848222996` | Repeatable ablations, implementation diagnosis, and scoring of all synthetic splits. |
| `calibration_recovery` | `4720161833425845668`, `7615923448626770708`, `7027338798249911494` | Frozen candidates; `calibration` split only. |
| `calibration_null` | `6922119454902611484`, `8687442551985640685`, `1663300583890477700` | Frozen candidates; exact full-size null policy only. |
| `preflight` | `736479105464814019` | Selected policy once; `development_test` split; no retuning after result. |
| `acceptance` | **`1969`, `2010`, `2013` in this order** | Final selected policy once, and only after every freeze check passes. |

The user supplied the ordered acceptance tuple before implementation. It is bound into the v2
protocol identity immediately and is an absolute denylist for generation, ablation, development,
calibration, null, and preflight entry points. No arrays, files, plots, or metrics may be generated
from those seeds before the candidate, code, configs, environment, calibration record, and
preflight result are frozen and hash-valid. The acceptance CLI has no seed override and rejects a
reordered or substituted tuple.

Every role is mutually exclusive and also rejects exposed v1 seeds `1179`, `2358`, `11`,
`20260710`, `20260711`, and `20260712`. Runners exclusively create the seed lock and output root
before generation and refuse an existing root. Once any scientific bytes or metric are produced,
that role/seed cannot be retried. A mechanical pre-generation failure can be retried only in a new
root after immutable proof that zero scientific content was generated and explicit user approval.

Development can inspect only development outputs. Before calibration, write a canonical,
write-once v2 preregistration that binds cycle ID, unchanged generator spec/schema and thresholds,
the exact candidate menu, all seed roles, ranking, configs and CLI entry points, code/environment
hashes, and test status. Any post-freeze scientific/code/config change invalidates the cycle; it
does not reopen calibration. If no candidate passes calibration or preflight, close v2 as rejected
and begin a separately reviewed v3.

### V2.3 Stagewise diagnostic flow

All fitting and diagnostic entry points use `PipelineRunner.run_from_config()`:

```
synthetic inputs (no oracle paths)
  -> basis_train -> joint bias/projection fit -> projection -> wavelet -> scaling
  -> immutable production artifacts and manifests
  -> separate evaluation pipeline loads scaling + read-only oracle
  -> diagnostic-only stage decomposition and report
```

The evaluation pipeline computes these matched-mask, matched-policy stages without writing or
replacing any fit artifact:

1. filtered analytic target projected by full-grid weighted least squares into the learned EOFs
   (`learned_basis_oracle_nrmse`);
2. oracle bias plus noiseless masked production projection (mask/inverse/ridge loss);
3. oracle bias plus noisy masked projection (observation-error loss);
4. fitted bias plus oracle daily EOF coefficients (bias-fit loss);
5. fitted bias plus unfiltered production coefficients (projection loss);
6. post-wavelet coefficients (temporal-filter loss);
7. post-support/scaling `delta_log_applied` (final policy loss).

It also reports true/fitted common bias, identifiable relative sensor offsets, coefficient
correlation/slope/NRMSE before and after filtering, support and COI strata, and non-additive stage
increments. The v2 name for the legacy comparison is
`estimate_vs_unfiltered_in_span_nrmse`; `best_representable_nrmse` remains a read-only v1 alias and
is never a v2 gate. Oracle paths and outputs carry `diagnostic_only=true` and
`eligible_for_calibration=false`; fit configs remain `inputs/`-only and reject `/oracle/`, truth
variables, evaluation artifacts, or windows outside `basis_train`/`bias_fit`.

### V2.4 Calibration, preflight, and acceptance

After the development report and explicit user approval, freeze the eligible menu and run every
candidate on all three full-size `synthetic_osse` calibration-recovery seeds. Score only the
immutable `calibration` split. Add a versioned `synthetic_osse_null` control that copies the exact
OSSE geometry, schedule, masks, reported errors, correlations, outages, and resource profile while
setting physical bias/anomaly/off-basis truth to zero; it does not alter the recovery or acceptance
generator. Run every candidate on all three null seeds.

A candidate is ineligible if any recovery seed or its equal-seed mean fails correlation `>=0.90`,
slope `0.8-1.2`, NRMSE `<=0.35`, filter-target AOD ratio `<=0.70`, full-target AOD ratio `<1`,
exclusion `<=0.80`, resources, or evidence completeness. It is also ineligible if any null seed or
its equal-seed mean exceeds retained-energy or significant-fraction `0.10`. Rank eligible candidates
by equal-seed mean NRMSE, AOD ratio, `abs(slope-1)`, declared simplicity, then ID. Stagewise/oracle
metrics explain results but never weaken a threshold or enter ranking.

Write the selected record atomically, freeze all identities, then run the selected policy once on
the full-size preflight seed and score `development_test`. A failed preflight rejects v2 without
retuning. Only a passing frozen preflight unlocks the exact ordered acceptance tuple
`(1969, 2010, 2013)`. Acceptance retains the existing per-seed plus equal-mean gates, Student-t
intervals, `<1800 s` and `<8 GiB` per-seed limits, complete artifacts/manifests/checksums, and one
immutable execution. A passing program status is `passed_pending_user_review`; only explicit user
review may set `SYNTHETIC_READY` and authorize P9.

### V2.5 Concrete test and entry-point approval

User approval of this entry-point/data-flow design was received before implementation. The listed
v2 tests and entry points are implemented; existing T1-T6 remain unchanged regression tests.

**Pure/unit tests:**

- `test_fable_v2_protocol.py`: canonical/write-once protocol, exact deterministic seed derivation,
  role disjointness, v1/acceptance denylists, mutation detection, and freeze identity.
- `test_joint_projection_bias.py`: exact synthetic decomposition, all three gauges, objective
  monotonicity/convergence, sensor-order invariance, one-sensor identity, disconnected-overlap
  failure, support independence from values, and saved-artifact round trip.
- `test_fable_v2_diagnostics.py`: exact-zero learned-span and stage oracles, matched masks/weights,
  legacy metric rename, non-additive increments, and oracle-ineligible provenance.
- `test_fable_v2_calibration.py`: 3x recovery/null aggregation, any-seed hard rejection,
  deterministic ranking/no-eligible outcome, and atomic immutable record.
- `test_fable_v2_acceptance.py`: exact ordered seed lock, early-generation/reuse/reorder/substitution
  rejection, failed/missing preflight rejection, hash binding, and permanent failed disposition.

**Pipeline integration tests** in `test_aerosol_tuning_v2_pipeline.py`, each entering through
`PipelineRunner.run_from_config()`:

- `test_fable_v2_joint_fit_chain`: compact two-sensor synthetic inputs flow through EOF, joint fit,
  projection, wavelet, and scaling; assert fit outputs, improvement, and no oracle source in fitting.
- `test_fable_v2_stage_diagnostic_chain`: an `exact_micro` production run feeds a separate
  evaluation pipeline; assert exact stage closures and that diagnostic artifacts cannot become fits.
- `test_fable_v2_saved_fit_fresh_runner`: persist the joint fit, start a fresh runner, prove no
  refit, reproduce application output, and reject basis/grid/window/policy/hash mismatch.
- `test_fable_v2_null_policy_pipeline`: compact full-policy null data flow through the complete
  chain and exercise the unchanged false-positive metrics.

Full-size OSSEs remain opt-in developer runs, not routine pytest. Versioned CLIs are
`run_v2_diagnostics.py`, `calibrate_v2_synthetic.py`, `run_v2_preflight.py`, and
`run_v2_acceptance.py`; scripts only render/validate configs and invoke the pipeline. Calibration,
preflight, and acceptance CLIs expose no seed override. The versioned implementation surfaces are a
focused joint-bias core/adapter, schema fields for the fit/offset policy, v2 diagnostics and
protocol/calibration/acceptance modules, v2 YAML templates, and v2 records. New versioned modules
stay below the project's 500-line goal; the shared `_aerosol_contracts.py` is 505 lines after its
v2 additions and is a recorded follow-up split rather than a post-freeze refactor. V1
files/identities remain readable and unmodified.

### V2.6 Phase gates

| Phase | Deliverable | Stop/go gate |
|---|---|---|
| **V2-D0** | Correct metric semantics; add seed protocol and stagewise diagnostic pipeline. | Unit/integration design above approved; compact exact closures green. |
| **V2-D1** | Implement joint seasonal bias/anomaly fit and immutable saved artifact. | Pure solver, pipeline chain, leakage, and fresh-runner tests green. |
| **V2-D2** | Run only the three development seeds; publish stagewise/control/candidate report. | Every unchanged gate passes for at least one eligible method; user approves freeze. |
| **V2-C** | Freeze preregistration; run three recovery plus three null calibration seeds; select atomically. | At least one candidate passes every per-seed/mean gate; otherwise reject v2. |
| **V2-P** | Run the selected policy once on the preflight seed. | All gates/hashes/resources pass; otherwise reject v2 without retuning. |
| **V2-A** | Lock and run acceptance seeds `1969`, `2010`, `2013` once. | User reviews a passing report before `SYNTHETIC_READY`; P9 remains blocked otherwise. |

---

## 0. Pre-flight disposition (2026-07-10)

**Development boundary.** Phases P0-P8 are strictly synthetic: no MERRA-2, GEOS-IT,
MODIS/VIIRS files, real retrieval masks, product downloads, or real-data-derived tuning choices.
Real readers/catalog entries and real-data experiments start only after the `SYNTHETIC_READY`
gate in §8.5 is approved. Generated NetCDF/Zarr and plots are untracked run artifacts.

**Blocking design corrections incorporated by this revision:**

1. `ln(A + epsilon)` does not imply `r = exp(delta)`. Section 2.7 now uses the exact shifted-log
   inverse and an explicit low-AOD policy before applying physical ratio bounds.
2. A global EOF can reconstruct into a cell that was never observed. Section 2.3 now defines a
   monthly spatial support field that is applied during reconstruction, making the promised
   no-observation/no-correction behavior explicit rather than assumed.
3. The live derived-analysis API accepts exactly one `source`, but projection/scaling require
   several named inputs. Section 4.1 defines one named-input contract used by schema validation,
   topological ordering, and execution, plus required/fatal-chain semantics.
4. Projection previously had no time-varying model field. A first-class `aod_preprocess` analysis
   (§4.2) now produces the one daily AOD/log-AOD pseudo-source consumed by both EOF training and
   innovation projection.
5. The live full, float64 SVD and monolithic eager artifact path do not scale to the proposed
   real problem. Truncated/chunk-aware EOF and artifact benchmark gates now precede real data.
6. Daily MODIS support is not catalog-only: timestamp cadence, canonical variable naming, and QA
   behavior require reader work. That work is explicitly deferred until after `SYNTHETIC_READY`.

**Historical repository baseline at pre-flight (not caused by this document change):** on
`develop` at `2b62d7f`, in the
active `davinci` environment, pytest reported 1,749 passed, 10 skipped, and 5 failed because
`pycwt` is not installed; mypy reported 2 errors; Black flagged 1 file; isort passed. Restore or
explicitly disposition those baseline gates before implementation. The required environment name
throughout this plan is `davinci`.

---

## 1. Context

**Goal.** Tune MERRA-2 (later GEOS-IT) aerosol mass mixing ratios (MMRs) so that model AOD
better tracks MODIS/VIIRS AOD. The corrected MMRs feed a radiative transfer model.

**Method (user's concept, refined).** Compute EOFs of the analysis (model) AOD field; project
satellite AOD *innovations* (obs − model) onto that basis with a missing-data-aware regularized
least-squares (reduced-space optimal interpolation); wavelet-filter the resulting obs-space PC
time series (denoise + band-select + gap-bridge); reconstruct a correction field from the
filtered coefficients and the analysis EOFs; apply it as a **uniform per-column scale factor to
all configured aerosol species MMRs. Exact optical scaling is conditional on a fixed homogeneous
forward operator (species/size/composition/RH/meteorology unchanged); the synthetic optical oracle
tests that condition before any claim is transferred to a real radiative-transfer model.

**Scope decisions (original choices plus pre-flight corrections):**

| Decision | Choice | Rationale |
|---|---|---|
| Initial evidence | **Synthetic only through `SYNTHETIC_READY`** | Separates algorithm/software correctness from reader quirks and prevents tuning against real observations before the recovery limits are known. |
| Projection under missing data | Ridge/OI innovation projection (reduced-space OI, Kaplan et al. 1998) | EOFs are not orthogonal on the observed subdomain; naive inner products alias the seasonal mask into the PCs. Innovation projection makes "mode unobserved → zero correction" the graceful default. Multi-sensor blending falls out of the same solve. |
| Pipeline integration | **(A)** chained derived analyses in the existing `analyses:` block | Follows repo rule "all analyses through DAVINCI pipelines"; every intermediate is a pseudo-source → QA plots free; pipeline-level integration tests possible. |
| Working space | **shifted log-AOD** (`ln(AOD + ε)`) | AOD is closer to Gaussian and variance is homogenized, but the shift requires the exact inverse in §2.7; `exp(Δ)` alone is not the physical MMR scale factor. |
| Correction structure | `Δ = Δ_clim(month, x) + Δ_anom(t, x)`, then exact physical `r` | Separates systematic bias from variability transfer in transformed space; a stored monthly support field gates both terms where synthetic observations provide no evidence. |
| Mode-space grid | Everything at **1°** for the real target (coarser in CI) | Matches the MODIS L3 grid and bounds the state size. Treating the truncated correction as smooth enough for native-grid interpolation is a tested modeling assumption, not a no-loss theorem. |
| Analysis product | Synthetic MERRA-like first; **MERRA-2 first real product**; GEOS-IT follow-on | Core behavior is proven without real files. MERRA-2 then exercises the existing reader; GEOS-IT begins with a grid/collection audit, not an assumed clone. |
| Obs | Synthetic daily L3 sensors first; **MODIS Aqua (`MYD08_D3`) first real sensor**; projection accepts a list from day one | Complementary masks and precision weighting are tested with known truth before Terra/VIIRS or retrieval-specific QA is introduced. |
| Wavelet role | (a) significance-gated denoise + (b) band selection + (c) temporal gap-bridging | Requires retaining complex CWT coefficients and adding inverse CWT — the current `wavelet` analysis keeps only `|W|²`. |
| Domain / period | Small global synthetic cases first; later **decades-long daily** real EOF training | CI and synthetic OSSE establish correctness; truncated/chunked solver benchmarks gate the decades-long run. |
| Cadence | Daily correction, applied to 3-hourly `inst3_3d_aer_Nv` via log-linear time interpolation | Once-daily sun-synchronous obs cannot constrain the diurnal cycle; MERRA-2's diurnal shape is accepted by design. |

---

## 2. Method — complete mathematical specification

### 2.1 Notation & preprocessing

- Grid cells `x` on the 1° L3 grid; area weight `w(x) = cos(lat)` (the same metric the EOF SVD
  uses via its `sqrt(cos lat)` field weighting, `analysis/eof.py:45`).
- Model AOD `A_m(t, x)`: produced once by `aod_preprocess` (§4.2), then consumed unchanged by
  the EOF and projection analyses. For the real follow-on this is MERRA-2 `TOTEXTTAU` from hourly
  `tavg1_2d_aer_Nx`, sampled at ~13:30 local solar time (nearest `UTC = 13.5 - lon/15` per
  longitude column), area-weight coarsened 0.5°×0.625° → 1°, and stamped by calendar day.
  Synthetic sources reproduce the same cadence and longitude-dependent selection. The raw model
  source loads one adjacent UTC day on each side; the preprocessor clips its daily output back to
  the requested window so dateline columns and date-only boundaries cannot lose an edge day.
- Log transform: `y = ln(A + ε)`, `ε = 0.01` (config `log_epsilon`). Applied identically to
  model and obs **after** linear-space coarsening. Its inverse and the MMR ratio are defined in
  §2.7; no code may substitute `exp(Δ)` for that ratio.
- Obs `y_o^s(t, x)` for sensor `s`, defined on the observed set `Ω_s(t)` (the day's valid L3
  cells, QA-filtered). Time alignment is by calendar day.

### 2.2 Basis (EOF training)

Anomalies about the monthly climatology of the training period (`remove_seasonal_cycle: true`):
`y'_m(t,x) = y_m(t,x) − C(month(t), x)`. Weighted SVD (existing `_svd_decompose`,
`analysis/eof.py:122`) yields:

- unit-variance, mutually uncorrelated PCs `p_k(t)` (unrotated), and
- regression patterns `E_k(x)` in log-AOD units (`_patterns_from_pc`, `analysis/eof.py:164`),
  so that `y'_m ≈ Σ_k p_k(t) E_k(x)`.

`n_modes` K ≈ 50 to start; truncation is selected on the synthetic validation split, then frozen
for the synthetic test ensemble. Real selection later uses scree/North's-rule plus sensitivity
tests. The real-size path must use a truncated/randomized, chunk-aware solver; the live full
float64 SVD is retained only as a small-case reference oracle.

**Projection-basis constraints:** `rotation: none` and `standardize: false`. Rotated PCs are not
uncorrelated, and standardized EOF patterns are not in log-AOD units unless explicitly
de-standardized. Violations are clear cross-spec validation errors.

### 2.3 Innovation and climatological bias term

Per sensor, day, and observed cell:

```
d_s(t, x) = y_o^s(t, x) − y_m(t, x),   x ∈ Ω_s(t)        # log(obs/model)
```

Split into systematic + anomaly parts, `d_s = b(month, x) + d'_s`:

- `b_mean(m,x)` is the precision-weighted mean of `d` over the **bias-fit window only**, retaining
  per-sensor counts and standard error. Within cells meeting `f_min`, form `b_hat` with two
  mask-aware 3x3 passes (cyclic longitude, clipped latitude); restore unsupported cells afterward.
  The baseline synthetic sensors share one physical bias; a stress scenario adds sensor offsets
  to quantify the limitation of a common term.
- Let `f(m,x)` be the number of unique `bias_fit` days in month `m` with at least one valid sensor,
  divided by the total `bias_fit` calendar days in that month. With defaults `f_min=0.20` and
  `f_full=0.50`, define `S0=0` below `f_min`, `S0=(f-f_min)/(f_full-f_min)` between the bounds,
  and `S0=1` at/above `f_full`. Smooth `S0` with two mask-aware 3x3 passes (cyclic longitude,
  clipped latitude) independently for each calendar month, then restore `S=0` wherever
  `f<f_min` and clip to `[0,1]`. Define `b_applied=S*b_hat`; the reconstructed anomaly is also
  multiplied by `S` in §2.7. This is an explicit post-estimation confidence taper. Observations at
  `S=0` are excluded from projection so a newly appearing unsupported cell cannot alter global
  coefficients elsewhere.
- `b_hat` is bounded in transformed space by explicit `delta_bounds`; asymmetric physical
  `r_bounds` are enforced only by the exact conversion in §2.7.
- The projection (§2.4) acts on `d' = d-b_hat`; tapering occurs only at reconstruction, so the
  un-applied fraction of a partial-support bias cannot leak into anomaly coefficients.

Note the basis climatology `C` cancels in `d` (both obs and model would subtract the same `C`),
so the innovation needs no climatology handling beyond `b_hat`.

### 2.4 Per-day projection (reduced-space OI / ridge)

For each day `t`, stack all valid supported sensor/cell observations into `d(t)` and the matching
basis rows into `G(t)`. Define `C_obs(t)` as the **effective joint covariance** of that stacked
innovation vector, including the intended area representation and any cross-sensor blocks:

```
a_hat(t) = argmin_a [d' - G a]^T C_obs^-1 [d' - G a] + lambda ||a||^2

<=> (H(t) + lambda I) a_hat(t) = g(t),
    H(t) = G(t)^T C_obs(t)^-1 G(t)               (KxK)
    g(t) = G(t)^T C_obs(t)^-1 d'(t)              (K)
```

For independent errors the default is `C_obs,ii = sigma_i^2 / cos(lat_i)`, exactly reproducing the
original area-weighted objective; thus `sigma_i` is the unweighted log-error scale and `C_obs` is
the covariance used by the solve. Structured v1 covariance is block-diagonal plus configured
low-rank common sensor modes, never an assembled dense global matrix. `lambda=1` is a baseline
prior assumption, not a physical identity: it is MAP only under `a~N(0,I)` and `C_obs`. Absolute
covariance scale and ridge strength are confounded, so one is frozen while the other tunes only on
calibration. Overlapping sensors are not counted independent when a common-mode block is configured.

Properties that answer the seasonal-mask question:

- Cross-mode leakage from mask-broken orthogonality is handled exactly (full `H`, not its diagonal).
- An exact null eigen-direction of `H` is set to zero by the prior. Near-null mixed directions are
  shrunk according to their eigenvalues and may cross-talk across named modes; geographic
  extrapolation is controlled separately by `S`, not inferred from `H_kk` alone.
- Cost includes assembly of `H`, nominally `O(T Ncell K^2)`, not only the K×K solve. The plan
  requires a benchmark of chunked/batched assembly before any real-data claim (§6).

### 2.5 Observability diagnostics (stored with the PCs)

- `resolution(t, mode)` = `diag[(H+λI)⁻¹ H]` ∈ [0, 1) — recovered fraction of a unit true amplitude.
- `coverage(t, mode)` = `Σ_{x∈Ω} w E_k² / Σ_x w E_k²` — mask coverage of mode k's variance.
- `n_obs(t, sensor)` — observed-cell counts.
- `posterior_variance(t, mode)`, minimum/maximum resolution eigenvalue, condition number, and
  effective observed rank — diagonal resolution alone cannot expose poorly observed combinations.
- `spatial_support(month, lat, lon)`, support counts, and climatological-bias standard error.
- Flag `(t, k)` when `resolution < ρ_min` (default 0.3, config `min_resolution`); flagged entries
  are treated as **gaps** by the wavelet filter rather than trusted as shrunk-to-zero values.

### 2.6 Wavelet filter (per mode k)

1. Set flagged entries of `a_hat_k` to NaN. Interpolate only interior gaps no longer than
   `max_bridge_days`; record `bridged(time, mode)` and synthesized fraction. Long gaps split the
   series into independent segments; outside an accepted segment the anomaly correction is zero,
   not edge-held or linearly extrapolated.
2. Fix one `dt,dj,s0,J` and period grid from the full requested axis and finite configured `T_max`.
   Require `min_segment_days >= 2*T_max`; shorter segments emit zero anomaly and invalid QA.
3. For each accepted segment fit/remove `mu + beta*(t-t_bar)`, estimate AR(1), and compute the CWT
   on the common scale grid. Mean/trend are re-added only inside that segment and intentionally
   bypass the configured wavelet band.
4. If `keep_significant: true`, zero coefficients failing the pointwise AR(1) significance test.
   The calibrated v1 all-band policy sets this false but still records significance diagnostics.
5. Zero coefficients outside the configured period band `[T_min, T_max]` (`band:`); COI is kept
   for reconstruction but is excluded from acceptance scoring; the retained COI fraction is stored.
6. Inverse CWT (`pycwt.icwt`, Torrence & Compo 1998 eq. 11), restore segment mean/trend, then apply
   a cosine taper over `min(T_max, segment_length/4)` at each segment edge so correction approaches
   identity continuously. Power is NaN outside segments; global spectra/significance aggregate
   segment values with valid-sample weighting on the shared period grid.
7. Record per mode: retained variance fraction, and the unfiltered icwt round-trip relative error
   (Morlet reconstruction is approximate, ~few %); warn above 5 %.

Pointwise AR(1) significance is initially a filtering heuristic because ridge shrinkage and gap
fill make its nominal probability imperfect. A zero-correction synthetic ensemble measures the
false-positive rate. If its frozen threshold fails, FDR or synthetic Monte Carlo significance is
required before `SYNTHETIC_READY`; the test may not be weakened to preserve the heuristic.

### 2.7 Reconstruction, scale factor, application

```
Δ_requested  = b_applied(month(t),x) + S(month(t),x) * Σ_k a_tilde_k(t) E_k(x)
Δ_rmin       = ln([A_m*r_min + ε] / [A_m + ε])
Δ_rmax       = ln([A_m*r_max + ε] / [A_m + ε])
Δ_safe       = clip(Δ_requested, Δ_rmin, Δ_rmax)
Δ_safe       = 0 where A_m < aod_floor or S = 0
A*_raw       = [A_m + ε] * exp(Δ_safe) - ε
r_1deg       = clip(A*_raw / A_m, r_min, r_max)           # only A_m>=floor, S>0; roundoff clip
r_1deg       = 1 elsewhere
A_target     = A_m * r_1deg
Δ_applied    = ln(A_target + ε) - ln(A_m + ε)
ln_r_native  = periodic_bilinear(ln(r_1deg) -> native grid)
S_native     = periodic_bilinear(S -> native grid)
ln_r_native  = 0 where S_native = 0
ln_r_3hr     = linear-in-time interp(ln_r_native)         # only inside correction coverage
r_3hr        = exp(ln_r_3hr); outside coverage r_3hr = 1 (or file is skipped by config)
q_tilde_i    = r_3hr * q_i                                # configured aerosol species only
```

The AOD-dependent transformed bounds are applied **before** exponentiation (with stable
`log1p`/`expm1` forms where appropriate), so an extreme reconstructed anomaly cannot overflow.
This ordering keeps the physical MMR multiplier positive and bounded, preserves exact
`A_target = A_m * r_1deg`, and makes daily-to-3-hourly interpolation multiplicative. Exact identity
holds at analysis-grid `S=0` cells and native cells whose interpolated support is zero; boundary
cells taper continuously. The writer does not hold the first/last correction into pre-observation
or post-observation files. Counts are stored
for low-AOD identity, spatial-support identity, lower/upper clipping, and outside-coverage identity.

For a fixed homogeneous optical operator (same species list, size/composition, RH, meteorology,
and optical coefficients), uniform scaling makes its column AOD scale by exactly `r`. The
synthetic generator includes that forward-operator oracle (§7.4). This is an optical correction,
not a chemically or mass-balanced aerosol analysis. Speciation fractions and vertical profile
shape are preserved; configured gas-phase tracers and fill values are not scaled.

### 2.8 Success criteria

- **Synthetic algebra gates:** noiseless full-mask projection, shifted-log round trip, native/time
  interpolation, and MMR optical closure meet numerical tolerances.
- **Synthetic recovery gates (§8.1):** on untouched seeded test cases, corrected AOD and `Δ_applied`
  improve field error against the latent nature state, not noisy observations. Report coefficient
  correlation **and** slope/bias/NRMSE after weighted mode matching, plus field metrics by season,
  latitude, observation support, and resolution bin. Include multiple seeds/confidence intervals,
  clipping rate, null false-positive rate, and the matched filtered-target learned-span oracle.
- Cells with `S = 0` have `r = 1` exactly. With the support gate disabled in an explicit research
  scenario, global EOF extrapolation is allowed and no geographic no-correction claim is made.
- **Real data (deferred):** corrected AOD vs assimilated Aqua is an assimilation diagnostic;
  external success is improvement against predeclared unassimilated AERONET and/or a wholly
  withheld sensor. Basis, bias/support, and hyperparameters freeze before external evaluation.

---

## 3. Architecture & data flow

```
sources:   model_hourly             sensor_a_raw + sensor_b_raw       native MMR files
                 │                            │ QA
analyses:  1. aod_preprocess ────────────────┘
              model_daily + sensor_daily {aod, log_aod, valid/support metadata}
           2. eof              basis {eofs, pc, variance, climatology, ...}
           3. eof_projection   a_hat + bias/support/posterior diagnostics
           4. wavelet_filter   a_tilde + bridge/COI/reconstruction diagnostics
           5. aod_scaling      mode-grid r/A_target + chunked analysis-grid artifact
           6. mmr_writer       corrected native files + checksummed manifest
plots:     eof_pattern, eof_scree, pc timeseries, wavelet_scalogram (QA on 2–3),
           spatial maps of r / b_hat / aod_target
oracle:    truth sidecar is absent from fitting sources; only the post-fit evaluation loads it
```

Steps 1–5 are required derived analyses with named inputs (§4.1). Large outputs are persisted by
an analysis-declared, chunked artifact policy and remain lazy in pipeline context. Step 6 is a
side-effect-capable result with atomic per-file writes and a checksummed manifest; it is fatal on
partial failure unless an explicit resume policy proves each existing output complete.

### 3.1 Synthetic development reference YAML

This is the first executable target. Paths are created under `tmp_path` in CI or an ignored
synthetic run directory. The oracle file is deliberately absent from the config.

```yaml
analysis:
  start_time: "2001-01-01 00:00:00"
  end_time:   "2006-12-31 23:59:59"
  output_dir: ${FABLE_SYNTH}/output
  log_dir:    ${FABLE_SYNTH}/logs

sources:
  model_hourly:
    type: generic
    files: ${FABLE_SYNTH}/inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4
    variables: { TOTEXTTAU: { units: "1" } }
    time_padding: "1D"             # NEW: retained through load; output clips to analysis window
  sensor_a_raw:
    type: satellite_l3
    files: ${FABLE_SYNTH}/inputs/obs/sensor_a.nc
    variables: { aod_550nm: { units: "1" }, reported_sigma_log: {}, QA: {} }
    qa_variable: QA
    qa_values: [3]
  sensor_b_raw:
    type: satellite_l3
    files: ${FABLE_SYNTH}/inputs/obs/sensor_b.nc
    variables: { aod_550nm: { units: "1" }, reported_sigma_log: {}, QA: {} }
    qa_variable: QA
    qa_values: [3]

analyses:
  model_daily:
    type: aod_preprocess          # NEW
    source: model_hourly
    variable: TOTEXTTAU
    sample_local_time: 13.5
    day_anchor_hour: 12.0
    target_grid: 30.0             # 6x12 CI grid; real config uses 1.0
    log_epsilon: 0.01
    required: true

  sensor_a_daily:
    type: aod_preprocess
    source: sensor_a_raw
    variable: aod_550nm
    uncertainty_variable: reported_sigma_log
    day_anchor_hour: 12.0
    target_grid_from: model_daily
    log_epsilon: 0.01
    required: true

  sensor_b_daily:
    type: aod_preprocess
    source: sensor_b_raw
    variable: aod_550nm
    uncertainty_variable: reported_sigma_log
    day_anchor_hour: 12.0
    target_grid_from: model_daily
    log_epsilon: 0.01
    required: true

  aod_basis:
    type: eof
    source: model_daily
    variable: log_aod
    n_modes: 3
    remove_seasonal_cycle: true
    standardize: false
    rotation: none
    solver: full                    # reference oracle for small CI; real path uses randomized
    fit_window: { start: "2001-01-01", end: "2002-12-31 23:59:59" }
    required: true

  obs_pcs:
    type: eof_projection
    basis: aod_basis
    model: model_daily
    model_variable: log_aod
    obs:
      - { source: sensor_a_daily, variable: log_aod, error_variable: obs_error_std }
      - { source: sensor_b_daily, variable: log_aod, error_variable: obs_error_std }
    ridge: 1.0
    bias_fit_window: { start: "2003-01-01", end: "2004-12-31 23:59:59" }
    clim_bias: true
    spatial_support: monthly_taper
    support_min_fraction: 0.2
    support_full_fraction: 0.5
    support_smoothing_passes: 2
    delta_bounds: [-1.6094379, 1.6094379]
    min_resolution: 0.3
    required: true

  filtered_pcs:
    type: wavelet_filter
    source: obs_pcs
    variable: pc
    keep_significant: false
    significance_level: 0.95
    band: { min: 4, max: 180, units: days }
    max_bridge_days: 7
    min_segment_days: 360
    omega0: 6.0
    required: true

  scaling:
    type: aod_scaling
    basis: aod_basis
    projection: obs_pcs             # bias, support, diagnostics
    coefficients: filtered_pcs
    model: model_daily              # A_m for exact shifted-log inverse
    r_bounds: [0.2, 5.0]
    aod_floor: 0.001
    required: true

  corrected:
    type: mmr_writer
    scaling: scaling
    files: ${FABLE_SYNTH}/inputs/mmr/*.nc4
    species: null                  # synthetic files contain the complete default list
    output_dir: ${FABLE_SYNTH}/corrected
    time_interp: log_linear
    outside_coverage: identity
    overwrite: false
    required: true

plots:
  basis_maps:  { type: eof_pattern, source: aod_basis, variable: eofs }
  basis_scree: { type: eof_scree,   source: aod_basis, variable: explained_variance }
  pc1:         { type: timeseries,  source: filtered_pcs, variable: pc, mode: 1 }
  pc1_scal:    { type: wavelet_scalogram, source: filtered_pcs, variable: power, mode: 1 }
  r_map:       { type: spatial, source: scaling, variable: r }
```

The generator's immutable schedule is 2001-02 `basis_train`, 2003-04 `bias_fit`, 2005
`calibration`, and 2006 `development_test`; the explicit config windows must hash-match that
schedule. Production saved-fit semantics use explicit basis/bias artifact references and are
implemented/tested synthetically rather than deferred to real-data evaluation.

### 3.2 Deferred real-data mapping

After `SYNTHETIC_READY`, replace the three synthetic raw sources with MERRA-2 hourly AOD and
QA-filtered daily MODIS sources, change `target_grid` to 1°, choose the randomized solver, and use
saved train/calibration artifacts. The real config uses canonical reader output `aod_550nm`, not
an HDF SDS name. It uses explicit end-of-day timestamps and source padding. No mathematical or
pipeline contract is allowed to change merely to make the real run pass; reader/metadata defects
are fixed at their boundary and covered by reader integration tests.

---

## 4. New components (file by file)

All new implementation modules stay < 500 lines (project goal); pure math lives in plain functions
for unit testability, analysis classes are thin adapters — mirroring `eof.py`'s structure. Test
modules follow the repository's existing organization and may group broader scenario contracts.

### 4.1 Named-input execution and result contract

- Add `AnalysisSpecBase.input_refs() -> dict[str, str]` and `required: bool = false`. Existing
  single-source specs return `{"source": spec.source}`. New specs return named roles such as
  `basis`, `model`, `projection`, `coefficients`, and `obs[0]`; raw and derived inputs use the
  same resolver. Schema validation, DAG construction, and runtime resolution all call this one
  method rather than maintaining type switches that can drift.
- Make legacy `DerivedAnalysis.analyze()` a concrete compatibility hook (remove `@abstractmethod`;
  its default raises `NotImplementedError`) and make `analyze_inputs(inputs, spec, runtime)` the
  stage entry point, where immutable
  `AnalysisRuntime` supplies the requested analysis window and artifact service without exposing
  mutable pipeline context. Its default adapter calls the existing `analyze(inputs["source"],
  spec)`, preserving current analyses. Multi-input analyses override the named-input method.
- Add `AnalysisResult(dataset, artifacts, manifest_entries)`; plain `xr.Dataset` returns are
  adapted for backward compatibility. Artifact-producing analyses declare policy in their result,
  not via `if spec.type == "gridded_analysis"` in the stage.
- If a required analysis fails, the stage is `FAILED`, descendants are recorded as dependency
  blocked, and no pipeline success is possible. Optional independent analyses retain soft-failure
  behavior. A writer error is always fatal after cleaning its temporary file; already finalized
  outputs remain listed for explicit resume, never silently treated as a complete run.
- Add `DataGeometry.ARTIFACT` for manifest-only pseudo-sources; it is excluded from pairing and
  plotting unless a renderer explicitly declares support.

### 4.2 `davinci_monet/analysis/aod_preprocess.py` — `aod_preprocess`

- Named inputs are `source` and optional derived `target_grid_from`; the latter participates in
  schema validation and the DAG even though only its coordinates are used. Output is
  `aod(time,lat,lon)`, `log_aod`, `valid`, and optional standardized `obs_error_std`, with attrs and
  source hashes. Both model and sensor paths use the same ordered operations: QA/finite screening
  (`AOD >= 0` valid; negative invalid), optional local-solar-time sampling, area-weighted regrid,
  then shifted log. `aod_floor` controls scaling identity separately and does not invalidate zero.
- Initial sensor errors already share the target grid and pass through exactly. If uncertainty is
  coarsened, use the declared covariance of a weighted mean (`var=sum_ij alpha_i alpha_j C_ij`),
  never an ordinary mean of standard deviations; reject missing covariance assumptions. Optional
  `common_factor_variables` are propagated through the same linear regrid weights and keep their
  `(time, common_mode, lat, lon)` contract.
- `time_padding` is consumed by `LoadSourcesStage` and added to `SOURCE_LOADER_CONFIG_KEYS` so it
  is never forwarded to `xr.open_dataset`. Output is clipped to the
  requested calendar-day window only after longitude-dependent sampling. Synthetic tests cover
  dateline selection, a date-only last day, adjacent UTC days, and non-commensurate grids.

### 4.3 `davinci_monet/analysis/projection.py` — `eof_projection`

- Named inputs: one basis, the preprocessed daily model field, and one or more preprocessed obs
  fields. Output geometry is GRID with `pc(time, mode)`, `resolution`, `coverage`, posterior
  diagnostics, `clim_bias`, `clim_bias_applied`, `spatial_support`, counts/standard errors, and
  `n_obs(time, sensor)`.
- The output time axis is the complete daily `model` axis. Every sensor is reindexed to it before
  masking; a missing file/day and a present-but-all-invalid day both remain explicit rows with
  `n_obs=0`, zero coefficients/resolution, and gap QA.
- Pure functions cover innovation, precision-weighted bias/support, `C_obs` construction, one-day
  solve, posterior diagnostics, and chunked orchestration. The small exact solver and the batched
  path are compared on identical synthetic inputs.
- Validate basis metadata (`rotation=none`, `standardize=false`, log epsilon/grid identity), input
  coordinates, error positivity/covariance shape, nonempty time overlap, and frozen fit artifacts.
- Days with no usable obs emit zero projected anomaly and zero resolution. They may be short-gap
  bridged only under §2.6; long gaps remain zero anomaly. Bias/support are saved as reproducible
  artifacts so validation/test runs cannot refit them.

### 4.4 `davinci_monet/analysis/wavelet_filter.py` — `wavelet_filter`

- Output geometry is SPECTRUM. It emits filtered `pc(time, mode)`, `power`, normalized
  `power_significance`, `coi`, `global_power`, `global_significance`, period units and
  `wavelet_quantity`, plus `bridged`, `valid_segment`, `retained_variance`, `recon_error`, and
  `synth_fraction`. All segments share the configured period coordinate; outside-segment values
  are NaN/invalid and global quantities are valid-sample weighted. This is the exact existing
  `wavelet_scalogram` contract, extended by `mode`.
- It implements the bounded-gap/segment rules in §2.6. Mode-selected renderer tests verify the
  actual artist/data contract; PNG-size-only assertions are insufficient.

### 4.5 `davinci_monet/analysis/scaling.py` — `aod_scaling`

- Named inputs: basis, unfiltered projection (bias/support), filtered coefficients, and daily
  model AOD. Output geometry is GRID and stays on the analysis grid only:
  `r`, `delta_log_requested`, `delta_log_applied`, `aod_target`, support/clip/low-AOD masks, and
  per-time fractions. This avoids the invalid attempt to put different mode/native grids on the
  same `(time, lat, lon)` dimensions.
- Pure functions implement reconstruction and the exact shifted-log conversion in §2.7. The
  scaling artifact is time-chunked and lazily reopened. The writer reads the needed daily chunks,
  interpolates `ln(r)` to each file's native grid with periodic longitude, then interpolates in
  time; a decades-long native-grid ratio is never retained in memory or one monolithic file.

### 4.6 `davinci_monet/analysis/cwt_core.py` — shared CWT

- Extract from `WaveletAnalysis.analyze()` a typed `CWTResult` holding complex coefficients,
  scales, periods, COI, AR(1), and significance arrays, plus `cwt_reconstruct(...)` around
  `pycwt.icwt`.
- `WaveletAnalysis` remains behavior-identical and `wavelet_filter` is the second consumer. Pin
  and install `pycwt==0.4.0b0` without dependency resolution in `davinci` before this phase.

### 4.7 `davinci_monet/analysis/mmr_writer.py` — `mmr_writer`

- Streams the direct file glob one file at a time. It rejects input/output aliasing, validates the
  complete configured species set before writing, reads only bracketing daily ratio chunks,
  applies periodic spatial plus log-linear temporal interpolation, and follows the explicit
  outside-coverage policy (`identity`, `skip`, or `error`; never edge-hold by accident).
- Default aerosols are `DU001-005, SS001-005, SO4, BCPHOBIC, BCPHILIC, OCPHOBIC, OCPHILIC`;
  real names are audited only in the real phase. Gas tracers, fill values, coordinates, unrelated
  fields, dtype/dim order/attrs/compression/chunks are preserved.
- Write to a same-filesystem temporary path, fsync/close/validate, then `os.replace`. `overwrite`
  defaults false. Resume accepts an existing file only when its input/config/scaling hashes match.
- Return an ARTIFACT pseudo-source and manifest entries with final checksums, coverage/clip stats,
  config/code/scenario hashes, and file status. The run manifest consumes these entries directly.

### 4.8 Config schema (`config/schema.py`)

- Add `AnalysisSpecBase`, `AODPreprocessSpec`, `EOFProjectionSpec` with nested obs/covariance
  entries, `WaveletFilterSpec`, `AODScalingSpec`, `MMRWriterSpec`, and evaluation-only
  `KnownTruthSpec`; extend the union/dispatcher.
- `EOFSpec` gains `solver: full|randomized`, solver seed/oversampling/iterations, fit-window or
  fit-artifact selection, and stored preprocessing metadata. A projection basis rejects
  `standardize=true` and non-`none` rotation.
- Constrain positive epsilon/error/ridge/grid values, ordered ratio/delta/band bounds, fractions in
  `[0,1]`, local time in `[0,24)`, nonnegative gap length, valid covariance dimensions, disjoint
  output paths, and every named dependency. `monthly_taper` defaults are exactly §2.3; segmented
  wavelets require finite `band.max` and `min_segment_days >= 2*band.max`. Add source-level
  `time_padding` duration validation.

**Structured covariance config.** V1 represents `C_obs = D + U U^T`: `D` comes from each obs
entry's `error_variable` after area scaling; optional `common_factor_variables` have dimensions
`(time, common_mode, lat, lon)` in shifted-log effective-error units. Stack rows deterministically
in config sensor order then C-order `(lat,lon)`, concatenate common-mode columns by declared name,
and subset the same rows for each day's valid mask. Apply `C_obs^-1` with the Woodbury identity
using diagonal `D`; never materialize dense `C_obs`. Schema rejects mismatched common-mode names,
grids, units, or missing factors. The generator serializes both factors and realized common errors.

### 4.9 Shared preprocessing utilities

- `util/regrid.py`: conservative/area-weighted coarsening and periodic bilinear interpolation,
  with explicit center/edge convention and longitude normalization.
- `util/local_time.py`: calendar-day local-solar sampling with adjacent-day input and an output
  validity mask; nearest/tie behavior is deterministic.
- `util/logspace.py`: shifted-log forward/inverse plus `delta_to_ratio` with low-AOD and bounds
  policies. Generator oracles do **not** import these production functions.

### 4.10 Daily satellite support (real-data phase only)

- Add `MOD08_D3`/`MYD08_D3` and later VIIRS catalog entries, but also make the reader cadence-aware:
  daily files retain their parsed day rather than being snapped to month start. Return canonical
  `aod_550nm` plus explicit QA/support fields; tests use representative synthetic HDF/NetCDF before
  any real file. Verify C6.1 SDS/QA details against real metadata only after `SYNTHETIC_READY`.

### 4.11 EOF output and solver (`analysis/eof.py`)

- Store reconstruction metadata currently discarded: climatology, time mean, optional std,
  training split/window, solver/seed, source/preprocess hashes, and log epsilon/grid attrs.
- Keep the full SVD as a small-array reference. Add a deterministic truncated/randomized solver
  that never constructs full right-singular factors, reports approximation/subspace error against
  the reference on synthetic cases, operates on float32/chunks at real scale, and is benchmarked
  before a real run. Explained variance denominator/accuracy must remain well-defined.

### 4.12 Artifact persistence (`analysis/artifacts.py` and pipeline manifest)

- Replace the gridded-analysis type check with analysis-declared artifacts. Migrate existing
  `GriddedAnalysis` and its regression tests to return the new policy. Use atomic, yearly/monthly
  time-chunked **NetCDF4 collections** (the existing supported stack), lazy `open_mfdataset`, scientific
  content hashes, and lightweight summaries that do not eagerly scan multi-GB arrays.
- Artifact manifests record role, dimensions/chunks, source/config/code hashes, split/fit identity,
  and checksums. Wall-clock timestamps are metadata but excluded from scientific hashes.

### 4.13 `davinci_monet/analysis/known_truth.py` — evaluation-only `known_truth`

- A generic named-input ARTIFACT analysis compares an estimate artifact with an explicitly configured truth
  source after fitting is frozen. It computes the weighted field/subspace/coefficient and strata
  metrics in §8.1 and emits a small recovery dataset/CSV. It cannot create or replace basis,
  bias/support, coefficient, scaling, or corrected-MMR artifacts.
- Synthetic evaluation config validation allows `oracle/` only when every fit input is a finalized,
  hash-validated artifact and rejects `known_truth` in a fitting config.

---

## 5. Changes to existing code — summary table

| File | Change | Size |
|---|---|---|
| `analysis/base.py`, `config/schema.py` | Named-input/result contract, required semantics, 6 new specs + nested obs/covariance entries, extend `EOFSpec` | scoped; split schema if module limit requires |
| `pipeline/stages/analyses.py` | Single resolver for validation/DAG/execution; required failure and artifact handling | ~100 lines |
| `pipeline/stages/load.py` | Loader-only per-source time padding before analysis-window clipping | small |
| `analysis/eof.py` | Stored fit metadata + deterministic full/reference and randomized/chunked solvers | substantial, keep helpers isolated |
| `analysis/artifacts.py`, `pipeline/stages/manifest.py` | Analysis-declared chunked/atomic artifacts and checksummed manifest entries | scoped |
| `analysis/gridded.py` | Migrate existing product artifacts to the new result policy without regression | small |
| `analysis/wavelet.py` | refactor guts into `cwt_core.py`; behavior-identical outputs | net ~0 |
| `core/protocols.py` | Add non-pairable `ARTIFACT` geometry | small |
| `pipeline/stages/plot.py` | Replace whole-array finite check with chunk-aware/lazy sampling before real-scale plots | small, gated in P7 |
| `datasets/satellite/modis_viirs.py` + catalog | Post-gate D3 cadence, canonical variable, QA contract | deferred real phase |
| `analysis/__init__.py`, docs, CLAUDE.md | register new modules; document new analyses | small |

New production modules: `aod_preprocess.py`, `projection.py`, `wavelet_filter.py`, `scaling.py`,
`mmr_writer.py`, `cwt_core.py`, `known_truth.py`, `util/regrid.py`, `util/local_time.py`, and
`util/logspace.py`.
Synthetic modules/files are specified in §7.1. Each implementation module remains cohesive and
under the project 500-line target; split orchestration from pure math when needed rather than
compressing behavior.

---

## 6. Performance & memory

- **CI cases:** coarse global grids and six synthetic years; full SVD is the deterministic oracle.
  `masked_chain_ci` is fixed at 12x24 native cells, 6x12 analysis cells, hourly 2001-2006 model
  input (plus padding), daily sensors, and float32 fields: model input < 75 MiB, complete FABLE CI
  < 60 seconds and peak RSS < 2 GiB on the development machine. A documented environment-based
  skip is allowed only for the opt-in OSSE, never for CI.
- **Real training matrix:** 30 yr x 365 d by 64.8 k cells is ~2.9 GB at float32 and ~5.7 GB at
  float64 before factors/workspace. The current eager float64 full SVD is not viable. The real gate
  requires bounded-memory randomized/truncated SVD, no full `Vt`, deterministic seed, and measured
  subspace/explained-variance error against the full solver on graduated synthetic matrices.
- **Projection:** assembling daily `H` is `O(T Ncell K^2)`. Benchmark independent-diagonal and
  structured-covariance paths, peak RSS, chunk size, and wall time on a synthetic 1°/multi-year
  stress case before estimating real cost. A 50x50 solve is not the dominant-cost argument.
- **CWT:** benchmark 50 x 11k series including segmentation and significance, not only the raw FFT.
- **Artifacts/scaling:** never eagerly summarize/reload a whole decades-long field or retain both
  mode/native grids. Use time chunks, lazy reopen, and per-file native interpolation in the writer.
- **Writer:** one input file at a time with bounded memory and atomic finalization; parallel writing
  remains deferred until deterministic serial behavior and resume semantics pass.
- **Opt-in synthetic OSSE budget:** 8 years, 36x72 native and 18x36 analysis cells, peak RSS < 8 GiB
  and wall time < 30 minutes. The pre-real 30-year/1° solver-artifact benchmark must stay below
  16 GiB peak RSS; wall time is measured and approved by the user rather than guessed in advance.
- Run in conda env **`davinci`**. Install `pycwt==0.4.0b0` with `--no-deps` as documented and use
  `HDF5_USE_FILE_LOCKING=FALSE` for test/runs where needed.

---

## 7. Synthetic data design and generation

### 7.1 Files, API, and ownership

Add these tracked, text-only components:

- `davinci_monet/tests/synthetic/aerosol_tuning.py` and private `_aerosol_*` helpers: coupled pure
  generator, independent oracles, serializers, policy rendering, and acceptance orchestration.
- `davinci_monet/tests/synthetic/aerosol_calibration.py`, `fable_calibration_identity.py`, and
  `fable_calibration_runner.py`: immutable calibration schema, current-code/design validator, and
  production-pipeline evidence runner.
- `davinci_monet/tests/synthetic/fable_acceptance_gate.py`, `fable_acceptance_record.py`, and
  `fable_artifact_validation.py`: resource/recovery/evidence gates, exclusive seed lock, immutable
  acceptance record, and production manifest/artifact validation bridge.
- Unit tests under `davinci_monet/tests/unit/synthetic/` cover generator/oracle identities,
  prefix isolation, calibration selection/provenance, and acceptance contracts. Integration tests
  in `test_aerosol_tuning_pipeline.py` and `test_aerosol_tuning_projection_pipeline.py` run T1-T6.
- `analyses/aerosol-tuning/scripts/generate_synthetic.py`, `calibrate_synthetic.py`, and
  `run_acceptance.py`: thin CLIs for development generation, calibration, and user-seeded runs.
- `analyses/aerosol-tuning/configs/fable-synthetic.example.yaml`: portable config matching §3.1.
- `analyses/aerosol-tuning/configs/fable-synthetic-eval.example.yaml`: post-fit-only artifact vs
  oracle pairs/stats plus `known_truth`; it is never a fitting input.
- `analyses/aerosol-tuning/.gitignore`: generated inputs, oracle, output, plots, logs, and corrected
  files. No generated binary or machine-specific path is committed.

`generate_aerosol_tuning_bundle(root, spec) -> SyntheticTuningBundle` returns in-memory datasets
and, through a separate writer adapter, creates:

```
root/
  inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4
  inputs/obs/sensor_a.nc
  inputs/obs/sensor_b.nc
  inputs/mmr/MERRA2_SYNTH.inst3_3d_aer_Nv.YYYYMMDD.nc4
  oracle/truth.nc
  scenario.json
```

**Fitting/tuning configs** may reference only `inputs/`. Tests fail if an `oracle/` path or truth
variable appears there. A separate post-fit evaluation config may load oracle fields after all fit
artifacts and parameters are frozen; it cannot write or replace fit artifacts.

### 7.2 Determinism and provenance

- `SyntheticTuningSpec` is a frozen, validated dataclass containing grids, periods, split dates,
  mask/error settings, species, bounds, and a root seed. Reuse existing `Domain`/`TimeConfig`, but
  not the independent random-field scenarios, because this feature requires coupled latent truth.
- Use `np.random.Generator(PCG64)` with stable, named stream IDs derived from `(master_seed,
  stream_name)`, not order-dependent sequential draws. Streams include model residual, correction
  residual, each cloud mask/noise source, common error, outages, and MMR perturbations.
- `scenario.json` records normalized spec, schema/root seed/stream map, serializer and NumPy/
  xarray/netCDF versions, roles, and two hashes: byte SHA-256 for file integrity plus a canonical
  coordinate/array/attribute hash for scientific reproducibility across serializer versions.
  Dataset attrs contain `synthetic=true`, scenario/schema/seed/spec hash. Wall-clock time is not
  part of scientific content.

### 7.3 Coupled latent process (independent of production code)

Create analytic spatial patterns, then weighted-orthonormalize them with an independently written
`cos(lat)` Gram-Schmidt implementation. Model variability and correction variability have
different periods, phases, amplitudes, and random streams:

```
y_m(t,x) = mu(x) + C_m(month,x)
           + sum_j gamma_j p_model,j(t) F_j(x) + eta_m(t,x)

delta_true(t,x) = b_true(month,x)
                  + sum_k a_true,k(t) F_k(x)
                  + delta_perp(t,x)

y_nature(t,x) = ln(A_m_overpass(t,x) + epsilon) + delta_true(t,x)
y_obs,s(t,x)  = y_nature + sensor_bias_s + error_common + error_s
```

`delta_perp` is generated independently and orthogonalized against the retained model subspace.
`exact_micro` sets it to zero; acceptance/stress cases do not. This exposes the representable
error floor and avoids the inverse crime of generating truth by calling the same EOF/projection
functions being tested. Generator code may not import FABLE preprocessing, projection, scaling,
wavelet-filter, interpolation, or writer helpers.

Model files are hourly on a MERRA-like native grid with a known longitude-dependent diurnal term.
Observations are formed from the independent local-time/regrid oracle in linear AOD, converted via
the exact shifted-log equations, corrupted in log space, and finally masked/QA-screened. Every
daily product is anchored at **12:00 UTC** for interpolation; local overpass selection remains
longitude dependent but maps to that labeled calendar day.

### 7.4 Truth and MMR schemas

`oracle/truth.nc` uses distinct dimensions for distinct grids:

| Variable | Dimensions | Meaning |
|---|---|---|
| `pattern_true` | `(truth_mode, mode_lat, mode_lon)` | prescribed weighted-orthogonal model patterns |
| `model_pc_true`, `correction_pc_true` | `(time, truth_mode)` | independent model/correction coefficients |
| `clim_bias_raw_true`, `spatial_support_true` | `(month, mode_lat, mode_lon)` | pre-taper bias and frozen support policy |
| `delta_in_span_true`, `delta_perp_true`, `delta_requested_true` | `(time, mode_lat, mode_lon)` | recoverable, irreducible, and requested correction |
| `delta_supported_true`, `delta_applied_true` | `(time, mode_lat, mode_lon)` | post-support and post-floor/bounds corrections |
| `delta_best_representable_true` | `(time, mode_lat, mode_lon)` | legacy supported/policy-limited **unfiltered** in-span comparator; not an optimal ceiling |
| `delta_filter_target_true` | `(time, mode_lat, mode_lon)` | independent spatial+temporal passband/segment/policy target |
| `model_aod_overpass_true` | `(time, mode_lat, mode_lon)` | independent local-time/regrid oracle |
| `r_requested_true`, `r_applied_true`, `clip_mask_true` | `(time, mode_lat, mode_lon)` | pre-policy ratio, applied ratio, clip reason |
| `aod_target_requested_true`, `aod_target_applied_true` | `(time, mode_lat, mode_lon)` | noise-free nature before/after configured policy |
| `aod_filter_target_true` | `(time, mode_lat, mode_lon)` | AOD implied by `delta_filter_target_true` |
| `r_native_true` | `(time, native_lat, native_lon)` | independent support-aware periodic interpolation |
| `r_3hour_true` | `(mmr_time, native_lat, native_lon)` | independent log-time interpolation |
| `valid_mask`, `qa_flag`, `mask_reason` | `(sensor, time, mode_lat, mode_lon)` | support and decomposed mask causes |
| `obs_error_log`, `reported_sigma_log` | sensor/time/grid | realized and reported errors |
| `innovation_noise_true` | `(time, mode_lat, mode_lon)` | precision-combined noise field for null-energy scoring |
| `obs_holdout_aod`, `obs_holdout_error_log` | `(time, mode_lat, mode_lon)` | independent unassimilated replicate |
| `kappa`, `layer_weight`, `baseline_optical_aod`, `scaled_optical_aod` | species/level/time/grid | serialized optical operator and closure oracle |
| `split` | `(time)` | immutable basis/bias/calibration/development-test schedule |

Generate at least two daily native MMR files with eight 3-hourly samples, four CESM-style pressure
levels (pressure increases with index; surface last), all 15 default aerosols, SO2/DMS/MSA, RH,
pressure thickness, and an unrelated meteorological field. Species have distinct normalized
fractions and vertical profiles. An independent optical oracle
`AOD = sum_i,z kappa_i(RH,z) q_i dp/g` normalizes model AOD and proves `AOD_out = r*AOD_in` under
the fixed synthetic operator. Files exercise float32, `_FillValue`, NaNs, attrs, zlib compression,
chunking, and dimension order. The holdout observation and optical fields use their own named RNG
streams and never appear in fitting inputs.

### 7.5 Masks, errors, and scenario ladder

Store each mask component and form `valid_s = footprint & seasonal_visibility & cloud &
day_available & qa_pass`. Include hemispheric winter gaps, correlated synthetic clouds, whole
absent days, present-but-all-invalid days, a permanent unsupported region, a wholly unobservable
mode, complementary sensors, and overlap. Invalid raw AOD remains finite and extreme with `QA=0`
so a skipped QA filter is detectable; valid data use `QA=3`.

| Scenario | Purpose | Characteristics |
|---|---|---|
| `exact_micro` | algebra/unit oracle | full mask, no noise/clipping/off-basis term; ridge-zero and analytic-ridge variants |
| `masked_chain_ci` | full preprocess-to-scaling CI | six years with the §3.1 schedule, 12x24 native/6x12 analysis grids, separated modes, seasonal/cloud masks, high SNR |
| `multi_sensor_ci` | QA/precision/covariance | complementary footprints, controlled overlap, distinct known errors |
| `writer_ci` | file and optical closure | reuse `masked_chain_ci` analysis inputs; write two selected MMR days with full species/gas/static/fill cases |
| `null_ci` | false-positive/gap control | zero true bias and anomaly, noise, short and long gaps |
| `calibration_null` | frozen-policy null calibration | six years and the exact 4-180-day band/360-day segment policy; zero true bias, anomaly, and off-basis correction |
| `low_aod_ci` | shifted-log closure | zero/near-zero model AOD, both ratio clips, support-zero cells |
| `synthetic_osse` | opt-in acceptance | 8 years, 36x72 native/18x36 analysis grids, off-basis truth, drift/correlated errors/MNAR stress |

Canonical recovery cases use independent Gaussian log errors with known diagonal `C_obs`. Stress cases
add common/correlated error, heteroscedasticity, sensor bias, missing-not-at-random cloud masks,
and basis drift. Stress cases diagnose assumption failure; they are not mislabeled exact recovery.
The six-year `masked_chain_ci`/`writer_ci` recovery tones are 20, 60, and 120 days, safely inside
the 4-180-day passband. A separate pure-sinusoid transfer test locks rejection, cutoff-transition,
and interior-band response; a tone centered exactly on a Morlet cutoff is not used as an amplitude
recovery oracle.

### 7.6 Leakage controls and comparison oracles

- Basis and climatological support/bias fit only their explicit windows. Hyperparameters (`K`,
  covariance simplification, band, ridge, resolution, gap length) tune only on 2005 calibration;
  2006 is a repeatable development test. After config/thresholds freeze, the user or gate runner
  supplies three acceptance seeds not used or hard-coded during development; they are recorded and
  run once for `SYNTHETIC_READY`. For the 2026-07-10 acceptance, the first immutable execution
  stopped before generation, and the same locked seed order was retried in a new root only after a
  full-size development-seed pre-flight; the retry was the only evaluative execution. No v1
  parameter, policy, or threshold may be changed from these now-exposed results.
- Calibration uses fixed non-acceptance seeds `20260710` (`writer_ci`) and `20260711`
  (`calibration_null`) and the immutable `calibration` split. The predeclared candidates are
  `fable-v1-diagonal`, `fable-v1-significant`, and `fable-v1-all-band`; each candidate runs both
  recovery and the exact frozen-policy null, rather than accepting hand-entered scalar metrics.
  Reproduce the record in a new work directory with
  `python analyses/aerosol-tuning/scripts/calibrate_synthetic.py WORK_ROOT RECORD_PATH`.
- Candidate selection hard-rejects recovery or null failures, then ranks eligible policies by
  NRMSE, AOD RMSE ratio, distance of slope from one, and a deterministic simplicity tie-break. The
  canonical calibration record is written atomically, refuses overwrite, carries a self-verifying
  SHA-256, and is revalidated against the fitting template before acceptance rendering.
- The record binds each candidate to scenario, rendered config, normalized run-manifest,
  scientific recovery/null report, production code, generator/oracle/calibration code, both
  fitting/evaluation YAML templates, `environment.yml`, and `pyproject.toml` SHA-256 identities.
  Generator masks/errors/noise are prefix-invariant: values in calibration and earlier splits
  cannot depend on held-out suffix values or suffix RNG draws.
- Daily test observations may enter projection because this is retrospective assimilation; their
  latent truth and scores never enter fitting/tuning. Score against `aod_target_applied_true` and
  an unused independent observation replicate, not the noisy observations assimilated.
- Compare learned bases using weighted subspace angles/projector error. For separated modes, use
  weighted Hungarian sign/permutation matching before coefficient scores; never assume EOF labels.
- Primary reconstruction comparison uses `delta_filter_target_true`; also report full-policy
  `delta_applied_true`, pre-policy `delta_in_span_true`, `delta_perp`, and the legacy distance from
  `delta_best_representable_true`. That last quantity compares against an unfiltered target and may
  not be interpreted as a representability floor. The
  independent temporal oracle is assembled analytically from known in-band/out-of-band components,
  configured mean/trend, segment taper, support, floor, and clipping; it does not call pycwt or
  production filters. Exclude CWT COI/segment edges from primary scores but report excluded fraction
  and full-domain results so exclusion cannot hide failure.

---

## 8. Validation & testing design

### 8.1 Synthetic OSSE (centerpiece; opt-in, no real data)

Run the fitting config through `PipelineRunner.run_from_config()`, freeze its artifacts, then run a
separate evaluation-only pipeline that loads scaling output and oracle AOD as ordinary raw gridded
sources for standard pairs/stats/plots. A read-only `known_truth` analysis in that evaluation
pipeline produces subspace/coefficient metrics; it has no artifact-write access to fitting roles.

Primary hard gates score `delta_log_applied` against `delta_filter_target_true` on development/acceptance
times, supported cells, and non-COI valid segments. Each day is equally weighted and cells use
normalized `cos(lat)` weights. Required: weighted correlation >= 0.90, origin-constrained slope
0.8-1.2, and NRMSE <= 0.35 where NRMSE divides by weighted RMS oracle correction. Separately,
weighted AOD RMSE against `aod_filter_target_true` must be <= 70% of uncorrected model RMSE; RMSE
against the full `aod_target_applied_true` must also improve over the uncorrected model.
Report full-domain and support/resolution/season/latitude strata, excluded fraction, coefficient
metrics, and the explicitly labeled estimate-to-unfiltered-in-span diagnostic; raw AOD correlation
is diagnostic only. A true learned-span oracle must solve the matched filtered target by full-grid
weighted least squares as specified in V2.3.

For `calibration_null`, define false-positive energy as
`sum(w*delta_log_applied^2) / sum(w*innovation_noise_true^2)` over the same non-COI test domain; it
must be <= 0.10. The fraction of significant coefficients among valid, non-COI coefficients inside
the configured band must also be <= 0.10. `null_ci` retains the shorter 4-16-day CI gap/identity
oracle, but it is not evidence for the frozen 4-180-day production policy.
Exact policy/IO cases use numerical closure tolerances rather than statistical targets.

### 8.2 Pipeline integration tests (CI; all enter through `PipelineRunner.run_from_config()`)

- **T1 `test_aerosol_tuning_known_mode_chain`**: generate `masked_chain_ci`; run preprocess → EOF
  → projection → filter → scaling. Assert required pseudo-sources/artifacts, no analysis errors,
  subspace recovery, shifted-log closure, support-zero identity, and latent-target improvement.
- **T2 `test_aerosol_tuning_multi_sensor_projection`**: produce A-only, B-only, and blended analyses
  from one basis. Assert exact QA counts, analytic precision weighting in controlled overlap,
  absent-sensor zero contribution, and lower analytic posterior variance. Empirical RMSE advantage
  is assessed only as a locked multi-seed aggregate, not required from one random realization.
- **T3 `test_aerosol_tuning_writer_pipeline`**: run the complete chain through `mmr_writer`; inspect
  both files, optical closure, metadata/fill preservation, atomic outputs, checksums, and run manifest.
- **T4 `test_aerosol_tuning_null_and_gap_behavior`**: projection is zero on all-missing days and a
  wholly unobservable mode; only bounded gaps bridge; long gaps/outside coverage remain identity;
  null retained anomaly energy/false-positive rate stays below the frozen 10% ceiling.
- **T5 `test_aerosol_tuning_required_failure_is_fatal`**: inject a required descendant failure and
  assert failed stage/run and dependency-blocked descendants. Hash-validated files finalized before
  the failure may remain resumable, but the run/manifest is explicitly incomplete and never success.
- **T6 `test_aerosol_tuning_saved_fit_fresh_runner`**: write basis/bias/support artifacts, start a
  fresh runner using only those fits but the **same complete 2001-2006 application axis** (so CWT
  detrending/AR(1)/COI/segment context is identical), prove no refit occurred, compare 2006 output
  with the same-run result, and reject source/config hash mismatch.

Total synthetic CI target: under 60 seconds. Large `synthetic_osse` is developer-run, not CI.

### 8.3 Unit and generator-contract tests

- Generator: deterministic named streams, positivity, mask composition, QA rejection, split
  immutability, truth hidden from configs, shifted-log identities, and independent optical closure.
- Projection: exact full-mask coefficients; analytic ridge shrinkage; masked/correlated `C_obs` cases;
  posterior/resolution bounds and rank; support preservation after smoothing; common-bias precision.
- Wavelet: band pass/reject, mean/trend policy, bounded gaps/segments, masks/COI, reconstruction
  error, and null ensemble. If the false-positive gate fails, implement calibrated FDR/Monte Carlo.
- Scaling: exact shifted-log inverse including low AOD, asymmetric bounds, support identity,
  pre-exponential AOD-dependent bounds (including extreme finite anomalies), periodic seam, and
  log-time interpolation. Float64 pure math uses tight `rtol <= 1e-10`; writer
  float32 closure uses `rtol <= 5e-6` unless the independent oracle proves a stricter bound.
- Writer: prescribed-ratio pure unit test (separate from T3), full species coverage, unchanged gas/
  fill/static fields, dtype/dims/attrs/compression/chunks, collision/overwrite/resume, atomic failure.
- Pipeline/schema: named DAG order, raw+derived refs, unknown/cycle errors, numeric bounds, required
  failures, artifact laziness/manifest, and exact renderer output contracts.

### 8.4 Deferred real evaluation

Only after §8.5: enable daily readers/catalogs and generate frozen real fit artifacts. A second
standard pipeline evaluates saved `aod_target` and MERRA-2. Aqua observations used in projection,
including a later time window, yield **assimilation diagnostics only**. Headline external validation
uses a predeclared unassimilated source (prefer AERONET; optionally a completely withheld satellite)
and reports N/MB/RMSE/R/NMB/NME/IOA globally and by season/support.

### 8.5 `SYNTHETIC_READY` gate

The gate requires: repository tests/mypy/Black/isort green; all T1-T6 and pure oracles green;
three user-supplied, then locked, `synthetic_osse` seeds meet frozen aggregate/per-seed targets;
null and off-basis floors
are honestly reported; peak memory/runtime benchmarks satisfy documented limits; artifacts/manifests
reproduce from hashes; and the user reviews the recovery report. No real-data phase begins earlier.

Acceptance gives each seed equal aggregate weight, reports 95% Student-t intervals, and requires
every per-seed gate plus the aggregate mean gate. Each seed is limited to 30 minutes and 8 GiB peak
RSS; primary exclusion must be <= 0.80. The record retains full strata/decomposition/holdout
diagnostics and validates fitting/evaluation manifests plus every recovery-artifact checksum before
completion. Seeds are locked with exclusive creation before generation and are never inferred.

**Historical v1 acceptance disposition (2026-07-10): rejected.** The three user-supplied seeds
completed the synthetic OSSE fitting and evaluation pipelines under the frozen `fable-v1-all-band`
policy. All evidence, resource, exclusion, correlation, slope, and AOD-improvement gates passed,
but each seed
and the equal-seed aggregate failed `field_nrmse <= 0.35`; the aggregate was 0.5067 with 95%
Student-t interval [0.4860, 0.5275]. The original record, immutable seed locks, and reporting
supplement identified in the checkpoint above are retained as the audit trail. `SYNTHETIC_READY`
cannot be approved for v1, and no P9 real-data work is authorized. Any future method revision must
start a new versioned development/calibration cycle and use new held-out acceptance seeds rather
than tuning to or reevaluating on these seeds.

**V2 acceptance disposition (2026-07-11): `passed_pending_user_review`.** The frozen
`v2-joint-seasonal-offset` policy passed every per-seed and equal-seed scientific, evidence,
exclusion, and resource gate on the one-time ordered acceptance tuple `(1969, 2010, 2013)`. The
aggregate NRMSE was 0.2178 with 95% Student-t interval [0.2114, 0.2243]; full metrics, identities,
locks, and review-only plot provenance are recorded in the v2 checkpoint above. This passing
program result does not itself set `SYNTHETIC_READY`: explicit user review remains required, so
`SYNTHETIC_READY` remains unset and P9 remains blocked.

---

## 9. Implementation phases

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Restore/disposition baseline gates in `davinci`; install pinned pycwt; approve this test design. Build named-input/result/required-failure/artifact foundations and migrate existing `gridded_analysis`. | existing behavior/artifact regressions green; T5 foundation green |
| **P1** | Coupled generator, independent truth/MMR oracle, serializers, provenance, scenario unit tests, synthetic example skeleton. | generator identities and no-truth-leak checks green |
| **P2** | Regrid/local-time/log utilities, source time padding, and `aod_preprocess`; no real reader changes. | exact/date-edge/dateline/low-AOD unit tests + pipeline preprocess test |
| **P3** | EOF metadata + full reference and deterministic randomized solver; graduated synthetic benchmark. | existing EOF tests + subspace/variance/memory gates |
| **P4** | Shared `cwt_core.py` refactor with current wavelet output regression. | existing wavelet tests behavior-identical |
| **P5** | `eof_projection`, saved bias/support fit artifacts, covariance/posterior diagnostics. | pure solver tests + T2 |
| **P6** | Bounded-gap `wavelet_filter` with exact scalogram contract and null calibration. | wavelet unit/null tests + renderer contract |
| **P7** | Exact `aod_scaling`, chunked/lazy artifacts, and chunk-aware plot finite checks. | T1 + low-AOD/support/clip oracles |
| **P8** | Atomic `mmr_writer`, `known_truth` evaluation, full-species optical closure, manifests; run user-supplied acceptance seeds. | T3-T6 + user-approved `SYNTHETIC_READY` report |
| **P9** | **Real-data enablement begins:** MERRA-2 audit, MODIS D3 cadence/catalog/canonical variable/QA readers, real configs. | reader tests with synthetic representative files, then controlled real-file smoke |
| **P10** | Frozen-fit MERRA-2/Aqua run; assimilation diagnostics plus external unassimilated evaluation; tune nothing on evaluation data. | before/after and sensitivity report reviewed by user |
| **P11** | GEOS-IT grid/collection/species audit, reader work, then configs; Terra/VIIRS expansion separately. | same synthetic regression + product-specific gates |

P0-P8 remain strictly synthetic. Each phase is TDD, but repository rules require presenting that
phase's concrete test entry points/data flow and receiving approval before writing tests. No commits
or pushes occur without explicit user approval.

Implementation approval was received on 2026-07-10. P0-P8 software and synthetic validation are
complete, and the required P8 acceptance execution has concluded. The frozen v1 recovery gate
failed, so v1 `SYNTHETIC_READY` remains rejected. The separately versioned v2 recovery cycle passed
with status `passed_pending_user_review`; `SYNTHETIC_READY` remains unset until explicit user review.
This checkpoint explicitly blocks P9.

---

## 10. Risks & open questions

1. **MODIS DT/DB regional biases** leak into `b_hat` (it will faithfully "correct" toward biased
   retrievals, e.g. DB over bright surfaces). Mitigations later: per-region σ inflation,
   AERONET cross-check of `b_hat` maps before trusting them.
2. **Basis stationarity over decades** — major eruptions (Pinatubo) distort covariance. Option:
   `exclude_periods:` on the EOF training window; decide after inspecting the scree/patterns.
3. **Correction-subspace mismatch** remains a structural risk. Synthetic `delta_perp` and
   basis-drift cases quantify it, and real claims must not imply EOF-span completeness. The v1
   `best_representable_nrmse=0.5078` value was mislabeled estimate-to-unfiltered-in-span error, not
   an optimum or irreducible floor; it cannot explain the 0.5067 primary result. The v2 stagewise
   learned-span oracle measures this risk against the matched filtered target. Frozen v1 evidence
   remains historical diagnostic context and cannot select v2 parameters.
4. **Wavelet significance after shrinkage/gap fill** remains heuristic, but the v2 full-size null
   ensemble passed every frozen per-seed and equal-seed false-positive gate. Retain the observed
   null rates as the current bound; FDR/Monte Carlo calibration is mandatory if a future frozen
   null gate fails.
5. **icwt fidelity** (~few % for Morlet) is measured per mode with a full-scale round trip and
   included in recovery error; the intentionally truncated filtering grid is not mislabeled as a
   full-CWT reconstruction diagnostic.
6. **Support gating trades covariance extrapolation for conservative identity.** The default honors
   the stated no-evidence/no-correction policy; an ungated research sensitivity is reported, not
   silently substituted.
7. **Observation covariance misspecification** affects shrinkage and sensor weighting. Synthetic
   correlated-error cases bound the consequence; real `C_obs` remains an effective covariance model.
8. **Artifact scale and restart integrity** require benchmarked chunks, hashes, atomic writes, and
   bounded summaries before real decades are attempted.
9. **Optical closure is conditional**, not chemical closure. The real RT operator/species/RH audit
   must confirm homogeneity and complete aerosol coverage before corrected MMRs are trusted.
10. **Daily → 3-hourly application** assumes MERRA-2's diurnal AOD shape — accepted by design
   (we correct daily-and-slower scales only).
11. **Overpass sampling approximation** — daily `r` stamped by calendar day though estimated at 13:30
   LST; consistent between training and innovation, so it cancels to first order.
12. **Exact real product details remain deliberately unverified until P9:** D3 SDS/QA/cadence,
   C6.1 choice, MMR tracer list/encoding, and GEOS-IT grids/collections.
13. **Frozen v2 reusable integrity validators need hardening.** They do not generically enforce
    locked-root path equality, corrected-MMR validation checks three provenance identities only for
    presence, prerequisite deep validation is process-cached, and preregistration stores test-log
    hashes without paths. The completed cycle's two post-hoc audits passed all 16 root/config/link/
    artifact checks, all 32 recomputed MMR provenance checks, and all five exact test-log bindings,
    so its recorded evidence is intact. Before any future synthetic recovery cycle, correct these
    generic validators and add adversarial cross-root, cross-policy, external-config, MMR-identity,
    cache-mutation, and test-log substitution tests; do not reuse the consumed v2 holdouts.
14. **Renderer compatibility** is an explicit data/artist contract in P6 rather than a late PNG
   smoke-test discovery.

---

## 11. Decisions & rationale

1. **Innovation projection, not raw-obs projection** — zero-observability limit = "keep the
   analysis", which is the only safe default under seasonally vanishing coverage.
2. **Ridge is an explicit prior model.** `lambda=1` is the baseline for unit-PC covariance only
   when paired with configured `C_obs`; synthetic calibration tests this assumption.
3. **Shifted log space with exact inverse** — preserve statistical benefits without pretending
   `exp(delta)` is the physical ratio at low AOD.
4. **Systematic + anomaly correction in transformed space**, both multiplied by stored monthly
   support; physical `r` is derived once after their sum.
5. **Synthetic truth is independent and coupled** — model/nature/obs/MMR share a known physical
   story, while oracle code never calls production implementations.
6. **Frozen train/calibration/test evidence** precedes all real data; real masks are not synthetic.
7. **1° real mode space** remains the target, but only a benchmarked truncated solver can reach it.
8. **Chained named-input analyses** preserve DAVINCI architecture; required/fatal and artifact
   results make a correction chain safer than today's single-source soft-failure behavior.
9. **Multi-sensor from day one** — obs is a list and covariance-aware contributions are additive;
   no merged observation product is required.
10. **Uniform per-column aerosol scaling** is an optical diagnostic under a fixed homogeneous
    operator; speciation/vertical shape are deliberately untouched and gases are excluded.
11. **Wavelet filtering bridges bounded gaps only.** Long gaps/outside coverage revert anomaly
    correction to zero; trend/mean preservation is explicit and null behavior is calibrated.
12. **MERRA-2/Aqua are first only after `SYNTHETIC_READY`**; GEOS-IT and Terra/VIIRS require their
    own reader/grid/QA audits rather than being assumed drop-in.

## 12. References

- Kaplan, A., et al. (1998): Analyses of global sea surface temperature 1856–1991 — reduced-space
  optimal interpolation of gappy obs onto complete-field EOFs (the method template).
- Torrence, C. & Compo, G. P. (1998): A practical guide to wavelet analysis — CWT, significance,
  reconstruction (eq. 11).
- North, G. R., et al. (1982): Sampling errors in the estimation of empirical orthogonal functions.
- Beckers, J.-M. & Rixen, M. (2003): DINEOF — EOF-based infilling (context; our fixed-basis ridge
  solve supersedes iteration).
- Existing in-repo spec: `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md`
  (derived-analysis layer this plan builds on).
