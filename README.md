# Chapter 6 Case Studies: Tracer Transport in Plant Vasculature

Reproducibility repository for the case studies in Chapter 6 of the dissertation
*"Advancing Model-based Analysis of Tracer Transport Dynamics in Plants"*
(Hannah Lanzrath, RWTH Aachen University).

Each notebook reproduces one figure or table from the chapter end-to-end: it loads the
experimental PET data, sets up the CADET-Process simulation, fits the MCT model parameters,
and produces the publication figure and a parameter-uncertainty estimate.

---

## Case studies

### Case 1: Influence of dispersion and cross-sectional area (poplar side branch)

Five time-activity curves (TACs) from consecutive ROIs along a 5.7 cm side branch of a poplar
tree are analyzed. The half-maximum distance-time fit shows a curved (non-linear) relationship,
indicating local transport heterogeneity. Two MCT model configurations are compared:

- **Changing dispersion**: cross-sectional areas fixed, dispersion fitted per segment
- **Changing cross-sectional areas**: dispersion fixed to zero, cross-sectional area fitted per
  segment (velocity changes via v = Q/A)

Both fits are of similar quality, illustrating that dispersive and geometric effects can be
confounded when interpreting TAC-based velocity estimates. The biological context (visible
thinning of the side branch) supports the geometric interpretation.

### Case 2a: Local velocity decline along a maize primary root

Four ROIs along the lower segment of a maize primary root show increasing temporal spacing
between TAC peaks, indicating a deceleration of the tracer front. A network of four MCT M01
unit operations with independently fitted cross-sectional areas reveals a progressive velocity
decrease from 3.40 mm/min to 2.38 mm/min, likely caused by lateral branching increasing the
effective transport cross-section toward the root tip.

### Case 2b: Temporal velocity response to leaf shading (phaseolus internode)

A phaseolus plant is shaded for 30 minutes and the resulting change in phloem transport velocity
is tracked via PET. A time-dependent velocity profile (piecewise linear, three transition
points) is fitted to a single MCT M02 unit. Two measurement sessions on the same plant are
analyzed (morning and afternoon), each with four unshaded control measurements (3 h prior and
the day before) for context. Both shaded measurements show a rapid velocity decline after shading
onset and recovery to baseline within ~30 minutes of shade removal.

### Case 3: Tracer distribution in a branched poplar structure

TACs from nine ROIs across three anatomical sections of a poplar branch (upstream, lower branch,
side branch) are fitted simultaneously with a branched MCT network. The model quantifies how the
tracer pulse splits at the branch point and how transport velocity and storage differ between
branches.

---

## Repository layout

```
data/                          raw experimental data (CSV, one file per case study)
utils.py                       shared helper functions used by all notebooks

Case 1: Poplar side branch
  case1_preliminary_hmax_pearson_poplar_side_branch.ipynb
  case1_appendix_m02_vs_m13_test_poplar_side_branch_cadet_process.ipynb
  case1_changing_area_poplar_side_branch_cadet_process.ipynb
  case1_changing_dispersion_poplar_side_branch_cadet_process.ipynb

Case 2a: Maize primary root
  case2a_preliminary_hmax_maize_root.ipynb
  case2a_local_velocity_change_maize_root_cadet_process.ipynb

Case 2b: Phaseolus internode (shading experiment)
  case2b_control_M10_3h_prior_cadet_process.ipynb      unshaded controls
  case2b_control_M10_day_before_cadet_process.ipynb
  case2b_control_M12_3h_prior_cadet_process.ipynb
  case2b_control_M12_day_before_cadet_process.ipynb
  case2b_temporal_velocity_shading_phaseolus_morning_cadet_process.ipynb   shaded
  case2b_temporal_velocity_shading_phaseolus_afternoon_cadet_process.ipynb

Case 3: Poplar branch (branched topology)
  case3_preliminary_hmax_pearson_poplar_branch.ipynb
  case3_branched_structure_poplar_branch_cadet_process.ipynb
```

Generated figures are written to `output/figures/` (gitignored).

---

## Setup

```bash
conda env create -f environment.yaml
conda activate chapter6_cases
```

---

## Running the notebooks

Open any notebook in JupyterLab and run all cells. Each notebook is self-contained: it loads
data, builds the process model, and produces figures. The optimizer is not re-run by default;
`run_optimization()` (called at the bottom of the optimization section) returns the stored
best-fit values without recomputing. To re-run the optimizer from the best-fit starting point:

```python
x_best_fit = run_optimization(optimize=True)
```

To start from a custom point:

```python
x_best_fit = run_optimization(optimize=True, x0=my_starting_point)
```

Available `method` options: `"nm"` (Nelder-Mead), `"tc"` (Trust-Constraint), `"nm+tc"`
(Nelder-Mead then Trust-Constraint, default for most notebooks), `"tc+nm"` (reversed).

---

## Data

All experimental data in `data/` are time-activity curves (TACs) from dynamic PET scans of
plant stems and roots, exported as wide-format CSV files (rows = time points in minutes,
columns = distances along the organ in mm). The raw scanner output was converted to this
format once by `scripts/convert_raw_data_to_csv.py`.

---

## Dependencies

| Package | Role |
|---|---|
| [CADET](https://github.com/modsim/CADET) | chromatography simulator (C++ binary) |
| [CADET-Process](https://github.com/fau-advanced-separations/CADET-Process) | Python API, `OptimizationProblem`, NelderMead/TrustConstr wrappers |
| NumPy / SciPy | numerics, Jacobian-based uncertainty estimates |
| pandas | data loading and tabular output |
| matplotlib | figures |
| seaborn | Pearson correlation heatmaps (preliminary notebooks) |
