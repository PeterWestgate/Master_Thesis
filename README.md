# Characterizing Visual Object Categorization through Spatial Arrangement and its Neural Correlates

Analysis code for the spatial arrangement behavioral experiment with fMRI-based Representational Similarity Analysis (RSA).

**KU Leuven — Faculty of Psychology and Pedagogical Sciences, 2024–2026**

---

## Overview

This study investigates how humans represent visual object categories, and where that representational structure is encoded in the brain. It combines:

- **Behavioral experiment**: A spatial arrangement task (inverse MDS) in which ~22 participants arranged 80 images of 8 object categories by perceived similarity, yielding per-participant behavioral RDMs.
- **fMRI experiment**: 20 participants viewed 400 images (80 exemplars × 5 image conditions: control, clutter, deletion, occlusion, phase scrambling), yielding neural RDMs per cortical region of interest (ROI).
- **RSA**: Pearson-based Representational Similarity Analysis correlating behavioral, neural, and theoretical model RDMs to map where object similarity structure is represented across the visual hierarchy.

**8 object categories**: person, cat, bird, banana, fire hydrant, tree, bus, building  
**22 ROIs**: Derived from the HCP Multi-Modal Parcellation atlas (Glasser et al., 2016)  
**3 theoretical models**: Category, Animacy, GIST

---

## Repository Structure

```
├── scripts/                      # Core RSA and analysis scripts
├── src/                          # Shared configuration and utilities
├── glass brain/                  # Glass brain visualization scripts and atlas data
├── Peter visualisation scripts/  # Explanatory visualizations of RSA concepts
├── models/
│   └── theoretical_models/       # Category, Animacy, and GIST model RDMs
└── output/
    ├── averaged/                  # Average behavioral RDM across participants
    ├── reliability/               # Noise ceiling and split-half reliability estimates
    ├── results/                   # Summary statistics (CSV)
    └── figures/                   # Generated figures
```
## Setup

**Python 3.9+** is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

All scripts are run from the repository root and use relative paths. No additional configuration is needed.

---

## Analysis Scripts

### Core pipeline (`scripts/`)

| Script | Description |
|---|---|
| `collect_behavioural_data.py` | Aggregates raw spatial arrangement trial data into per-participant RDMs |
| `average_behavioural_matrices.py` | Fisher z-transforms and averages behavioral RDMs across participants |
| `create_theoretical_rdms.py` | Generates theoretical model RDMs (Category, Animacy, GIST) |
| `reliability_behaviour.py` | Split-half reliability of behavioral RDMs |
| `reliability_rois.py` | Noise ceiling estimation for neural ROIs |
| `rsa_brain_behaviour.py` | RSA between each ROI and behavioral RDMs |
| `rsa_brain_theoretical.py` | RSA between each ROI and the three theoretical models |
| `rsa_bh_theoretical.py` | RSA between behavioral RDMs and theoretical models |
| `rsa_bh_partial_corr.py` | Semi-partial and partial correlations (behavior vs. models) |
| `rsa_neural_partial_corr.py` | Semi-partial and partial correlations (neural vs. models) |
| `rsa_brain_beh_partial_corr.py` | Semi-partial and partial correlations (brain–behavior) |
| `rsa_challenge_v_control.py` | RSA comparing degraded image conditions vs. intact control |
| `create_results_table.py` | Compiles summary results tables |

### Utilities (`src/`)

| File | Contents |
|---|---|
| `__init__.py` | Shared variables: category names, ROI names, image identifiers, stimulus conditions |
| `plot_config.py` | Color scheme and plot defaults |
| `figure_style.py` | APA-compliant axis and tick formatting |
| `utils.py` | Helper functions (e.g., upper triangle extraction) |

### Brain visualizations (`glass brain/`)

Glass brain plots of RSA results across the cortex, using the HCP-MMP1 atlas. The MNI-space atlas parcellation (`sub-01_atlas-HCPMMP1cortices_space-MNI.nii`) and ROI reference table (`HCPMMP1cortices_reference.tsv`) are included.

---

## Key Methods

### Correlation approach

All RSA correlations use **Pearson correlation** (*r*). The analysis pipeline:

1. Correlations are computed **per participant** — not on a single averaged matrix.
2. Per-participant *r* values are **Fisher z-transformed** before averaging.
3. The mean is inverse-transformed back to *r* for reporting, alongside *SD* and *p*-values.

### Semi-partial correlations

To isolate the unique contribution of each theoretical model to behavioral or neural similarity structure:

- **Semi-partial correlation**: Correlate target RDM *Y* with the residuals of model *X₁* after regressing out control model *X₂*. Quantifies unique variance *X₁* explains in *Y*.

### Image conditions (fMRI)

The fMRI experiment used five image conditions to test object recognition under degradation:

| Condition | Description |
|---|---|
| Control | Intact images |
| Clutter | Superimposed background clutter |
| Deletion | Random deletion of image regions |
| Occlusion | Objects partially occluded |
| Phase scrambling | Phase-scrambled images (low-level baseline) |

---

## Data Availability

| Data | Included | Reason |
|---|---|---|
| Theoretical model RDMs | Yes | Not participant-specific |
| Average behavioral RDM | Yes | Aggregate, de-identified |
| Noise ceiling / reliability estimates | Yes | Aggregate |
| Summary results (CSV) | Yes | Aggregate statistics |
| Per-participant behavioral trial data | No | Participant privacy |
| Per-participant neural RDMs | No | Participant privacy |

---

## Citation

If you use this code, please cite:

> Westgate, P. (2026). *Characterizing Visual Object Categorization through Spatial Arrangement and its Neural Correlates* [Master's thesis]. KU Leuven, Faculty of Psychology and Pedagogical Sciences.

---

## Acknowledgements

Conducted at KU Leuven (PPW) under the supervision of Tim Maniquet. ROI definition used the HCP Multi-Modal Parcellation atlas: Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*, *536*, 171–178. https://doi.org/10.1038/nature18933
