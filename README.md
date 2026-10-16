# CSpec — label-free cortical parcellation

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue)](CITATION.cff)

> Tag this release with the current date or version at publish time
> (e.g. `vYYYY-MM-DD-cgsd-brain-github`); `CITATION.cff` carries the
> canonical `version:` + `date-released:` metadata.

Reference implementation of **CSpec** (Curvature–Spectral): a label-free,
single-CPU spectral method that maps a FreeSurfer T1 MRI to a
Schaefer-2018-comparable per-vertex cortical parcellation without
functional data and without GPU compute.

> Wang, F. (2026). "CSpec: a label-free, single-CPU spectral method for
> cortical parcellation with the Forman–Ricci curvature gate."
> *Journal of Neuroscience Methods.*

CSpec is the canonical algorithm of the `cgsd_brain` line of work. The
same code base produced the Phase A–D results (NMI 0.7623 ± 0.0018 on
Schaefer-200, 10 IXI subjects × 3 seeds × 2 hemi) and the 5-atlas
sensitivity sweep (DK68 0.667, Destrieux 0.711, Schaefer-100 0.722,
Schaefer-200 0.762, Schaefer-400 0.791).

---

## What is in this repository

```
github/
├── cgsd_brain/             # CSpec implementation (10 Python modules)
│   ├── cluster.py          #   curvature-aware spectral clustering (G3/CSpec)
│   ├── curvature.py        #   Forman–Ricci curvature on the mesh
│   ├── sulcal.py           #   sulcal-depth-weighted k-NN affinity
│   ├── atlas_eval.py       #   atlas loaders + NMI / ARI / Dice
│   ├── data.py             #   subject loaders (synthetic / fsaverage5 / IXI)
│   ├── hemisphere.py       #   per-hemisphere surface container
│   ├── fs_extract.py       #   FreeSurfer file readers
│   ├── sheaf.py            #   SheafDiffusion (legacy cgsd_brain MVP path)
│   ├── losses.py           #   loss terms for the legacy encoder path
│   ├── train.py            #   training loop (legacy path only)
│   └── run_all.py          #   end-to-end entry point
├── configs/
│   └── default.yaml        # CSpec hyper-parameters (k=10, α=1.0, β=1.0)
├── scripts/
│   └── synthesize_surface.py  # synthetic Fsm-style surface for smoke test
├── reproduce.sh            # one-line smoke test (≈ 30 s on a single CPU)
├── requirements.txt        # Python dependencies
├── CITATION.cff            # GitHub "Cite this repository" metadata
├── DATA_AVAILABILITY.md    # where the IXI / FreeSurfer outputs live
├── RELEASE_NOTES.md        # what is and is not in this release
└── LICENSE                 # MIT
```

This is the **minimal subset** referenced in the paper's Data
Availability statement (§"Code availability"):

> *"The release contains a minimal subset (clustering, curvature,
> sulcal affinity, atlas evaluation) sufficient to reproduce the §2.6
> ablations; the complete pipeline can be re-run with the
> reproducibility commands documented in the release."*

The 18 paper-experiment driver scripts (`aggregate_*`, `batch_*`,
`download_ixi_*`, `launch_*`, `recon_all_runner.sh`,
`fastsurfer_runner.sh`, etc.) are **not** bundled here — they live in
the development tree at `experiments/cgsd_brain_parcellation_001/scripts/`
and can be re-run by following the commands in
[§ Reproducing the paper's full pipeline](#reproducing-the-papers-full-pipeline).

---

## Quick start

```bash
# 1. (Recommended) Create a virtualenv and install the dependencies.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the smoke test (≈ 30 s on a single CPU).
bash reproduce.sh
```

The smoke test generates a synthetic Fsm-style cortical surface
(`scripts/synthesize_surface.py`), runs the full CSpec pipeline on it
(`python -m cgsd_brain.run_all --smoke`), and prints a bit-exact NMI
number against a planted ground-truth atlas. The expected output is
`Smoke test PASSED ✓`.

A working Python ≥ 3.10 is required. On this development machine
(`/home/woody/...`) the pinned interpreter is
`/home/woody/vp/vp3.12/bin/python`; on any other machine, the system
`python3` is fine — `reproduce.sh` reads the `PYTHON` env var if you
want to override.

---

## Using CSpec on your own FreeSurfer subject

Once FreeSurfer ≥ 8.0 is installed and a subject is recon-all'd under
`$SUBJECTS_DIR/<subject>/`:

```bash
# 1. Remap the subject to the fsaverage5 surface (one-time, ≈ 5 min).
mri_surf2surf --srcsubject <subject> --trgsubject fsaverage5 \
             --sval-annot <subject>/label/lh.aparc.annot \
             --tval  <subject>/label/lh.fsaverage5.aparc.annot \
             --hemi lh
# (repeat for rh and any custom parcellations)

# 2. Download the Schaefer atlas (≈ 5 min, one-time).
SUBJECTS_DIR=$SUBJECTS_DIR python scripts/download_atlases.py

# 3. Run CSpec.
PYTHONPATH=. python -m cgsd_brain.run_all \
    --subject-id <subject> --atlas schaefer-200 --both-hemi --seed 0
```

The CLI exposes:

| flag | meaning |
|------|---------|
| `--smoke` | synthetic surface + 1 hemi + 1 atlas (CI / verification) |
| `--subject-id <id>` | `synthetic_smoke` / `fsaverage5` / `IXI###` |
| `--atlas <name>` | `desikan-killiany` / `destrieux` / `schaefer-{100,200,400}` |
| `--hemi {lh,rh}` | single hemisphere |
| `--both-hemi` | both hemispheres (averaged metrics) |
| `--seed <int>` | K-Means RNG seed |
| `--config <path>` | alternative YAML (default `configs/default.yaml`) |

Expected runtime on a single CPU thread: **≈ 35 s per hemisphere** for
Schaefer-200 on the fsaverage5 mesh (10 242 vertices / hemi).

---

## Hyper-parameter ablation (§2.6 of the paper)

The `configs/default.yaml` ships with `cspec_only: true`, `k: 10`,
`α: 1.0`, `β: 1.0`. The §2.6 ablations vary one of these at a time:

```bash
# k-NN degree k
for k in 5 10 20 40; do
    PYTHONPATH=. python -m cgsd_brain.run_all \
        --config <(yq ".k = $k" configs/default.yaml) \
        --smoke
done

# Curvature-gate scale α
for alpha in 0.0 0.5 1.0 2.0; do
    PYTHONPATH=. python -m cgsd_brain.run_all \
        --config <(yq ".alpha = $alpha" configs/default.yaml) \
        --smoke
done

# Sulcal affinity β
for beta in 0.0 0.5 1.0 2.0; do
    PYTHONPATH=. python -m cgsd_brain.run_all \
        --config <(yq ".sulcal_beta = $beta" configs/default.yaml) \
        --smoke
done
```

(`yq` from <https://kislyuk.github.io/yq/> is the simplest YAML editor
on the command line; alternatively, edit `configs/default.yaml` by hand.)

---

## Reproducing the paper's full pipeline

The headline numbers in the paper were produced by Phase A → B → C → D
runs on the IXI dataset (10 subjects, two hemispheres each, 3 seeds
each, 5 atlases for the sensitivity sweep). Those driver scripts are
**not** bundled here, but they live in the development tree:

| Step | Where it lives |
|------|----------------|
| IXI T1 download       | `experiments/cgsd_brain_parcellation_001/scripts/download_ixi_subject.py` |
| recon-all (one subj)  | `experiments/cgsd_brain_parcellation_001/scripts/recon_all_runner.sh` |
| Remap to fsaverage5   | `experiments/cgsd_brain_parcellation_001/scripts/remap_to_fs5.sh` |
| Atlas download        | `experiments/cgsd_brain_parcellation_001/scripts/download_atlases.py` |
| Run CSpec per-subject | `experiments/cgsd_brain_parcellation_001/scripts/batch_run_cgsd_brain.py` |
| Atlas-sensitivity sweep| `experiments/cgsd_brain_parcellation_001/scripts/batch_atlas_sensitivity.py` |
| Aggregate results     | `experiments/cgsd_brain_parcellation_001/scripts/aggregate_phaseD.py` |
| Compute ASI           | `experiments/cgsd_brain_parcellation_001/scripts/compute_asi.py` |

End-to-end instructions and the per-step runtime budget are in
`experiments/cgsd_brain_parcellation_001/docs/ATLAS_SENSITIVITY_5ATLAS.md`.

---

## Algorithm in one paragraph

CSpec builds a per-vertex feature vector from `(coords, sulc, curv,
thickness)` on the fsaverage5 surface, computes a curvature-reweighted
k-NN affinity, takes the Ng–Jordan–Weiss spectral embedding of the
symmetric normalized Laplacian of that affinity, and runs K-Means in
the eigenspace. The curvature logit on each edge,
`σ(α · κ_ij)`, suppresses inter-community bridges and amplifies
intra-community ties, biasing the spectral cut toward structurally
tight clusters. There is **no encoder**, no GPU, and no
hyper-parameter that touches a downstream label — the only knobs are
`k` (k-NN degree), `α` (curvature logit scale), and `β` (sulcal
affinity weight). Details in `cgsd_brain/cluster.py` (≈ 130 LoC) and
`cgsd_brain/curvature.py` (≈ 70 LoC).

---

## What is NOT in this repository

Per the project's large-files policy (2026-06-25), the following are
not in git and not in this release:

* **FreeSurfer recon-all outputs** (≈ 4.5 GB per subject × 10
  subjects). Re-run `recon-all` from the raw T1.
* **FreeSurfer binary installs** (proprietary). Install separately
  from <https://surfer.nmr.mgh.harvard.edu/>.
* **IXI raw T1 data** (≈ 16 GB). Download from
  <http://brain-development.org/ixi-dataset/> under the IXI data-use
  agreement.
* **IXI-derived per-subject outputs** (≈ 10 GB). Re-derive with
  `recon-all` + `mri_surf2surf` (commands above).

For provenance see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

---

## Citation

If you use this software, please cite both the paper and the version
tag of the code you used. Citation metadata is bundled in
[`CITATION.cff`](CITATION.cff) and below.

```bibtex
@article{wang2026cspec,
  title   = {CSpec: a label-free, single-CPU spectral method for
             cortical parcellation with the Forman--Ricci curvature gate},
  author  = {Wang, Feifan},
  journal = {Journal of Neuroscience Methods},
  year    = {2026},
  note    = {Software: \url{https://github.com/woodywff/cgsd_brain},
             tag v2026-10-16}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE). The accompanying paper is CC-BY-4.0.