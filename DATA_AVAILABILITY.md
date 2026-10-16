# Large Files Manifest — cgsd_brain_parcellation_001

> Paths and provenance for all data/files that are **OUT of git** but
> required to reproduce or continue work. Per project large-files policy
> (2026-06-25), datasets/weights/recon-all outputs/embeddings stay on local
> disk, not in the repository. This manifest is the single source of truth
> for "what lives where."

**Last updated:** 2026-07-01 (after 5-atlas sweep milestone)

---

## 1. IXI recon-all outputs (10 subjects, 4.5 GB total)

**Location:** `data/subjects/` (gitignored)

| Subject  | Path on disk | Source | Recon date | Size (approx) |
|----------|--------------|--------|------------|---------------|
| ixi002   | `data/subjects/ixi002/`   | IXI T1 (Guys) | 2026-06-28 | ~450 MB |
| ixi012   | `data/subjects/ixi012/`   | IXI T1 (HH)   | 2026-06-29 | ~450 MB |
| ixi013   | `data/subjects/ixi013/`   | IXI T1 (HH)   | 2026-06-29 | ~450 MB |
| ixi014   | `data/subjects/ixi014/`   | IXI T1 (HH)   | 2026-06-29 | ~450 MB |
| ixi015   | `data/subjects/ixi015/`   | IXI T1 (HH)   | 2026-06-29 | ~450 MB |
| ixi016   | `data/subjects/ixi016/`   | IXI T1 (Guys) | 2026-06-30 | ~450 MB |
| ixi017   | `data/subjects/ixi017/`   | IXI T1 (Guys) | 2026-06-30 | ~450 MB |
| ixi019   | `data/subjects/ixi019/`   | IXI T1 (Guys) | 2026-06-30 | ~450 MB |
| ixi020   | `data/subjects/ixi020/`   | IXI T1 (Guys) | 2026-06-30 | ~450 MB |
| ixi021   | `data/subjects/ixi021/`   | IXI T1 (Guys) | 2026-06-30 | ~450 MB |

**To re-acquire:** see `scripts/download_ixi.py` (auto-discovers tar URLs
from the IXI HTTP index) + `scripts/batch_recon_all.sh` (parallel recon-all,
~4 h per subject on 4 cores). After recon, `scripts/batch_remap_fs5.sh`
remaps each subject → fsaverage5.

**Symlinks (also gitignored, but cheap to recreate):**
- `data/subjects/fsaverage`  → `/usr/local/freesurfer/8.2.0/subjects/fsaverage`
- `data/subjects/fsaverage5` → `/usr/local/freesurfer/8.2.0/subjects/fsaverage5`

---

## 2. Parcellation atlases (gitignored — small but free to re-download)

**Location:** `data/atlas_cache/` (gitignored)

| Atlas            | K  | Source / files |
|------------------|----:|----------------|
| desikan-killiany |  68 | `$FREESURFER_HOME/subjects/fsaverage5/label/{lh,rh}.aparc.annot` |
| destrieux        | 150 | `$FREESURFER_HOME/subjects/fsaverage5/label/{lh,rh}.aparc.a2009s.annot` |
| schaefer-100     | 100 | `data/atlas_cache/schaefer-100/`, downloaded via `scripts/download_atlases.py` |
| schaefer-200     | 200 | same |
| schaefer-400     | 400 | same |

**To re-acquire:** `SUBJECTS_DIR=$PWD/data/subjects /home/woody/vp/vp3.12/bin/python
scripts/download_atlases.py` (links DK68 + Destrieux from FreeSurfer; pulls
Schaefer 2018 / 17 Networks from GitHub).

---

## 3. Pipeline / paper outputs (IN git — no manifest needed)

The following are version-controlled and **do not** appear in this manifest:

- `results/atlas_sensitivity/`        (4-atlas sweep, 240 runs)
- `results/atlas_sensitivity_5atlas/` (5-atlas sweep, 300 runs) — current
- `results/phaseA/`, `phaseB/`, `phaseC/`, `phaseD/` (per-milestone CSVs)
- `paper/paper.tex`, `paper/paper.pdf`, `paper/cas-sc.cls`,
  `paper/cas-common.sty`, `paper/thumbnails/`
- All `docs/*.md` write-ups, including `ATLAS_SENSITIVITY*.md`

For the latest git-tracked snapshot see tag
`v2026-07-01b-cgsd-brain-5atlas`.

---

## Recovery procedure (computer restart / disk failure)

```bash
# 1. Re-create FreeSurfer symlinks (if missing)
mkdir -p data/subjects
ln -sfn /usr/local/freesurfer/8.2.0/subjects/fsaverage  data/subjects/fsaverage
ln -sfn /usr/local/freesurfer/8.2.0/subjects/fsaverage5 data/subjects/fsaverage5

# 2. Re-download atlases (DK68 + Destrieux + Schaefer) — fast, ~5 min
SUBJECTS_DIR=$PWD/data/subjects \
  /home/woody/vp/vp3.12/bin/python scripts/download_atlases.py

# 3. Re-download IXI T1 subjects — sequential, ~30 min for 10 subjects
/home/woody/vp/vp3.12/bin/python scripts/download_ixi.py \
  --subjects ixi002 ixi012 ixi013 ixi014 ixi015 \
             ixi016 ixi017 ixi019 ixi020 ixi021

# 4. Re-run recon-all on each subject — ~4-8 h per subject
for sid in ixi002 ixi012 ixi013 ixi014 ixi015 \
           ixi016 ixi017 ixi019 ixi020 ixi021; do
  recon-all -subject "$sid" -i data/raw/${sid}/T1.nii.gz -all -sd data/subjects
done

# 5. Remap to fsaverage5
/home/woody/vp/vp3.12/bin/python scripts/batch_remap_fs5.py
```

After steps 1-5, all results in `results/atlas_sensitivity_5atlas/` and
`paper/paper.pdf` are reproducible from the commands in
`docs/ATLAS_SENSITIVITY_5ATLAS.md`.