# tcm-weak-pairing-audit

Code and data splits for the paper:

> **An information-theoretic and empirical audit of prototype-bridged multimodal traditional Chinese medicine pattern classification under weak pairing**

The paper asks a negative-result question: when inquiry text and tongue images come from
*different patient populations* ("weak pairing"), can a knowledge-encoded translation layer plus
population-level visual prototypes stand in for patient-level visual evidence? We prove an
architecture-level information ceiling (Proposition 1) and a data-level ceiling (Proposition 2),
and we test the strongest instantiation we could build: a MacBERT text tower, a BiomedCLIP visual
tower, and a versioned 21×10 tongue-morphology → nature-of-disease translation layer.

**Main finding.** The two arms that receive real visual prototypes ranked last among all pipeline
arms, with every pairwise difference inside the pre-registered noise band (|Δ| < 0.0066), while the
visual branch itself reached AUC 0.8844 in isolation. The failure lies in the bridging mechanism
and in the label space of the visual benchmark, not in the visual encoder.

---

## Contents

```
data/                  data preparation and audit scripts
  resplit_dedup.py     rebuilds splits_v2 (deduplication + grouping by patient id)
  tcm_sd_loader.py     loads the TCM-SD inquiry corpus
  tongue_coco_loader.py, tongue_region_loader.py   load the TMC-Tongue annotations
  syndrome_mapping.py  10 nature-of-disease label definitions
  gen_label_coverage_audit.py   reproduces the label-space coverage audit (Table 3)
  splits_v2/           train.csv / val.csv / test.csv used in the paper
models/                text tower, visual branch, fusion, and the translation layer
  tongue_label_mapping.py   the 21x10 translation layer (MATRIX_VERSION = "2026-09-15")
  full_model_v2.py          the prototype-bridged fusion model audited in the paper
  syndrome_proto_attention.py   prototype attention (the bridge)
training/              training entry points
eval/                  metrics, reports, prototype extraction
baselines/             baseline implementations
syndrome-*.json        10 nature-of-disease keyword/definitional files
run_*.sh               run scripts used for the 21-run campaign
smoke_test*.py         fast smoke tests (no GPU required)
_SHA1_CHECK.txt        integrity checksums for the released files
```

## Data

The two datasets are **not** redistributed here. Both are obtained from their official sources:

| Dataset | Modality | Source |
|---|---|---|
| TCM-SD | inquiry text | Aliyun Tianchi dataset 139034 — https://tianchi.aliyun.com/dataset/139034 (also https://github.com/Borororo/ZY-BERT); released under **CC BY-NC-SA 4.0** |
| TMC-Tongue | tongue images | Dryad, https://doi.org/10.5061/dryad.1c59zw48r |

`data/splits_v2/` contains the exact train/validation/test split used in the paper
(35,345 / 2,078 / 4,158 records after deduplication, grouped by patient id). It is pinned by
SHA-1 in the manuscript:

```
train.csv   660b42360c13defa0870bdf0a5bfe16ae980ce87
val.csv     5b9cc024b1cbfb77135535008a25462b40d3d0b4
test.csv    1c4c06c01c484bb27e9741d7cd39afe316d7a25c
```

**Licence note for `data/splits_v2`:** these files are derived from TCM-SD and are therefore
distributed under the same licence as the source corpus, **CC BY-NC-SA 4.0** (non-commercial,
attribution, share-alike). The code in this repository is released under the MIT licence
(see `LICENSE`); the licence of the split files is independent of the licence of the code.

## Environment

Experiments were run on a single RTX 4090 (24 GB) with PyTorch 2.6.0+cu124 and Python 3.12.3.
See `requirements.txt` for the package list. Seeds 42/43/44 are used throughout.

## Reproducing the main results

```bash
python smoke_test.py                 # fast pipeline check, no GPU needed
python -m training.train --split data/splits_v2 --seed 42    # text tower
python eval/extract_prototypes_v2.py # builds the frozen prototype buffer
bash run_seed_arm.sh 42             # one arm of the fusion campaign
```

See `run_baseline_seeds.sh` and `run_upstream_seeds.sh` for the upstream (visual branch) runs.
The full campaign is 21 runs and approximately 5.5 GPU-hours.

## Citation

If you use this code, please cite the paper:

> S. Guo, "An information-theoretic and empirical audit of prototype-bridged multimodal
> traditional Chinese medicine pattern classification under weak pairing."

## Contact

Shujie Guo — College of Arts & Information Engineering, Dalian Polytechnic University,
Dalian, China. guosj@caie.edu.cn
