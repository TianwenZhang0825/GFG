# GFG: Glance-Focus-Gaze for SAR Ship Detection

Official implementation of the paper:  
**"Glance-Focus-Gaze: A Novel Eagle-Eye Vision-Inspired Panorama-Population-Individual Progressive Screening Paradigm to Capture Ships in SAR Images"**  
*(ISPRS Journal of Photogrammetry and Remote Sensing, 2026)*

Authors: Tianwen Zhang, Gui Gao, Xiaoling Zhang  

---

## 🚀 Overview

We propose a novel **Glance-Focus-Gaze (GFG)** paradigm for synthetic aperture radar (SAR) ship detection, inspired by the eagle-eye vision (EEV). Instead of directly localizing ships in a raw panoramic image, our method progressively screens targets through three phases:

1. **Glance Panorama** – quickly preview the whole image to build global context.
2. **Focus Population** – narrow attention to regions where ships cluster (e.g., docks, shipping lanes).
3. **Gaze Individual** – precisely locate each ship within its population.

The complete framework, **Dual Pathway Synergistic Transcendence (DPST)**, consists of:

- An **upstream GFG branch** that implements the PPI (Panorama-Population-Individual) cascade.
- A **downstream SOSS branch** (Single Outlier-Specific Shot) that catches isolated ships not belonging to any population.
- A **Dynamic Population-Awareness Orchestration (DPAO)** module that transfers population priors from GFG to SOSS for guidance.
- A **Stacking Ensemble Meta-Learning (SEML)** module that fuses the outputs of both branches for final detection.

Experiments on **SSDD** and **HRSID** datasets demonstrate state-of-the-art performance, especially in complex inshore scenes.

---

## ✨ Key Features

- **First EEV-inspired SAR ship detector** – mimics the eagle’s bi-fovea vision for coarse-to-fine screening.
- **PPI progressive screening** – reduces false alarms and missed detections by leveraging ship clustering priors.
- **Dual-branch design** – GFG handles clustered ships, SOSS handles outliers; they complement each other.
- **Population-guided attention** – uses detected populations to constrain sampling and refinement in the DETR-like SOSS branch.
- **State-of-the-art results** – **78.0% AP** on SSDD, **76.3% AP** on HRSID.
- **Lightweight glance module** – PPLVT (Panorama Perception Line Vision Transformer) achieves fast global perception with only **0.5M parameters** and **0.2G FLOPs**.

---

## 📁 Method Overview

![DPST Architecture](assets/dpst_architecture.png)  
*Figure: Detailed architecture of DPST. The upstream GFG branch detects individuals within populations, while the downstream SOSS branch captures outliers. Their results are fused by SEML.*

### Upstream GFG Branch
- **PPLVT** – lightweight vision transformer for glance panorama.
- **PDet** – population detector with IOU Constraint Loss (ICL) to allow flexible population boundaries.
- **PCCN** – population context compensation network to enrich features before individual search.
- **IDet** – individual detector with centerness (ICP) and quantity perception (IQP) heads, optimized by Kullback–Leibler distance loss (KLDL).

### Downstream SOSS Branch (based on DETR)
- **PGSS** – population-guided sparse sampling in deformable attention.
- **PGQR** – population-guided query refinement using SKF (scattering key-point features) extracted by a GNN.
- **PGAM** – population-guided anchor modulation that incorporates population priors into anchor initialization and cross-attention.

### Fusion
- **SEML** – stacking ensemble meta-learning with a quantity-aware box aggregation (QABA) algorithm.

---

## 🛠️ Installation

```bash
git clone https://github.com/TianwenZhang0825/GFG.git
cd GFG
conda create -n gfg python=3.8
conda activate gfg
pip install -r requirements.txt
```

**Requirements**  
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- torchvision  
- opencv-python  
- numpy  
- scikit-learn  
- tqdm  
- tensorboard  
- pycocotools  

*(We recommend using the exact environment listed in `environment.yml`)*

---

## 📦 Data Preparation

Download the datasets:

- **SSDD**: [Official SSDD](https://github.com/TianwenZhang0825/Official-SSDD)
- **HRSID**: [HRSID](https://github.com/chaozhong2010/HRSID)

Organize the data in the following structure:

```
data/
├── SSDD/
│   ├── JPEGImages/
│   ├── Annotations/          (COCO format JSON)
│   └── train.txt / test.txt
├── HRSID/
│   ├── JPEGImages/
│   └── annotations/           (COCO format JSON)
```

Convert annotations to COCO format if necessary (SSDD and HRSID are provided in COCO-like format in the official releases).  
We also provide a script `prepare_data.py` to automatically convert and split.

---

## 🚦 Training & Testing

All configurations are stored in `configs/`.  

### Train on SSDD

```bash
# Train DPST (GFG + SOSS + DPAO)
python tools/train.py configs/dpst_ssdd.py --work-dir work_dirs/dpst_ssdd
```

### Train on HRSID

```bash
python tools/train.py configs/dpst_hrsid.py --work-dir work_dirs/dpst_hrsid
```

### Test with pretrained weights

```bash
python tools/test.py configs/dpst_ssdd.py work_dirs/dpst_ssdd/latest.pth --eval bbox
```

### Ablation experiments

We provide configs for each variant:
- `gfg_ssdd.py` – only GFG branch
- `gfg+soss_ssdd.py` – GFG + SOSS (without DPAO)
- `dpst_ssdd.py` – full DPST (GFG+SOSS+DPAO)

---

## 📊 Results

### Main Results on SSDD and HRSID

| Method                | SSDD AP | SSDD AP₅₀ | SSDD AP₇₅ | HRSID AP | HRSID AP₅₀ | HRSID AP₇₅ |
|-----------------------|---------|-----------|-----------|----------|------------|------------|
| Baseline (Faster R-CNN) | 71.4    | 97.8      | 89.1      | 69.4     | 90.2       | 80.4       |
| GFG                   | 73.3    | 97.8      | 88.9      | 71.6     | 90.8       | 82.5       |
| GFG + SOSS            | 76.0    | 98.0      | 91.8      | 74.2     | 93.0       | 84.5       |
| **DPST (full)**       | **78.0**| **98.8**  | **93.6**  | **76.3** | **93.7**   | **85.6**   |

- On **SSDD**, DPST surpasses the second-best method M²S-DETR by **1.8% AP**.
- On **HRSID**, DPST surpasses EDHC by **3.5% AP**.

### Inshore / Offshore Performance (SSDD)

| Scene     | AP   | AP₅₀ | AP₇₅ |
|-----------|------|------|------|
| Inshore   | 77.3 | 99.4 | 92.0 |
| Offshore  | 78.7 | 98.8 | 90.0 |

For the first time, inshore accuracy slightly exceeds offshore (+0.6% AP), proving the method's robustness in complex scenes.

### Complexity

| Model          | #Params | GFLOPs | Inference Time (ms) |
|----------------|---------|--------|----------------------|
| Baseline       | 45.8M   | 120    | 24                   |
| GFG            | 55.2M   | 166    | 32                   |
| GFG+SOSS       | 62.4M   | 288    | 36                   |
| DPST (full)    | 68.3M   | 309    | 41                   |

---

## 📌 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhang2026glance,
  title={Glance-Focus-Gaze: A Novel Eagle-Eye Vision-Inspired Panorama-Population-Individual Progressive Screening Paradigm to Capture Ships in SAR Images},
  author={Zhang, Tianwen and Gao, Gui and Zhang, Xiaoling},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2026},
  note={accepted}
}
```

---

## 📄 License

This project is released under the MIT License.


---

## 🙏 Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (Grant No. U24A20589), in part by the National Key Research and Development Program of China (Grant No. 2023YFB3905504), in part by the Innovation Team of the Ministry of Education of China (Grant No. 8091B042227), and in part by the Innovation Group of Sichuan Natural Science Foundation (Grant No. 2023NSFSC1974).
