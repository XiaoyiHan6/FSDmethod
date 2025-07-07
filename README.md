<h1 align="center">
<strong>Fire and Smoke Detection with Burning Intensity Representation</strong>
</h1>

<p align="center">
  <img src="./assets/images/logo_FSD.png" align="center" width="100%">

<p align="center">
  <a href="https://dl.acm.org/doi/10.1145/3696409.3700165" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-MM%20Asia%202024-1765A5?style=flat-square">
  </a>
&nbsp;&nbsp;&nbsp;
 <a href="https://arxiv.org/abs/2410.16642" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-Arxiv-FFD700?style=flat-square">
  </a>
&nbsp;&nbsp;&nbsp;
  <a href="https://xiaoyihan6.github.io/FSD/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-XiaoyiHan6/FSD-C43779?style=flat-square">
  </a>
&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/XiaoyiHan6/FSDmethod" target='_blank'>
    <img src="https://img.shields.io/badge/Code-FSD%20method-CD5C5C?style=flat-square">
  </a>
&nbsp;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/1aEx8wLQhT0u-INCtbTCzOqW-LFLQxK0P?usp=drive_link" target='_blank'>
    <img src="https://img.shields.io/badge/Weight-Model-FF9CAD?style=flat-square">
  </a> 
&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/XiaoyiHan6/FSDmethod" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=XiaoyiHan6.FSDmethod">
  </a> 
</p>
 <p align="center">
  <font size=5><strong>Fire and Smoke Detection with Burning Intensity Representation</strong></font>
    <br>
        <a href="https://xiaoyihan6.github.io/">Xiaoyi Han</a>,
	<a >Yanfei Wu</a>,
        <a href="https://tpcd.github.io/">Nan Pu</a>,
        <a href="https://person.zju.edu.cn/fengzunlei">Zunlei Feng</a>,<br>
        <a href="https://person.zju.edu.cn/zhangqf">Qifei Zhang</a>,
	<a href="https://person.zju.edu.cn/beiyj">Yijun Bei</a>,
        <a href="https://faculty.hfut.edu.cn/ChengLechao/zh_CN/index.htm">Lechao Cheng</a><br>
    <br>
  Zhejiang University & University of Trento & Hefei University of Technology
  <br>
  Accepted to MM Asia 2024
  </p>
</p>

---

<h2 align="center">
<strong>Note</strong>
</h2>

Dear Visitors,
<br>
We would like to inform you that the currently provided code supports only the following object detection models (or other components):
- SSD
- RetinaNet
- FCOS
- Attentive Transparency Detection Head (ATDH) [We placed the ATDH in FCOS]
<br>
Best regards,
<br>
Xiaoyi Han

---

<h2 align="center">
<strong>Compiling environment</strong>
</h2>

```
python == 3.8.5

torch == 1.11.0+cu113

torchaudio == 0.11.0+cu113

torchvision == 0.12.0+cu113

pycocotools == 2.0.4

numpy

Cython

matplotlib

opencv-python  (maybe you want to use skimage or PIL etc...)

scikit-image

tensorboard

tqdm

...

```

---

<h2 align="center">
<strong>Folder Organization</strong>
</h2>

I use Ubuntu20.04 (OS).

```
# Project
FSDmethod path: /data/PycharmProject/FSDmethod
├── assets
├── README.md
├── SSD ( layout-> the same as FCOS)
├── RetinaNet ( layout-> the same as FCOS)
└── MyFireNet (FCOS)
      ├── checkpoints
      ├── configs
      ├── data
      ├── log (accuracy)
      ├── models (Head->ATDH)
      ├── options
      ├── results (visualization)
      ├── tensorboard
      ├── tools
      └── utils
      
# Dataset      
Dataset path: /data/
├── 1_VisiFire (layout -> the same as MS-FSDB)
├── 2_FIRESENSE (layout -> the same as MS-FSDB)
├── 3_furg_fire-dataset (layout -> the same as MS-FSDB)
├── 4_BoWFireDataset (layout -> the same as MS-FSDB)
├── 5_FIRE_SMOKE_DATASET (layout -> the same as MS-FSDB)
└── MS-FSDB
       ├──data
       ├──images
       ├──labels
       └──layout

```
We use <strong>MyFireNet</strong> instead of the name <strong>FCOS</strong>.

---

<h2 align="center">
<strong>Training and Evaluation, Visualization</strong>
</h2>

```
# Object Detection(SSD, RetinaNet, FCOS)
# path:/data/PycharmProject/FSDmethod/Object Detection/tools

# Training 
run train.py

# Evaluation
run eval_voc.py

# Visualization
run visualize.py
```
---
<h2 align="center">
<strong>Results</strong>
</h2>

<h3 align="center">
<strong>Quantitative Results</strong>
</h3>

Table 1: Baseline model comparison across different datasets. Fire, Smoke and mAP are given in the subsection "Setting and Details". "avg" represents the average of mAP (mean Average Precision) values of all models across the FSD datasets. "s" represents the input image of small size, while "l" represents the input image of large size. F-RCNN means Faster RCNN.

  <img src="./assets/images/All_Method_FSD.png" align="center" width="100%">

---

Table 2: Comparison between generic detection heads and the Attention Transparency Detection Head (ATDH) across the MS-FSDB. Fire, Smoke and "mAP" are given in the subsetion “Setting and Details”. “s" represents the input image of small size, while “l" represents the input image of large size.

<img src="./assets/images/atdh_baselines.png" align="center" width="100%">

---

Table 3: The attention mechanism algorithm added to the baseline (FCOS) on the MS-FSDB. Fire, Smoke and "mAP" are given in the subsetion “Setting and Details”. the input image of small size is used.

<img src="./assets/images/FSD_Ablation_Study.png" align="center" width="60%">

---
<h3 align="center">
<strong>Qualitative Results</strong>
</h3>

The Detection of Transparent Targets Images in FSD, (a) the false results of generic detection, (b) that the proposed method successfully detected the previous failure result. In the diagram, blue boxes represent ground truth and red boxes represent predicted results.

<h4>Row 1 transparent targets suffers from missed detection</h4>
<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/images/4a1.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4a2.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4a3.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4a4.jpg" style="width:23%; min-width:23%; display: inline-block;">
</div>

<h4>Row 2 proposed method results</h4>
<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/images/4b1.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4b2.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4b3.jpg" style="width:23%; min-width:23%; display: inline-block;">
  <img src="assets/images/4b4.jpg" style="width:23%; min-width:23%; display: inline-block;">
</div>

---

<h2 align="center">
<strong>BibTeX</strong>
</h2>
 
```
@inproceedings{han2024mmAsia,
author = {Han, Xiaoyi and Wu, Yanfei and Pu, Nan and Feng, Zunlei and Zhang, Qifei and Bei, Yijun and Cheng, Lechao},
title = {Fire and Smoke Detection with Burning Intensity Representation},
year = {2024},
isbn = {9798400712739},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696409.3700165},
doi = {10.1145/3696409.3700165},
abstract = {An effective Fire and Smoke Detection (FSD) and analysis system is of paramount importance due to the destructive potential of fire disasters. However, many existing FSD methods directly employ generic object detection techniques without considering the transparency of fire and smoke, which leads to imprecise localization and reduces detection performance. To address this issue, a new Attentive Fire and Smoke Detection Model (a-FSDM) is proposed. This model not only retains the robust feature extraction and fusion capabilities of conventional detection algorithms but also redesigns the detection head specifically for transparent targets in FSD, termed the Attentive Transparency Detection Head (ATDH). In addition, Burning Intensity (BI) is introduced as a pivotal feature for fire-related downstream risk assessments in traditional FSD methodologies. Extensive experiments on multiple FSD datasets showcase the effectiveness and versatility of the proposed FSD model. The project is available at https://xiaoyihan6.github.io/FSD/.1},
booktitle = {Proceedings of the 6th ACM International Conference on Multimedia in Asia},
articleno = {5},
numpages = {8},
keywords = {Fire and Smoke Detection, Attentive Transparency Detection Head, Burning Intensity},
location = {
},
series = {MMAsia '24}
}


```

---

**Note**:Could you please give me a "one-click triple support"🔥 ("**Star**"🚀,"**Fork**"🔖,"**Issues**"❓)<br>
