<h1 align="center">
<strong>Fire and Smoke Detection with Burning Intensity Representation</strong>
</h1>

<p align="center">
  <img src="./assets/images/logo_FSD.png" align="center" width="100%">

<p align="center">
  <a href="https://link.springer.com/chapter/10.1007/978-981-97-8795-1_14" target='_blank'>
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

<table>
    <caption style="text-align: left">Table 1:Baseline model comparison across different datasets. Fire, Smoke and mAP are given in the subsection "Setting and Details". "avg" represents the average of mAP (mean Average Precision) values of all models across the FSD datasets. "s" represents the input image of small size, while "l" represents the input image of large size. F-RCNN means Faster RCNN.</caption>
    <thead>
        <tr>
            <th rowspan="2">Dataset</th>
            <th colspan="3">SSD (s/l)</th>
            <th colspan="3">RetinaNet (s/l)</th>
            <th colspan="3">F-RCNN (s/l)</th>
            <th colspan="3">FCOS (s/l)</th>
            <th colspan="3">a-FSDM (s/l)</th>
            <th rowspan="2">avg (mAP)</th>
        </tr>
        <tr>
            <th>Fire</th><th>Smoke</th><th>mAP</th>
            <th>Fire</th><th>Smoke</th><th>mAP</th>
            <th>Fire</th><th>Smoke</th><th>mAP</th>
            <th>Fire</th><th>Smoke</th><th>mAP</th>
            <th>Fire</th><th>Smoke</th><th>mAP</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Fire-Smoke-Dataset (s)</td>
            <td>77.5</td><td>90.8</td><td>84.1</td>
            <td>90.7</td><td>90.7</td><td>90.7</td>
            <td>97.3</td><td>93.2</td><td>95.3</td>
            <td>97.2</td><td>98.5</td><td>97.8</td>
            <td><strong>97.5</strong></td><td><strong>99.2</strong></td><td><strong>98.3</strong></td>
            <td style="color: red;">93.2</td>
        </tr>
        <tr>
            <td>Furg-Fire-Dataset (s)</td>
            <td>75.8</td><td>86.9</td><td>81.4</td>
            <td>81.7</td><td>90.0</td><td>85.8</td>
            <td><strong>95.8</strong></td><td>93.4</td><td>94.6</td>
            <td>93.7</td><td><strong>98.1</strong></td><td>95.9</td>
            <td>94.2</td><td><strong>98.1</strong></td><td><strong>96.1</strong></td>
            <td style="color: red;">90.8</td>
        </tr>
        <tr>
            <td>VisiFire (s)</td>
            <td>78.2</td><td>89.5</td><td>83.9</td>
            <td>84.2</td><td>90.7</td><td>87.4</td>
            <td>92.8</td><td>88.8</td><td>90.8</td>
            <td>88.8</td><td>96.7</td><td>92.8</td>
            <td><strong>96.2</strong></td><td><strong>99.4</strong></td><td><strong>97.8</strong></td>
            <td style="color: red;">90.5</td>
        </tr>
        <tr>
            <td>FIRESENSE (s)</td>
            <td>89.0</td><td>90.4</td><td>89.7</td>
            <td>90.9</td><td>90.9</td><td>90.9</td>
            <td>96.8</td><td>95.8</td><td>96.3</td>
            <td>96.1</td><td>96.1</td><td>96.1</td>
            <td><strong>98.3</strong></td><td><strong>98.1</strong></td><td><strong>98.2</strong></td>
            <td style="color: red;">94.2</td>
        </tr>
        <tr>
            <td>BoWFireDataset (s)</td>
            <td>69.6</td><td>84.7</td><td>77.1</td>
            <td>72.3</td><td>88.4</td><td>80.3</td>
            <td><strong>86.3</strong></td><td>95.0</td><td>90.6</td>
            <td>85.1</td><td>95.2</td><td>90.2</td>
            <td>86.1</td><td><strong>97.9</strong></td><td><strong>92.0</strong></td>
            <td style="color: red;">86.0</td>
        </tr>
        <tr>
            <td>miniMS-FSDB (s)</td>
            <td>71.2</td><td>84.4</td><td>77.8</td>
            <td>80.4</td><td>89.4</td><td>84.9</td>
            <td>98.0</td><td>93.0</td><td>95.5</td>
            <td>94.1</td><td>95.9</td><td>95.0</td>
            <td><strong>98.3</strong></td><td><strong>99.3</strong></td><td><strong>98.8</strong></td>
            <td style="color: red;">90.4</td>
        </tr>
        <tr>
            <td>MS-FSDB (s)</td>
            <td>81.0</td><td>90.2</td><td>85.6</td>
            <td>81.0</td><td>90.5</td><td>85.8</td>
            <td><strong>98.2</strong></td><td>93.5</td><td>95.8</td>
            <td>95.6</td><td>98.4</td><td>97.0</td>
            <td>97.1</td><td><strong>98.9</strong></td><td><strong>98.0</strong></td>
            <td style="color: red;">92.4</td>
        </tr>
        <tr>
            <td>miniMS-FSDB (l)</td>
            <td>75.9</td><td>87.1</td><td>81.5</td>
            <td>87.0</td><td>88.4</td><td>87.7</td>
            <td>98.0</td><td>94.1</td><td>96.1</td>
            <td>95.5</td><td>96.2</td><td>95.8</td>
            <td><strong>98.1</strong></td><td><strong>97.6</strong></td><td><strong>97.9</strong></td>
            <td style="color: red;">91.8</td>
        </tr>
        <tr>
            <td>MS-FSDB (l)</td>
            <td>88.0</td><td>90.9</td><td>89.4</td>
            <td>89.5</td><td>89.6</td><td>89.6</td>
            <td>97.3</td><td>95.9</td><td>96.6</td>
            <td>96.0</td><td>96.7</td><td>96.3</td>
            <td><strong>98.4</strong></td><td><strong>98.6</strong></td><td><strong>98.5</strong></td>
            <td style="color: red;">94.1</td>
        </tr>
    </tbody>
</table>

<style>
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: Arial, sans-serif;
    }
    caption {
        font-weight: bold;
        margin-bottom: 10px;
        text-align: left;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    th {
        background-color: #f2f2f2;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>

---

<table>
    <caption style="text-align: left">Table 2:Comparison between generic detection heads and the Attention Transparency Detection Head (ATDH) across the MS-FSDB. Fire, Smoke and mAP are given in the subsection "Setting and Details". "s" represents the input image of small size, while "l" represents the input image of large size.
    </caption>
    <thead>
        <tr>
            <th rowspan="2" style="border-bottom: 2px solid #000; text-align: left; padding: 8px;">Model</th>
            <th rowspan="2" style="border-bottom: 2px solid #000; text-align: left; padding: 8px;">Dataset</th>
            <th colspan="3" style="border-bottom: 1px solid #ddd; padding: 8px;">miniMS-FSDB(s)</th>
            <th colspan="3" style="border-bottom: 1px solid #ddd; padding: 8px;">MS-FSDB(s)</th>
            <th colspan="3" style="border-bottom: 1px solid #ddd; padding: 8px;">miniMS-FSDB(l)</th>
            <th colspan="3" style="border-bottom: 1px solid #ddd; padding: 8px;">MS-FSDB(l)</th>
        </tr>
        <tr>
            <th style="padding: 8px;">Fire</th>
            <th style="padding: 8px;">Smoke</th>
            <th style="padding: 8px;">mAP</th>
            <th style="padding: 8px;">Fire</th>
            <th style="padding: 8px;">Smoke</th>
            <th style="padding: 8px;">mAP</th>
            <th style="padding: 8px;">Fire</th>
            <th style="padding: 8px;">Smoke</th>
            <th style="padding: 8px;">mAP</th>
            <th style="padding: 8px;">Fire</th>
            <th style="padding: 8px;">Smoke</th>
            <th style="padding: 8px;">mAP</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" style="border-left: 1px solid #000; text-align: left; padding: 8px; vertical-align: top;">SSD</td>
            <td style="text-align: left; padding: 8px;"></td>
            <td>71.2</td><td>84.4</td><td>77.8</td>
            <td>81.0</td><td>90.2</td><td>85.6</td>
            <td>75.9</td><td>87.1</td><td>81.5</td>
            <td>88.0</td><td><strong>90.9</strong></td><td>89.4</td>
        </tr>
        <tr>
            <td style="text-align: left; padding: 8px;">+ATDH</td>
            <td><strong>89.8</strong></td><td><strong>90.9</strong></td><td><strong>90.3</strong></td>
            <td><strong>87.7</strong></td><td><strong>90.9</strong></td><td><strong>89.3</strong></td>
            <td><strong>90.5</strong></td><td><strong>90.9</strong></td><td><strong>90.7</strong></td>
            <td><strong>89.6</strong></td><td>90.8</td><td><strong>90.2</strong></td>
        </tr>
        
        <tr style="border-top: 1px solid #ddd;">
            <td rowspan="2" style="border-left: 1px solid #000; text-align: left; padding: 8px; vertical-align: top;">RetinaNet</td>
            <td style="text-align: left; padding: 8px;"></td>
            <td>80.4</td><td>89.4</td><td>84.9</td>
            <td>81.0</td><td>90.5</td><td>85.8</td>
            <td>87.0</td><td>88.4</td><td>87.7</td>
            <td>89.5</td><td>89.6</td><td>89.6</td>
        </tr>
        <tr>
            <td style="text-align: left; padding: 8px;">+ATDH</td>
            <td><strong>87.8</strong></td><td><strong>90.9</strong></td><td><strong>89.3</strong></td>
            <td><strong>81.8</strong></td><td><strong>90.9</strong></td><td><strong>86.4</strong></td>
            <td><strong>90.4</strong></td><td><strong>90.0</strong></td><td><strong>90.2</strong></td>
            <td><strong>90.8</strong></td><td><strong>90.9</strong></td><td><strong>90.7</strong></td>
        </tr>
        
        <tr style="border-top: 1px solid #ddd;">
            <td rowspan="2" style="border-left: 1px solid #000; text-align: left; padding: 8px; vertical-align: top;">FCOS</td>
            <td style="text-align: left; padding: 8px;"></td>
            <td>94.1</td><td>95.9</td><td>95.0</td>
            <td>95.6</td><td>98.4</td><td>97.0</td>
            <td>95.5</td><td>96.2</td><td>95.8</td>
            <td>96.0</td><td>96.7</td><td>96.3</td>
        </tr>
        <tr style="border-bottom: 2px solid #000;">
            <td style="text-align: left; padding: 8px;">+ATDH</td>
            <td><strong>98.3</strong></td><td><strong>99.3</strong></td><td><strong>98.8</strong></td>
            <td><strong>97.1</strong></td><td><strong>98.9</strong></td><td><strong>98.0</strong></td>
            <td><strong>98.1</strong></td><td><strong>97.6</strong></td><td><strong>97.9</strong></td>
            <td><strong>98.4</strong></td><td><strong>98.6</strong></td><td><strong>98.5</strong></td>
        </tr>
    </tbody>
</table>


---

**Note**:Could you please give me a "one-click triple support"🔥 ("**Star**"🚀,"**Fork**"🔖,"**Issues**"❓)<br>
