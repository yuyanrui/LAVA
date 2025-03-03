# Lava: Language Driven Efficient Video Analytics

Yanrui Yu, Tianfei Zhou, Jiaxin Sun, Lianpeng Qiao, Ye Yuan

## Overview

This repository contains the implementation of the paper **"Lava: Language Driven Efficient Traffic Video Analytics"**, which is currently under submission to **IEEE Transactions on Knowledge and Data Engineering**.

## Abstract
Video analytics has become crucial for processing massive volumes of traffic video in modern urban infrastructures. However, existing methods are limited by either predefined categories that restrict query flexibility or a lack of support for complex query types such as object-level aggregation and selection. Language-driven video analytics offers expressiveness and adaptability but faces challenges including sparsely distributed targets and low accuracy in dynamic, real-world traffic scenarios. Additionally, the lack of semantically rich and diverse benchmark datasets further hinders the evaluation and development of such systems.  To tackle these challenges, we propose \textsc{Lava}, a language-driven video analytics system tailored for traffic videos. \textsc{Lava} supports natural language queries to retrieve semantically rich traffic information, enabling diverse query types such as selection, aggregation, and top-k across multiple levels of granularity, from frame-level analysis to object-specific statistics. By efficiently localizing relevant segments, integrating semantic filtering, and leveraging temporal relationships, \textsc{Lava} minimizes computational overhead while maintaining high accuracy. To enable comprehensive evaluation, we develop a benchmark with diverse, semantically rich natural language predicates and fine-grained annotations across multiple datasets. Experiments on this benchmark demonstrate that \textsc{Lava} improves $F_1$-scores for selection queries by $\mathbf{15\%}$, reduces MPAE for aggregation queries by $\mathbf{0.39}$, and achieves top-$k$ precision of $\mathbf{86\%}$, while processing videos 9.6× faster than the most accurate baseline.

## Installation

To run this code, you will need Python 3.8 and the following dependencies:

* Install dassl: 
```bash
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Install other dependencies:
```bash
pip install -r requirements.txt
```

### Usage
```bash
 bash scripts/pipline.sh $DATASET $PREDICATE
```
