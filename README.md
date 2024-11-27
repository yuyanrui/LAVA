# Lava: Trajectory-Based Query Optimization for Open-Vocabulary Video Analytics

Yanrui Yu, Tianfei Zhou, Jiaxin Sun, Lianpeng Qiao, Ye Yuan

## Overview

This repository contains the implementation of the paper **"Lava: Trajectory-Based Query Optimization for Open-Vocabulary Video Analytics"**, which is currently under submission to **ICDE 2025**.


## Abstract
Video analytics emerges as a vital tool for processing large-scale video datasets in various applications. However, existing methods are limited by either predefined categories that restrict query flexibility or a lack of support for complex query types such as object-level aggregation and selection. Language-driven video analytics offers expressiveness and adaptability but faces challenges including sparsely distributed targets and low accuracy in dynamic, real-world scenarios. Additionally, the lack of semantically rich and diverse benchmark datasets further hinders the evaluation and development of such systems. To address these issues, we propose Lava, a system that supports natural language queries to retrieve semantically rich information from videos, enabling diverse query types such as selection, aggregation, and top-k across multiple levels of granularity, from frame-level analysis to object-specific statistics. By efficiently localizing relevant segments, integrating semantic filtering, and leveraging temporal relationships, Lava minimizes computational overhead while maintaining high accuracy. To enable comprehensive evaluation, we develop a benchmark with diverse, semantically rich natural language predicates and fine-grained annotations across multiple datasets. Experiments on this benchmark demonstrate that Lava improves $F_1$-scores for selection queries by 13\%, reduces MPAE for aggregation queries by 0.29, and achieves top-$k$ precision of 84\%, while processing videos 10× faster than the most accurate baseline.


## Installation

To run this code, you will need Python 3.8 and the following dependencies:

```bash
pip install -r requirements.txt
```
### Usage
```bash
 bash scripts/pipline.sh $DATASET $PREDICATE