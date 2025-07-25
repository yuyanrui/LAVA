# LAVA: Language Driven Efficient Traffic Video Analytics

Yanrui Yu, Tianfei Zhou, Jiaxin Sun, Lianpeng Qiao, Lizhong Ding, Ye Yuan, and Guoren Wang

## News

- [2025/07/25] Our paper was accepted at **ACM Multimedia 2025 (MM '25)**, and we have released the code on [GitHub]((https://github.com/yuyanrui/Lava))!

## Overview
In modern urban environments, camera networks generate massive amounts of operational footage -- reaching petabytes each day -- making scalable video analytics essential for efficient processing.  Many existing approaches adopt an SQL-based paradigm for querying such large-scale video databases; however, this constrains queries to rigid patterns with predefined semantic categories,  significantly limiting analytical flexibility.  In this work, we explore a  language-driven video analytics paradigm aimed at enabling flexible and efficient querying of high-volume video data driven by natural language. Particularly, we build LAVA, a system that accepts natural language queries and  retrieves traffic targets across multiple levels of granularity and arbitrary categories.LAVA comprises three main components: 1) a multi-armed bandit-based efficient sampling method for video segment-level localization;
 2) a video-specific open-world detection module for object-level retrieval; and 3) a long-term object trajectory extraction scheme for temporal object association, yielding complete trajectories for object-of-interests. To support comprehensive evaluation, we further develop a novel benchmark by providing diverse, semantically rich natural language predicates and fine-grained annotations for multiple videos. Experiments on this benchmark demonstrate that LAVA improves $F_1$-scores for selection queries by $\mathbf{14\%}$, reduces MPAE for aggregation queries by $\mathbf{0.39}$, and achieves top-$k$ precision of $\mathbf{86\%}$, while processing videos $ \mathbf{9.6\times} $ faster than the most accurate baseline. 
 
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
### Dataset

```bash
The dataset will be released soon. Stay tuned!
