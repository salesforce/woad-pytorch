# WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos

## Introduction

- Pytorch implementation of [`WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos`](https://arxiv.org/pdf/2006.03732.pdf)

## Environment

- The code is developed with CUDA V9.0, Python 3.6.3

## Install
- pip install -r requirements.txt

## Data Preparation
- Download Thumos14 annotations from [`here`](https://github.com/sujoyp/wtalc-pytorch/tree/master/Thumos14reduced-Annotations)

- Download Thumos14reduced-I3D-JOINTFeatures from [`here`](https://github.com/sujoyp/wtalc-pytorch#data)

- Put the downloaded annotations under Thumos14reduced-Annotations/ and features under data/

## Pretrained Models
- THUMOS'14 [`weakly-supervised model`](https://storage.googleapis.com/sfr-mingfei-woad-models/thumos_weak_final.pkl) and [`strongly-supervised model`](https://storage.googleapis.com/sfr-mingfei-woad-models/thumos_sup_final.pkl)


## Evaluation

```
python eval.py --pretrained-ckpt MODEL_NAME
```
## Training

```
python main.py --supervision SUPERVISION_TYPE --model-name NAME_TO_SAVE_MODEL
```

## Citations
- If you find this codebase useful, please cite our paper:
```
@inproceedings{gao2021woad,
    title = {WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos},
    author = {Mingfei Gao, Yingbo Zhou, Ran Xu, Richard Socher, Caiming Xiong},
    booktitle = {CVPR},
    year = {2021}
}
```

## Contact
- Please send an email to mingfei.gao@salesforce.com if you have questions.

## Acknowledgement
We referenced [W-TALC](https://github.com/sujoyp/wtalc-pytorch) for the code.