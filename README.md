# MSDMT

This repo is the TF2.0 implementation of [MSDMT](https://ieeexplore.ieee.org/abstract/document/9231585).

Multi-source Data Multi-task Learning for Profiling Players in Online Games  [[PDF](https://ieee-cog.org/2020/papers/paper_45.pdf)]

MSDMT is a novel Multi-source Data Multi-task Learning approach for profiling players with both player churn and payment prediction in online games. 
On the one hand, MSDMT considers that heterogeneous multi-source data, including player portrait tabular data, behavior sequence sequential data, and social network graph data, can complement each other for a better understanding of each player.
On the other hand, MSDMT considers the significant correlation between the player churn and payment that can interact and complement each other.

## Folders
- `data/`: data of MSDMT (we only give **sample data generated randomly** to show the data format, **not the real data**).
  - `sample_data_player_portrait.csv`: the sample data for player portrait.
  - `sample_data_behavior_sequence.csv`: the sample data for behavior sequence.
  - `sample_data_social_network.csv`: the sample data for social network.
  - `sample_data_label.csv`: the sample data for label, where label1 is churn label (binary classification) and label2 is payment label (regression).
- `src/`: implementations of MSDMT.
  - `model.py`: the code for model.
  - `main.py`: the code for pipeline.

## Requirements
The code has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):
- tensorflow == 2.0
- spektral ==1.0.3
- numpy == 1.18.2
- pandas == 0.23.4
- sklearn == 0.19.1

## Training
```
$ cd src
$ python main.py 
```

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{zhao2020multi,
  title={Multi-source Data Multi-task Learning for Profiling Players in Online Games},
  author={Zhao, Shiwei and Wu, Runze and Tao, Jianrong and Qu, Manhu and Li, Hao and Fan, Changjie},
  booktitle={2020 IEEE Conference on Games (CoG)},
  pages={104--111},
  year={2020},
  organization={IEEE}
}
```
