# MSDMT

This repo is the implementation of MSDMT.

MSDMT is a novel Multi-source Data Multi-task Learning approach for profiling players with both player churn and payment prediction in online games. 
On the one hand, MSDMT considers that heterogeneous multi-source data, including player portrait tabular data, behavior sequence sequential data, and social network graph data, can complement each other for a better understanding of each player.
On the other hand, MSDMT considers the significant correlation between the player churn and payment that can interact and complement each other.

### Files in the folder

- `data/`: data of MSDMT.
  - `player_portrait.csv`: the sample data for player portrait.
  - `behavior_sequence.csv`: the sample data for behavior sequence.
  - `social_network.csv`: the sample data for social network.
- `src/`: implementations of MSDMT.
  - `preprocess.py`
  - `main.py`

### Required packages
The code has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):
- tensorflow == 1.15.0
- numpy == 1.18.2
- pandas == 0.23.4
- sklearn == 0.19.1

### Running the code 
```
$ cd src
$ python preprocess.py 
$ python main.py 
```