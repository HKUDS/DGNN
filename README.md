
**Requirements**
* python 3.7
* Pytorch 1.5+
* DGL 0.5.x
* numpy 1.17+

**Run**
``` bash
python main.py --dataset Ciao --data_path datasets/ciao/dataset.pkl --val_neg_path datasets/ciao/val_neg_samples.pkl --test_neg_path datasets/ciao/test_neg_samples.pkl
python main.py --dataset Epinions --data_path datasets/epinions/dataset.pkl --val_neg_path datasets/epinions/val_neg_samples.pkl --test_neg_path datasets/epinions/test_neg_samples.pkl
python main.py --dataset Yelp --data_path datasets/yelp/dataset.pkl --val_neg_path datasets/yelp/val_neg_samples.pkl --test_neg_path datasets/yelp/test_neg_samples.pkl
```
If OOM occurs, use --n_layers 1 instead.