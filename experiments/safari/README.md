# Wikitext-103 Experiments

This directory was directly adapted from [HazyResearch's safari repository](https://github.com/HazyResearch/safari).

### Setup

Install fftconv with:
```
cd csrc/fftconv && pip install .
```

Download WikiText-103 dataset:
```
mkdir data
cd data
wget https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/getdata.sh
bash getdata.sh
mv data/wikitext-103 wt103
```

### Run Wikitext-103 Experiments

```
# Hyena-RTF
DATA_PATH=data python3 train.py experiment=wt103/hyena_rtf dataset.batch_size=16 
# Hyena
DATA_PATH=data python3 train.py experiment=wt103/hyena dataset.batch_size=16
```