# Semi-Supervised Disentangled Framework for Transferable Named Entity Recognition

This is **TensorFlow** implementation for paper :

- Zhifeng Hao, Di Lv, Zijian Li, Ruichu Cai, Wen Wen, Boyan Xu "[Semi-Supervised Disentangled Framework for Transferable Named Entity Recognition](#)", NEURAL NETWORKS（NN）


![overview](/figures/SSD.jpg)

## Requirements
- python 3.x with package tensorflow-gpu (`1.10.0`)+,tabulate,matplotlib,numpy

## Usage
Run SSD model for cross-domain transfer setting on Ritter2011 NER task. More parameters setting in config.py.
```shell script
python SSD.py --config_path=../config/dsr_on_r1.cfg
```

Run SSD model for cross-lingual transfer setting on Spanish NER task.More parameters setting in config.py.

