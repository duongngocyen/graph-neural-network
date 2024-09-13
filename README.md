# Graph attention network implementation

## Setup

```bash
conda create -n gnn python 3.12
conda activate gnn
pip install -r requirements.txt
```

## Train GAT on cora

```bash
python training_script_cora.py --should_visualize --should_test --enable_tensorboard
```

## Reference
Credit to the main reference: https://github.com/gordicaleksa/pytorch-GAT

