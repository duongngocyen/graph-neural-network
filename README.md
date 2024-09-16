# Graph attention network implementation

## Setup

```bash
conda create -n gnn python 3.12
conda activate gnn
pip install -r requirements.txt
```

## Train GAT on cora

For CORA:
```bash
python training_script_cora.py --should_visualize --should_test --enable_tensorboard --batch_size 32
```

For PPI:
```bash
python training_script_ppi.py --should_visualize --should_test --enable_tensorboard --num_of_epochs 100 --batch_size 8
```

## Tracking training and evaluation result

```bash
tensorboard --logdir runs
```

## Reference
Credit to the main reference: https://github.com/gordicaleksa/pytorch-GAT

