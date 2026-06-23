# Beyond One-Layer Decision: Hierarchical Multi-view Clustering with Progressive Cross-Layer Fusion

Published in *IEEE Transactions on Circuits and Systems for Video Technology*
(TCSVT'26).

[Paper](https://ieeexplore.ieee.org/document/11552730)
<img width="1118" height="837" alt="image" src="https://github.com/user-attachments/assets/146f20e2-18b8-4be4-a3b0-a29c1818d1e4" />

## Project Structure

```text
HAMVC-main/
├── dataloader.py          # Multi-view dataset loaders
├── loss.py                # Training loss
├── metric.py              # Clustering evaluation metrics
├── network.py             # HAMVC network
├── train.py               # Training entry point
├── test.py                # Model parameter loading and clustering evaluation
├── torch_clustering/      # PyTorch clustering implementation
├── datasets/              # Dataset files
└── models/                # Saved model parameters
```

## Dataset Preparation

Create a `datasets` directory in the project root and place the required
dataset files in it:

```text
HAMVC-main/
└── datasets/
    ├── ALOI100.mat
    ├── Animal.mat
    ├── cifar10.mat
    ├── MNIST_fea.mat
    ├── NoisyMNIST.mat
    ├── Scene15.mat
    ├── handwritten.mat
    └── uci-digit.mat
```

The exact file names and data fields must match the corresponding loaders in
`dataloader.py`.

Currently configured datasets include:

- `NoisyMNIST`
- `UCIDigit`
- `handwritten`
- `ALOI100`
- `Animal`
- `Cifar10`
- `Scene15`
- `MNIST`

## Training

Run the following command to train HAMVC:

```bash
python train.py --dataset UCIDigit
```

The main optional arguments are:

```text
--dataset
--batch_size
--learning_rate
--weight_decay
--mse_epochs
--feature_dim
--h_dims
--fusion_indices
--steps
--alpha
--temperature
--lambda1
--lambda2
```

Example with custom parameters:

```bash
python train.py \
  --dataset UCIDigit \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --steps 3 \
  --alpha 0.5 \
  --temperature 0.1 \
  --lambda1 0.1 \
  --lambda2 0.1
```

Dataset-specific epochs and random seeds are configured in `train.py`.

After training, the model parameters are saved automatically as:

```text
models/<dataset>.pth
```

For example:

```text
models/UCIDigit.pth
```

Only `model.state_dict()` is saved. Network hyperparameters are not stored in
the model file.

## Testing

Use `test.py` to load the saved model parameters and perform clustering
evaluation:

```bash
python test.py --dataset UCIDigit
```

By default, `test.py` loads:

```text
models/<dataset>.pth
```

You can specify another model file:

```bash
python test.py \
  --dataset UCIDigit \
  --model_path models/UCIDigit.pth
```

Because the model file contains only `state_dict`, the network parameters used
for testing must be identical to those used for training. For example:

```bash
python test.py \
  --dataset UCIDigit \
  --feature_dim 1024 \
  --h_dims 1024,1024,1024 \
  --fusion_indices 0,1,2,3 \
  --steps 3 \
  --alpha 0.5 \
  --temperature 0.1
```

If CUDA is available, the current code uses GPU index `1`; otherwise it uses
the CPU.

The test script reports the following clustering metrics:

- Accuracy (ACC)
- Normalized Mutual Information (NMI)
- Purity (PUR)
- Adjusted Rand Index (ARI)

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{zhou2026beyond,
  title={Beyond One-Layer Decision: Hierarchical Multi-view Clustering with Progressive Cross-Layer Fusion},
  author={Zhou, Kaixuan and Li, Pengyuan and Zhang, Jiahui and Chang, Dongxia and Wang, Yiming and Zhao, Yao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```
