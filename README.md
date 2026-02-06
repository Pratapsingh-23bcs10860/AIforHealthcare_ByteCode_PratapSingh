# MRI CN vs AD training

Small training package to train a binary classifier (Cognitively Normal vs Alzheimer Disease) on the provided preprocessed `.npy` subjects.

Files added:

- `app/dataset.py` - `MRIDataset` loads `.npy` files listed in `data/*csv` and filters to CN/AD.
- `app/model.py` - small CNN that accepts slices as input channels.
- `app/train.py` - training script, saves best model to `models/best_model.pth`.
- `requirements.txt` - minimal packages.

Quick run (from repo root):

```bash
python -m app.train --epochs 1 --batch_size 8
```
