## Spatial Mixture Models with Learnable Deep Priors for Perceptual Grouping

This is the code repository of the paper ["Spatial Mixture Models with Learnable Deep Priors for Perceptual Grouping"](https://arxiv.org/abs/1902.02502).

### Dependencies

- pytorch == 1.0
- torchvision == 0.2
- numpy >= 1.15
- h5py >= 2.8
- scipy >= 1.1
- scikit-learn == 0.19
- matplotlib >= 2.2

### Datasets

Change the current working directory to `data` and run `create_datasets.sh`.

```bash
cd data
./create_datasets.sh
cd ..
```

### Experiments

Change the current working directory to `experiments` and run `run.sh`.

```bash
cd experiments
./run.sh
cd ..
```

Run `experiments/evaluate.ipynb` to evaluate the trained models.
