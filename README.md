# LeafDoc: Early Plant Disease Diagnosis from Leaf Images

Course project for **Computer Vision (0135326)**, Al-Zaytoonah University of Jordan.
Instructor: Dr. Akram Fatayer.

**Team:** _Your name(s) here_

> Educational project. Not a substitute for an agronomist's diagnosis.

---

## 1. Problem

Plant diseases spread quickly, and late diagnosis leads to crop loss. Farmers often cannot identify a disease from a leaf by eye. LeafDoc classifies a photo of a leaf into a healthy or diseased class, so treatment can start earlier.

## 2. Goals

1. Build a **baseline CNN from scratch** (adapted from Lab 2B).
2. Improve it with **Transfer Learning** using a pretrained ResNet18: Feature Extraction (frozen backbone) vs. Fine-Tuning.
3. Evaluate with accuracy, per-class recall, and a confusion matrix.
4. Study **generalization**: does a model trained on lab-style images work on real field photos?

## 3. Dataset

**PlantVillage** (Kaggle): <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset>

- Leaf images grouped into folders named `Plant___Disease`.
- Expected: about 54,000 images and 38 classes (verify the class count after download; some PlantVillage copies on Kaggle are smaller).
- We use the **color** version.

Known limitations:

- Images are taken in a lab setting with simple backgrounds, so accuracy here is usually higher than in real fields.
- Many images show the same leaf from different angles, so a random split can place near-duplicates in train and test and inflate accuracy.

Download in Colab:

```python
import kagglehub
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print(path)
```

## 4. Repository structure

```
LeafDoc/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # class distribution, samples, duplicates, split
│   ├── 02_baseline_cnn.ipynb          # CNN from scratch (Lab 2B adapted)
│   └── 03_resnet18_transfer.ipynb     # feature extraction vs fine-tuning
├── split.json                         # fixed train/val/test split (created in notebook 01)
├── results/                           # plots, confusion matrices, metrics
├── requirements.txt
└── README.md
```

Large files (data, `.zip`, `.pth`, `runs/`) are **not** committed. See `.gitignore`.

## 5. Methods

| Model | Description |
|---|---|
| Baseline CNN | 2 conv layers (5x5) + max pooling + 3 fully connected layers, ReLU, dropout. Input 128x128 RGB. |
| ResNet18 (frozen) | ImageNet-pretrained backbone, only the new final layer is trained. |
| ResNet18 (fine-tuned) | Last block unfrozen and trained with a lower learning rate. |

Training: CrossEntropyLoss, Adam (SGD with momentum also compared), data augmentation (flip, rotation, color jitter).
Split: stratified 70% train / 15% validation / 15% test, fixed seed (42), saved to `split.json` so all models use the same data.

## 6. Results

_Fill in after running the experiments._

| Model | Val acc | Test acc | Params | Train time |
|---|---|---|---|---|
| Baseline CNN | TBD | TBD | TBD | TBD |
| ResNet18 (frozen) | TBD | TBD | TBD | TBD |
| ResNet18 (fine-tuned) | TBD | TBD | TBD | TBD |

Weakest classes: _TBD_ (see `results/`).

## 7. How to run

1. Open the notebooks in Google Colab and enable **GPU** (Runtime > Change runtime type).
2. Run `01_data_exploration.ipynb` first. It downloads the data and creates `split.json`.
3. Run `02_baseline_cnn.ipynb`, then `03_resnet18_transfer.ipynb`.

Dependencies: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`, `pillow`, `kagglehub`.

## 8. Limitations

- Lab-style images; field performance is expected to be lower.
- Possible near-duplicate leaves across splits (no leaf IDs are provided).
- The 38 classes cover a limited set of crops and diseases.
 
