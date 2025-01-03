# DEFT 2009

## Installation

```
git clone git@github.com:Karlemami/DEFT.git && make
```

## Configuration

You can update `config.yaml` as follows:

```
train_path: "data/deft09_parlement_appr"
test_path: "data/deft09_parlement_test"

languages:
  - "fr"  # French
  - "en"  # English
  - "it"  # Italian

models:
  - "RFC"          # Random Forest Classifier
  - "SVM"          # Support Vector Machine
  - "LR"           # Logistic Regression
  - "Perceptron"   # Perceptron Algorithm

drop_duplicates: true
```

## Usage

```
make run
make clean
```
