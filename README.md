# DEFT 2009

## Installation

```sh
git clone git@github.com:Karlemami/DEFT.git && cd DEFT
make
```

To download dataset:
```sh
bash data/download_data.sh
```

## Configuration

You can update `config.yaml` as follows:

```yaml
train_path: "data/deft09_parlement_appr"
test_path: "data/deft09_parlement_test"

languages:
  - "fr"  # French
  - "en"  # English
  - "it"  # Italian

models:
  #- "RFC"          # Random Forest Classifier
  #- "SVM"          # Support Vector Machine
  #- "LR"           # Logistic Regression
  - "Perceptron"   # Perceptron Algorithm

drop_duplicates: true
downsample: true
vectorizer: "doc2vec" #tfidf or doc2vec
```

## Usage

```sh
make run
# To compile and open the paper
make paper
# To delete venv etc.
make clean
```
