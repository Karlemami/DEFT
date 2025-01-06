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

make sure you run the `download_data.sh` to get the data before running anything.
You can update `config.yaml` as follows:

```yaml
languages:  # Comment out languages you don't want
  - "fr"  # French
  - "en"  # English
  - "it"  # Italian

usage : train # train will run a gridsearch to find the best parameters, test will test the models with their best parameters if they exist

models: # Comment out models you don't want
  #- "RFC"          # Random Forest Classifier
  #- "SVM"          # Support Vector Machine
  #- "LR"           # Logistic Regression
  - "Perceptron"   # Perceptron Algorithm

drop_duplicates: true # whether to remove the duplicates between the train and test split (recommended)
downsample: true # whether to balance the classes size 
vectorizer: "doc2vec" # choose between tfidf, doc2vec or bert
```

## Usage

```sh
make run
# To compile and open the paper
make paper
# To delete venv etc.
make clean
```
