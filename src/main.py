import pandas as pd
import numpy as np
import json
import argparse
from classification.Classifier import Classifier
from classification.Trainer import Trainer
from preprocessing.DataLoader import DataLoader
from preprocessing.parse_data import convert_corpus_to_dataframe, get_train_test


SAVED_MODELS_PATH = "saved_models"
TRAIN_PATH = "data/deft09_parlement_appr"
TEST_PATH = "data/deft09_parlement_test"
LANGUAGE = "fr"


def main():
    loader = DataLoader(TRAIN_PATH, TEST_PATH, LANGUAGE)
    df = loader.df_unique
    print(df)

    X_train, X_test, y_train, y_test = loader.get_train_test_vectorized()
    # print(X_train, X_test, y_train, y_test)
    labels = sorted(set(df["y"]))
    print(labels)

    # vectorizer_path = f"{SAVED_MODELS_PATH}/tfidf_vectorizer.joblib"
    # X = vectorize(df, vectorizer_path)
    trainer = Trainer(["LR", "RFC"], X_train, X_test, y_train, y_test, labels, LANGUAGE)
    # trainer.get_best_params(save=True)
    trainer.compare_results(save_results=True, defined=True, save_best=False)


if __name__ == "__main__":
    main()
