import pandas as pd
import numpy as np
import json
import argparse
from classification.Classifier import Classifier
from classification.Trainer import Trainer
from preprocessing.DataLoader import DataLoader
from preprocessing.parse_data import convert_corpus_to_dataframe, get_train_test


SAVED_MODELS_PATH = "saved_models"


def main():
    # loader = DataLoader()
    df = convert_corpus_to_dataframe()
    labels = sorted(set(df["y"]))
    vectorizer_path = f"{SAVED_MODELS_PATH}/tfidf_vectorizer.joblib"
    X = vectorize(df, vectorizer_path)
    trainer = Trainer(["LR", "RFC"], X, np.array(df["y"]), labels)
    trainer.compare_results(save_results=True, defined=True, save_best=False)


if __name__ == "__main__":
    main()
