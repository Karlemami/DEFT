import yaml
import argparse


def validate_config(config: dict):
    for model in config["models"]:
        if model not in ["RFC", "SVM", "LR", "Perceptron"]:
            raise ValueError("Invalid model specified.")
    for language in config["languages"]:
        if language not in ["fr", "en", "it"]:
            raise ValueError("Invalid language selected.")
    if not isinstance(config["train_path"], str):
        raise ValueError("Train path must be a string.")
    if not isinstance(config["test_path"], str):
        raise ValueError("Test path must be a string.")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as inf:
        config = yaml.safe_load(inf)
    validate_config(config)
    return config
