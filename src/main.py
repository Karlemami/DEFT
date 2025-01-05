from classification.Trainer import Trainer
from preprocessing.DataLoader import DataLoader
from load_config import load_config
from utils.Colors import colors
import warnings


def main(config: dict):
    for language in config["languages"]:
        print(
            f"{colors.bold}{colors.red}\n\nCurrently training on {language}...{colors.reset}"
        )
        loader = DataLoader(config["train_path"], config["test_path"], language, drop_duplicates=config["drop_duplicates"])
        X_train, X_test, y_train, y_test = loader.get_train_test_vectorized(
            downsample=config["downsample"],
            vectorizer=config["vectorizer"],
        )
        labels = sorted(set(loader.df["y"]))
        trainer = Trainer(
            config["models"], X_train, X_test, y_train, y_test, labels, language
        )
        # trainer.get_best_params(save=True)
        trainer.compare_results(save_results=True, defined=False, save_best=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = load_config()
    print(
        f"{colors.bold}{colors.green}Train path:{colors.reset} {config['train_path']}"
    )
    print(f"{colors.bold}{colors.green}Test path:{colors.reset} {config['test_path']}")
    print(f"{colors.bold}{colors.green}Languages:{colors.reset} {config['languages']}")
    print(f"{colors.bold}{colors.green}Models:{colors.reset} {config['models']}")
    print(
        f"{colors.bold}{colors.green}Drop duplicates:{colors.reset} {config['drop_duplicates']}"
    )
    print(f"{colors.bold}{colors.green}Downsample:{colors.reset} {config['downsample']}")
    print(f"{colors.bold}{colors.green}Vectorizer:{colors.reset} {config['vectorizer']}")
    main(config)
