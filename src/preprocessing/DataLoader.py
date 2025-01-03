import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import joblib


class DataLoader:
    def __init__(self, train_path, test_path, language):
        full_df = DataLoader.convert_corpus_to_dataframe(train_path, test_path)
        self.df = full_df.query("`language` == @language")
        self.language = language

    @staticmethod
    def convert_corpus_to_dataframe(
        train_path: str = "data/deft09_parlement_appr",
        test_path: str = "data/deft09_parlement_test",
    ) -> pd.DataFrame:
        """
        Renvoie le corpus (train et test) en dataframe avec les colonnes (id: str, language: str, paragraphs[list[str]], split: str, y: str)
        """
        train_directory = Path(train_path)
        test_dictory = Path(test_path)
        files = list(train_directory.glob("*.xml")) + list(test_dictory.glob("*.xml"))
        df = pd.DataFrame()
        for f in files:
            df = pd.concat([df, DataLoader.extract_texts_from_file(f)])
        return df

    @staticmethod
    def extract_texts_from_file(path: Path) -> list[dict]:
        if "appr" in path.name:
            train = True
            split = "train"
        elif "test" in path.name:
            train = False
            split = "test"
        else:
            raise FileNotFoundError(
                "Cette fonction prend en entrée un fichier d'apprentissage ou de test"
            )

        docs = []
        tree = ET.parse(path)
        root = tree.getroot()
        language = path.name.split(".")[-2][-2:]
        ys = [] if train else DataLoader.extract_test_y(language)
        for doc in root.findall("doc"):
            doc_id = doc.get("id")
            parti = doc.find(".//PARTI")
            if train:
                ys.append(parti.get("valeur"))
            texts = ""
            for p in doc.findall(".//p"):
                texts += p.text if p.text else ""
            paragraphs = [p.text for p in doc.findall(".//p")]
            docs.append(
                {
                    "id": doc_id,
                    "language": language,
                    "paragraphs": texts,
                    "split": split,
                }
            )
        df = pd.DataFrame(docs)
        df["y"] = ys
        return df

    @staticmethod
    def extract_test_y(language) -> list[str]:
        lines = open(
            f"data/deft09_parlement_ref/deft09_parlement_ref_{language}.txt"  # TODO: remove hardcoded path
        ).readlines()
        return [line.split("\t")[-1].strip() for line in lines]

    def get_train_test_vectorized(self) -> tuple[pd.Series]:
        vectorizer = TfidfVectorizer()
        X_train = self.df["paragraphs"][self.df["split"] == "train"]
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test = self.df["paragraphs"][self.df["split"] == "test"]
        X_test_vectorized = vectorizer.transform(X_test)
        y_train = self.df["y"][self.df["split"] == "train"]
        y_test = self.df["y"][self.df["split"] == "test"]

        return X_train_vectorized, X_test_vectorized, y_train, y_test
