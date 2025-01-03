import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
import pandas as pd
from pathlib import Path
import joblib


class DataLoader:
    def __init__(self, train_path, test_path, language):
        full_df = DataLoader.convert_corpus_to_dataframe(train_path, test_path)
        self.df = full_df.query("`language` == @language and `y` != ''").reset_index(
            drop=True
        )
        self.df_unique = self.df.drop_duplicates(subset="paragraphs").reset_index(
            drop=True
        )
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

    def get_train_test_vectorized(self, drop_duplicates=True) -> tuple[pd.Series]:
        vectorizer = TfidfVectorizer()
        df = self.df_unique if drop_duplicates else self.df
        for i in range(2):
            df = DataLoader.get_downsampled(df)
        X_train = df["paragraphs"][df["split"] == "train"]
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test = df["paragraphs"][df["split"] == "test"]
        X_test_vectorized = vectorizer.transform(X_test)
        y_train = df["y"][df["split"] == "train"]
        y_test = df["y"][df["split"] == "test"]

        return X_train_vectorized, X_test_vectorized, y_train, y_test

    @staticmethod
    def get_downsampled(df) -> pd.DataFrame:
        """
        Balances the dataset by downsampling the majority class.
        NB: Only useful when 1 class has much more documents than the others.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame with a "y" column for class labels.

        Returns
        -------
        pd.DataFrame
            A DataFrame with a balanced class distribution, where the size of
            each class is reduced to the median class size.
        """
        class_counts = df["y"].value_counts()
        biggest_class = class_counts.idxmax()
        # We separate the majority class from the rest of the samples
        biggest_class_df = df.query("`y` == @biggest_class")
        df_without_biggest = df.query("`y` != @biggest_class")
        resampled_class = resample(
            biggest_class_df,
            replace=False,
            n_samples=int(class_counts.median()),  # reduce n_sample to median
            random_state=42,
        )
        # and then concatenate them after reducing the size
        return pd.concat([df_without_biggest, resampled_class])
