import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


def extract_test_y(language) -> list[str]:
    lines =  open(f"../data/deft09_parlement_ref/deft09_parlement_ref_{language}.txt").readlines()
    return [line.split("\t")[-1].strip() for line in lines]

def extract_texts_from_file(path: Path) -> list[dict]:
    if "appr" in path.name:
        train = True
        split = "train"
    elif "test" in path.name:
        train = False
        split = "test"
    else:
        raise FileNotFoundError("Cette fonction prend en entrée un fichier d'apprentissage ou de test")
    
    docs = []
    tree = ET.parse(path)
    root = tree.getroot()
    language = path.name.split(".")[-2][-2:]
    if train:
        ys = []
    else:
        ys = extract_test_y(language)
    for doc in root.findall('doc'):
        doc_id = doc.get('id')
        parti = doc.find(".//PARTI")
        if train:
            ys.append(parti.get("valeur"))
        paragraphs = [p.text for p in doc.findall(".//p")]
        docs.append(
            {
                "id" : doc_id,
                "language": language,
                "paragraphs" : paragraphs,
                "split" : split,
            }
        )
    df = pd.DataFrame(docs)
    df["y"] = ys
    return df


def convert_corpus_to_dataframe(
    train_path: str = "../data/deft09_parlement_appr",
    test_path: str = "../data/deft09_parlement_test",
) -> pd.DataFrame:
    """
    Renvoie le corpus (train et test) en dataframe avec le colonnes (id: str, language: str, paragraphs[list[str]], split: str, y: str)
    """
    train_directory = Path(train_path)
    test_dictory = Path(test_path)
    files = list(train_directory.glob("*.xml")) + list(test_dictory.glob("*.xml"))
    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, extract_texts_from_file(f)])
    return df

def get_train_test(corpus: pd.DataFrame = convert_corpus_to_dataframe()) -> tuple[list[str]]:
    """
    Renvoie le tuple (X_train, X_test, y_train, y_test)
    """
    df = convert_corpus_to_dataframe()
    X_train = df["paragraphs"][df["split"]=="train"]
    X_test = df["paragraphs"][df["split"]=="test"]
    y_train = df["y"][df["split"]=="train"]
    y_test = df["y"][df["split"]=="test"]

    return X_train, X_test, y_train, y_test

