import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict
from pathlib import Path


def extract_texts_from_file(path: Path) -> List[Dict]:
    docs = []
    tree = ET.parse(path)
    root = tree.getroot()

    for doc in root.findall("doc"):
        doc_id = doc.get("id")
        language = doc_id.split("_")[1][:2]
        parti = doc.find(".//PARTI")
        nom_parti = parti.get("valeur")
        for text in doc.findall("texte"):
            paragraphs = [p.text for p in text.findall("p")]
        docs.append(
            {
                "id": doc_id,
                "language": language,
                "parti": nom_parti,
                "paragraphs": paragraphs,
            }
        )
    return docs


def convert_corpus_to_dataframe(
    path: str = "../data/deft09_parlement_appr",
) -> pd.DataFrame:
    directory = Path(path)
    docs = []
    for f in directory.glob("*.xml"):
        docs.extend(extract_texts_from_file(f))
    return pd.DataFrame(docs)
