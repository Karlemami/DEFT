import joblib
from sklearn.metrics import accuracy_score, f1_score


class Tester:
    """
    A class to test a trained model on new data or datasets.

    Attributes
    ----------
    clf : sklearn.base.BaseEstimator
        The trained model loaded from a file.
    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        The TF-IDF vectorizer loaded from a file for preprocessing text.
    labels : list
        A list of all class labels.
    accuracy : float
        The accuracy score of the model on the latest dataset test.
    f1_score : float
        The F1 score of the model on the latest dataset test.

    Methods
    -------
    from_dataset(texts, y_true=None):
        Tests the model on a dataset of texts and optionally evaluates its accuracy and F1 score.
    from_text(text, y_true=None):
        Tests the model on a single text input and prints the predicted label.
    """

    def __init__(self, model_path: str, vectorizer_path: str, labels: list):
        """
        Initializes the Tester with a trained model, vectorizer, and labels.

        Parameters
        ----------
        model_path : str
            Path to the serialized trained model (e.g., a joblib file).
        vectorizer_path : str
            Path to the serialized TF-IDF vectorizer (e.g., a joblib file).
        labels : list
            A list of all class labels.
        """
        self.clf = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.labels = labels
        self.accuracy = 0.0
        self.f1_score = 0.0

    def from_dataset(self, texts, y_true=None):
        """
        Tests the model on a dataset of texts and optionally evaluates its performance.

        Parameters
        ----------
        texts : list of str
            A list of text documents to classify.
        y_true : list or ndarray, optional
            The true labels for the texts, used for performance evaluation. Default is None.

        Returns
        -------
        ndarray
            The predicted labels for the input texts.
        """
        X = self.vectorizer.transform(texts)
        y_hat = self.clf.predict(X)
        if y_true is not None:
            self.accuracy = accuracy_score(y_true, y_hat)
            self.f1_score = f1_score(y_true, y_hat, average="macro")
            print(f"Accuracy = {self.accuracy:.2f}")
            print(f"F1 score = {self.f1_score:.2f}")
        return y_hat

    def from_text(self, text, y_true=None):
        """
        Tests the model on a single text input and prints the predicted label.

        Parameters
        ----------
        text : str
            A single text document to classify.
        y_true : int or str, optional
            The true label for the text, used for comparison. Default is None.

        Returns
        -------
        ndarray
            The predicted label for the input text.
        """
        X_new = self.vectorizer.transform([text])
        y_hat = self.clf.predict(X_new)
        if y_true:
            print(f"Expected: {y_true}")
        print(f"Predicted: {self.labels[int(y_hat)]}")
        return y_hat
