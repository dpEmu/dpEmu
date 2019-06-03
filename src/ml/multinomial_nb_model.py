import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from base_models import SKLearnBaseClassifier


class CustomMultinomialNB(SKLearnBaseClassifier):

    @staticmethod
    def create_pipeline():
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("multinomial_nb", MultinomialNB()),
            # ("truncated_svd", TruncatedSVD(random_state=42)),
            # ("normalizer", Normalizer(copy=False)),
            # ("linear_svc", LinearSVC(random_state=42)),
        ])


if __name__ == "__main__":
    CustomMultinomialNB(sys.argv[1:]).fit_and_optimize()
