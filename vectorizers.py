from dataset import fetch_text_extractor_console_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from stop_words import get_stop_words
import pickle


class Vectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectors = None

    @staticmethod
    def load(path_to_vectors):
        with open(path_to_vectors, 'rb') as f:
            vectors = pickle.load(f)

        return vectors

    def save(self, path_to_save):
        with open(path_to_save, 'wb') as f:
            pickle.dump(self.vectors, f)
        print(f'Save vectors to: {path_to_save}')


class Tfidf(Vectorizer):
    def __init__(self):
        super().__init__()

    def fit_transform(self, dataset):
        stop_words = get_stop_words('russian')
        stop_words.extend(get_stop_words('english'))

        self.vectorizer = TfidfVectorizer(
            use_idf=True,
            norm='l2',
            min_df=10,
            max_features=100000,
            ngram_range=(1, 3),
            stop_words=stop_words
        )

        self.vectors = self.vectorizer.fit_transform(dataset)

        return self.vectors


class CountVec(Vectorizer):
    def __init__(self):
        super().__init__()

    def fit_transform(self, dataset):
        stop_words = get_stop_words('russian')
        self.vectorizer = CountVectorizer(
            min_df=10,
            max_features=100000,
            ngram_range=(1, 3),
            stop_words=stop_words
        )

        self.vectors = self.vectorizer.fit_transform(dataset)

        return self.vectors


def save_tfidf():
    dataset_path = r'\\opr-rx-hv-smart\datasets\clasterization\data\2021.6.dirty\out'
    path_to_save = r'\\opr-rx-hv-smart\datasets\clasterization\data\vectors\2021.6.dirty.tfidf.pickle'

    dataset = fetch_text_extractor_console_data(dataset_path)
    tfidf_vec = Tfidf()
    tfidf_vec.fit_transform(dataset['text'].values)

    tfidf_vec.save(path_to_save)


def save_count():

    # clean
    dataset_path = r'\\opr-rx-hv-smart\datasets\clasterization\data\2021.6.clean\out'
    path_to_save = r'\\opr-rx-hv-smart\datasets\clasterization\data\vectors\2021.6.clean.count.pickle'

    dataset = fetch_text_extractor_console_data(dataset_path)
    count_vec = CountVec()
    count_vec.fit_transform(dataset['text'].values)

    count_vec.save(path_to_save)

    #  dirty
    dataset_path = r'\\opr-rx-hv-smart\datasets\clasterization\data\2021.6.dirty\out'
    path_to_save = r'\\opr-rx-hv-smart\datasets\clasterization\data\vectors\2021.6.dirty.count.pickle'

    dataset = fetch_text_extractor_console_data(dataset_path)
    count_vec = Tfidf()
    count_vec.fit_transform(dataset['text'].values)

    count_vec.save(path_to_save)


if __name__ == '__main__':
    pass
    # save_tfidf()
    # save_count()
