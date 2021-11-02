""" Document corpus reader """

from typing import List
import pathlib
import pandas as pd
import pickle
import random

from tqdm import tqdm
from pathlib import Path


TEXT_LABEL = 'text'
TARGET_LABEL = 'target'
TARGET_NAME_LABEL = 'target_name'
FILE_NAME = 'file_name'
FILE_PATH = 'file_path'


class DatasetTextExtractor:
    def __init__(self):
        self.dataset = None

    def read(self, root_path):
        self.dataset = fetch_text_extractor_console_data(root_path)

        return self.dataset

    def save(self, path_to_save):
        if self.dataset is not None:
            with open(path_to_save, 'wb') as f:
                pickle.dump(self.dataset, f)
            print(f'Save to {path_to_save}')

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def get_label_index(label: str, labels_list: List) -> int:
    return labels_list.index(label)


def fetch_text_extractor_console_data(root_path) -> pd.DataFrame:

    root_folder = pathlib.Path(root_path)

    rows = []
    for class_folder in root_folder.iterdir():
        for file in tqdm(class_folder.glob('*.txt'), f'{class_folder.stem}'):
            with open(file, 'r', encoding='utf8') as f:
                text = f.read()

                record = {
                    TARGET_NAME_LABEL: class_folder.stem,
                    TEXT_LABEL: text,
                    FILE_NAME: file.name,
                    FILE_PATH: file
                }
                rows.append(record)

    data = pd.DataFrame(rows)

    target_names_list = data[TARGET_NAME_LABEL].unique().tolist()

    data[TARGET_LABEL] = data[TARGET_NAME_LABEL].apply(lambda x: get_label_index(x, target_names_list))

    # show some stat
    print(f'{data.shape[0]} documents')
    unique_labels = data[TARGET_NAME_LABEL].unique().tolist()
    print(f'Кол-во классов: {len(unique_labels)}, {unique_labels}')

    return data


def add_pred_target_name(
        df: pd.DataFrame,
        target: str,
        target_name_column: str,
        new_target_name_column: str
) -> pd.DataFrame:

    """
    target - колонка в df c предсказанными кластерами
    target_name_column - колонка в df c текстовым именем исходного кластера
    new_target_name_column - колонка в df c текстовым именем предсказанного кластера
    """

    # df = dataframe.copy()

    # получить уникальные значения в target
    labels = df[target].unique().tolist()

    for label in labels:
        df_label = df.loc[df[target] == label]
        target_name_max_count = df_label[target_name_column].value_counts().idxmax()

        df.loc[df[target] == label, new_target_name_column] = target_name_max_count

    # return df


def check_add_pred_target_name():
    kind_doc = ['Акт', 'СФ', 'ТОРГ12']

    records = []
    for i in range(300):
        records.append(
            {
                'kind': random.choice(kind_doc),
                'y_pred': random.randint(0, 3)
            }
        )

    test_dataset = pd.DataFrame(records)

    res = add_pred_target_name(test_dataset, 'y_pred', 'kind', 'kind_pred')
    print(res)


def test_read_save_dataset():
    r_path = Path(r'./')
    dataset_raw_path = r_path.joinpath(r'dataset')
    dataset_save_path = r_path.joinpath(r'saved_dataset.pickle')

    dl = DatasetTextExtractor()
    dataset = dl.read(dataset_raw_path)

    dl.save(dataset_save_path)


if __name__ == '__main__':
    test_read_save_dataset()
    # check_add_pred_target_name()
