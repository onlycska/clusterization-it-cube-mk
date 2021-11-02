from typing import List, Union
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.express as px


def plot_cluster(features, y_pred, y_labels: Union[List, pd.DataFrame] = None):
    n_dim = features.shape[-1]
    if n_dim not in [2, 3]:
        raise ValueError(
            f'To visualize, the input matrix should be either 2D or 3D but got {n_dim}'
        )

    data = {}
    for axis_name, axis in zip(['x', 'y', 'z'], range(features.shape[-1])):
        data[axis_name] = features[:, axis]

    data['cluster'] = list(map(str, y_pred))

    hover_columns = []

    if y_labels is not None:
        if isinstance(y_labels, List):
            data['context'] = list(map(str, y_labels))
            hover_columns = ['context']
        else:
            for col in y_labels.columns:
                data[col] = y_labels[col].values.tolist()
                hover_columns.append(col)

    df = pd.DataFrame(data)

    if n_dim == 2:
        if y_labels is not None:
            fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=hover_columns, width=1000, height=800)
        else:
            fig = px.scatter(df, x='x', y='y', color='cluster', width=1000, height=800)
    else:
        if y_labels is not None:
            fig = px.scatter_3d(
                df,
                x='x',
                y='y',
                z='z',
                color='cluster',
                hover_data=hover_columns,
                width=800,
                height=800
            )
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', width=800, height=800)

    fig.show()


def plot_line(data: pd.DataFrame, x: str, y: str, title: str = ''):

    fig = px.line(data, x=x, y=y, title=title)

    return fig


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmd.plot(cmap='Blues', xticks_rotation='vertical', ax=ax)
    plt.savefig('confusion_matrix.jpg')
