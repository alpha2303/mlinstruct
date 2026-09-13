import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlinstruct.evaluate.plots.roc_plotter import ROCPlotter


@pytest.fixture
def ax():
    _, ax = plt.subplots()
    return ax


@pytest.fixture
def fpr_tpr():
    return np.array([0.0, 0.2, 1.0]), np.array([0.0, 0.8, 1.0])


def test_plot_without_auc(ax, fpr_tpr):
    fpr, tpr = fpr_tpr

    result_ax = ROCPlotter().plot(ax, fpr, tpr)

    assert result_ax.get_title() == "Receiver operating characteristic (ROC) curve"
    assert result_ax.get_xlabel() == "False Positive Rate"
    assert result_ax.get_ylabel() == "True Positive Rate"
    assert len(result_ax.get_lines()) == 2
    legend_labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert legend_labels == ["ROC Curve"]


def test_plot_with_auc(ax, fpr_tpr):
    fpr, tpr = fpr_tpr

    result_ax = ROCPlotter().plot(ax, fpr, tpr, auc=0.85)

    legend_labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert legend_labels == ["ROC Curve(AUC = 0.85)"]


def test_plot_without_legend(ax, fpr_tpr):
    fpr, tpr = fpr_tpr

    result_ax = ROCPlotter(add_legend=False).plot(ax, fpr, tpr)

    assert result_ax.get_legend() is None
