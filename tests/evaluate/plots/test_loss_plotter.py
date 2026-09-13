import numpy as np
import matplotlib.pyplot as plt
import pytest

from mlinstruct.evaluate.plots.loss_plotter import LossPlotter


@pytest.fixture
def ax():
    _, ax = plt.subplots()
    return ax


def test_plot_without_val_losses(ax):
    train_losses = np.array([0.9, 0.6, 0.3])

    result_ax = LossPlotter().plot(ax, train_losses)

    assert result_ax.get_title() == "Training Loss per Epoch"
    assert result_ax.get_xlabel() == "Epoch"
    assert result_ax.get_ylabel() == "Loss"
    assert len(result_ax.get_lines()) == 1
    assert result_ax.get_legend() is not None
    assert [text.get_text() for text in result_ax.get_legend().get_texts()] == ["Train"]


def test_plot_with_val_losses(ax):
    train_losses = np.array([0.9, 0.6, 0.3])
    val_losses = np.array([1.0, 0.7, 0.4])

    result_ax = LossPlotter().plot(ax, train_losses, val_losses)

    assert len(result_ax.get_lines()) == 2
    legend_labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert legend_labels == ["Train", "Validation"]


def test_plot_without_legend(ax):
    train_losses = np.array([0.9, 0.6, 0.3])

    result_ax = LossPlotter(add_legend=False).plot(ax, train_losses)

    assert result_ax.get_legend() is None
