from typing import List, Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib import colormaps

from .base_plotter import BasePlotter

DEFAULT_CMAP: Colormap = colormaps.get_cmap("Blues")


class ConfusionMatrixPlotter(BasePlotter):
    """Creates a plotting object to visualize a Confusion Matrix.

    Args:
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        xaxis_name (str, optional): Label of the X axis. Defaults to "Predicted".
        yaxis_name (str, optional): Label of the Y axis. Defaults to "True".
        cmap (Colormap, optional): Color palette for the heatmap. Defaults to DEFAULT_CMAP.
    """

    def __init__(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        cmap: Colormap = DEFAULT_CMAP,
    ):
        self.__title: str = title
        self.__xaxis_name: str = xaxis_name
        self.__yaxis_name: str = yaxis_name
        self.__cmap: Colormap = cmap

    def plot(
        self,
        ax: Axes,
        conf_matrix: np.ndarray,
        class_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> Axes:
        """Generates the Confusion Matrix Heatmap Plot.

        Args:
            ax (Axes): Matplotlib Axes object on which the plot will be drawn.
            conf_matrix (np.ndarray): NumPy array representing the confusion matrix.
            class_labels (Optional[list], optional): List of class labels. Defaults to None.
            **kwargs: Additional keyword arguments. Currently not supported.

        Returns:
            Axes: The matplotlib axes containing the plot.

        Raises:
            ValueError: If confusion matrix dimensions are not square or if number of
                class labels doesn't match matrix dimensions.
        """
        if conf_matrix.shape[0] != conf_matrix.shape[1]:
            raise ValueError(
                f"Invalid dimensions: {conf_matrix.shape}. Square matrix required."
            )

        if class_labels is None:
            class_labels = list(np.arange(conf_matrix.shape[0]))

        if len(class_labels) != conf_matrix.shape[0]:
            raise ValueError(
                f"Number of class labels ({len(class_labels)}) do not match length of confusion matrix ({conf_matrix.shape[0]})."
            )

        ax.matshow(conf_matrix, cmap=self.__cmap)
        ax.set_xlabel(self.__xaxis_name)
        ax.set_ylabel(self.__yaxis_name)
        ax.set_title(self.__title)
        ax.tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
        )

        if class_labels is not None:
            ax.set_xticks(np.arange(len(class_labels)), labels=class_labels)
            ax.set_yticks(np.arange(len(class_labels)), labels=class_labels)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    conf_matrix[i, j],
                    va="center",
                    ha="center",
                    color=self.__get_text_color(conf_matrix, i, j),
                )

        ax.figure.subplots_adjust(right=1.0)

        return ax

    def __get_text_color(
        self, conf_matrix: np.ndarray, row_idx: int, col_idx: int
    ) -> str:
        """Determines the text color based on the cell value.

        Args:
            conf_matrix (np.ndarray): The confusion matrix.
            row_idx (int): Row index of the cell.
            col_idx (int): Column index of the cell.

        Returns:
            str: Color name ('black' or 'white').
        """
        max_val = conf_matrix.max()
        color = "black"
        if max_val > 0 and conf_matrix[row_idx, col_idx] / max_val > 0.5:
            color = "white"
        return color
