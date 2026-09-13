import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture
def save_dir(tmp_path):
    return tmp_path
