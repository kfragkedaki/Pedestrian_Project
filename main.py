from utils.data_reader import SinD
import warnings
import os

ROOT = os.getcwd()
warnings.filterwarnings("ignore")


def get_data():
    """
    Load Full Daatset
    """

    if "sind.pkl" not in os.listdir(ROOT + "/resources"):
        sind = SinD()
        data = sind.data(input_len=90)
        _ = sind.labels(data, input_len=90)


if __name__ == "__main__":
    # NOTE: Get the full dataset for best results

    get_data()
