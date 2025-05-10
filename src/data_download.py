# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with other API's is wanted to be used, the code will need to be changed.

import os
import sys
from pathlib import Path

import kaggle

from .common import tools


config = tools.load_config()

try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.


def kaggle_download_data(
    data_slug: str,
    data_name: str,
    save_path: Path,
):

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api_key"]

    api_cli = kaggle.KaggleApi()

    # Download the lego piece dataset from kaggle.com
    api_cli.authenticate()

    logger.debug(f"Kaggle config: {api_cli.config_values}")

    save_path = save_path / data_name

    logger.info(
        f"Downloading files..."
        f"\n    From:  {data_slug}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
    )

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    api_cli.dataset_download_files(data_slug, path=save_path, unzip=True, quiet=False)


if __name__ == "__main__":
    repo_root_dir: Path = Path(__file__).parent.parent

    save_path: Path = repo_root_dir / config["data_path"] / "testing"

    kaggle_download_data(
        data_slug="andonians/random-linear-regression",
        save_path=save_path,
        data_name="random-linear-regression",
    )
