import os
import urllib.request as request
import zipfile
from src.text_summarizer.logging import logger
from src.text_summarizer.entity import DataIngestionConfig
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"Downloaded file and saved to {filename}")
        else:
            logger.info(f"File already exists ")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # Check if the file is actually a zip file
        if not zipfile.is_zipfile(self.config.local_data_file):
            logger.error(f"File {self.config.local_data_file} is not a valid zip file")
            # Let's check what we actually downloaded
            with open(self.config.local_data_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)  # Read first 500 characters
                logger.info(f"File content preview: {content}")
            raise ValueError(f"Downloaded file is not a valid zip file. It might be an HTML page or corrupted download.")
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file to {unzip_path}")

    