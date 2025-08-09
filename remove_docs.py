import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import shutil
from utils.config import DATA_DIR

def clean_data_dir(directory):
    temp_file = f"{directory}.tmp"
    if os.path.exists(temp_file):
        try:
            os.unlink(temp_file)
            print(f"Removed temp file '{temp_file}'.")
        except Exception as e:
            print(f"Failed to delete temp file {temp_file}. Reason: {e}")

    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    clean_data_dir(DATA_DIR)
    print(f"Cleaned '{DATA_DIR}' directory.")
    clean_data_dir("temp")
    print(f"Cleaned 'temp' directory.")