from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
repo_id = "PrachiRB/Tourism-package-prediction"  # Replace with your repository ID
repo_type = "dataset"
# Initialize API client with Hugging Face authentication token
api = HfApi(token=os.getenv("HF_TOKEN"))
# Step 1: Check if the repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")
# Step 2: Upload data folder
try:
    api.upload_folder(folder_path="tourism_project/data", repo_id=repo_id, repo_type=repo_type)
    print(f"Files from 'tourism_project/data' have been uploaded successfully.")
except HfHubHTTPError as e:
    print(f"Failed to upload folder: {str(e)}")

