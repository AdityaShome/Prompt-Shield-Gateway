from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="distilbert-base-uncased",
    local_dir="./model_service/models/distilbert-base-uncased",
    local_dir_use_symlinks=False
)
