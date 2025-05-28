from huggingface_hub import snapshot_download
folder = snapshot_download(
            "HuggingFaceFW/fineweb-edu", 
            repo_type="dataset",
            local_dir="./fineweb-edu/",
            # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
            allow_patterns="sample/10BT/*")