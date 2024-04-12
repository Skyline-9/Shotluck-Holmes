import base64
import requests
import argparse


def download_annotation(repo_owner="bytedance", repo_name="Shot2Story", file_path="data/annotations/20k_train.json",
                        branch="master", destination_path="./downloaded_file.txt"):
    # Construct the API URL
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"

    response = requests.get(url)
    if response.status_code == 200:
        # Get content data and decode from base64
        # Save file content
        with open(destination_path, "w") as file:
            file.write(response.content.decode("utf-8"))
        print(f"File downloaded successfully to {destination_path}")
    else:
        print(f"Request failed. Status code: {response.status_code}")


def main(args):
    files = ["meta.csv", "test.json", "train.json", "val.json"]
    for file in files:
        download_annotation(file_path=f"data/annotations/20k_{file}",
                            destination_path=f"{args.data_dir}/annotations/20k_{file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotation Downloader')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    args = parser.parse_args()

    main(args)
