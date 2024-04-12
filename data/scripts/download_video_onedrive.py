import argparse
import requests
import os


def main(args):
    video_dir = os.path.join(args.data_dir, 'videos')
    if not os.path.exists(os.path.join(video_dir, 'videos')):
        os.makedirs(os.path.join(video_dir, 'videos'))

    try:
        response = requests.get(args.url, stream=True)
        response.raise_for_status()  # Raise an exception for unsuccessful requests

        with open(os.path.join(video_dir, "collation.tar.gz"), "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        print(f"File downloaded successfully to: {video_dir}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OneDrive Video Downloader')
    parser.add_argument('--url', type=str,
                        help='URL of the video file on OneDrive.',
                        default="https://qxoeeg.sn.files.1drv.com/y4mAKk5dlemi9hfp9ESmQYWwy9T5N9tDtAsoqw5XOZlUYYSOyx9H45E6bNLs4cWB6W438fNBvTT0PcX0GgVFkfaq5NZ6l_q2W7W_U6r4oPHoImfefbleLOi5KwFvz8Z7XnbOrIKwzSmFVOxAt7ayOk37aAFjCH8dAkc7Rvq_eInEG_DTRCvG4qcLaVpa3zDrhynDxmfRHeAaRGUDYbPwE_2DQ")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    args = parser.parse_args()

    main(args)
