import json
import argparse

from decord import VideoReader
import os


# Function to recursively replace keys
def replace_keys(obj, old_key, new_key):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == old_key:
                obj[new_key] = obj.pop(old_key)
                caption = obj[new_key]

                val = [
                    {"from": "human", "value": "<image>\nCaption this video"},
                    {"from": "gpt", "value": caption},
                ]
                obj[new_key] = val
            else:
                replace_keys(obj[key], old_key, new_key)
    elif isinstance(obj, list):
        for item in obj:
            replace_keys(item, old_key, new_key)


def separate_shots(obj):
    """
    Separate shots from a given input object and organize them into a list of dictionaries representing each shot and the total video.

    Args:
        obj (dict or list): The input object containing information about shots. If a dictionary, it should contain keys
                            'video_names', 'audio_captions', 'captions', 'ASR', 'video', 'whole_ASR', and 'whole_caption'.
                            If a list, it should contain dictionaries with similar keys.

    Returns:
        list: A list of dictionaries, each representing a shot. Each dictionary contains keys 'video' and 'conversations'.
              'video' contains the name of the video, and 'conversations' contains a list of dictionaries representing
              conversations for that shot.

    Raises:
        TypeError: If the input object is not a dictionary or a list.
    """
    if isinstance(obj, dict):
        obj_shot_list = []

        # List data of each shot
        video_names = obj["video_names"]
        audio_captions = obj["audio_captions"]
        captions = obj["captions"]
        asr = obj["ASR"]

        # Add data for each shot
        for i, shot in enumerate(video_names):
            shot_obj = {}  # Create a new entry for each shot

            try:
                long_shot = os.path.join("data/videos_extracted", shot)
                vr = VideoReader(uri=long_shot)
                print("not cooked")
            except Exception as e:
                print("cooked")
                continue

            # Populate from large video
            shot_obj["video"] = shot
            shot_obj["conversations"] = [
                {"from": "human", "value": f"<image>\nPlease describe this video. Do not include details that you are not sure of. This is what the speech in the video is saying: {asr[i]}."},
                {"from": "gpt", "value": captions[i]},
            ]

            obj_shot_list.append(shot_obj)
        
        # Add data for entire video
        shot_obj = {}
        shot_obj["video"] = obj["video"]
        whole_asr = obj["whole_ASR"]
        shot_obj["conversations"] = [
            {"from": "human", "value": f"<image>\nPlease describe this video. Do not include details that you are not sure of. This is what the speech in the video is saying: {whole_asr}."},
            {"from": "gpt", "value": obj["whole_caption"]},
        ]

        obj_shot_list.append(shot_obj)
        return obj_shot_list
    elif isinstance(obj, list):
        total_list = []
        for entry in obj:
            total_list.extend(separate_shots(entry))  # Recursively call separate shots
        return total_list
    else:
        raise TypeError(f"separate_shots called with unsupported type: {type(obj)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/annotations/20k_train.json",
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/my_annotations/20k_train.json",
        help="Path to the output JSON file",
    )

    args = parser.parse_args()

    # Load the JSON data from the file
    with open(args.input, "r") as file:
        data = json.load(file)

    data = separate_shots(data)

    # Write the updated data back to the file
    with open(args.output, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
