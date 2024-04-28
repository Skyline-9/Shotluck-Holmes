import json
import argparse


# Function to recursively replace keys
def replace_keys(obj, old_key, new_key):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == old_key:
                obj[new_key] = obj.pop(old_key)
                caption = obj[new_key]

                val = [
                    {
                        "from": "human",
                        "value": "<image>\nCaption this video"
                    }, {
                        "from": "gpt",
                        "value": caption
                    }
                ]
                obj[new_key] = val
            else:
                replace_keys(obj[key], old_key, new_key)
    elif isinstance(obj, list):
        for item in obj:
            replace_keys(item, old_key, new_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str,
                        default="/home/hice1/apeng39/scratch/Shotluck-Holmes/data/my_annotations/20k_val.json",
                        help='Path to the input JSON file')
    parser.add_argument('--o', type=str,
                        default="/home/hice1/apeng39/scratch/Shotluck-Holmes/data/my_annotations/20k_val.json",
                        help='Path to the output JSON file')

    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Replace keys recursively
    replace_keys(data, 'whole_caption', 'conversations')

    # Write the updated data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()
