import json
import argparse


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
        default="data/my_annotations/20k_train",
        help="Path to the output JSON file name (excluding file extension)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default="8",
        help="Number of shards to create from annotation",
    )
    args = parser.parse_args()

    # Load the JSON data from the file
    with open(args.input, "r") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(f"Annotations are of type {type(data)} instead of list format!")

    # Calculate the size of each chunk
    chunk_size = len(data) // args.n

    split_lists = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(args.n)]

    # Write data to n different files
    for i in range(args.n):
        file_name = f"{args.output}_{i}.json"
        with open(file_name, "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
