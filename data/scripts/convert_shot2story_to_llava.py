import json
import argparse


def convert(data, output_path):
    keys = data.keys()

    ## TODO: Map old data keys to new data keys
    write_file(data, output_path)


def write_file(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the json to process")
    parser.add_argument("output_path", help="Path to the output json")
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)
        convert(data, args.output_path)


if __name__ == "__main__":
    main()
