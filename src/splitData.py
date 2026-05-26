import sys
import argparse
from utils import splitDataset

def main():
    parser = argparse.ArgumentParser(description="Split data arguments")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to save the training dataset")
    parser.add_argument("--validation_path", type=str, required=True, help="Path to save the validation dataset")
    args = parser.parse_args()
    try:
        splitDataset(args.data_path, args.train_path, args.validation_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Train set succesfully saved at {args.train_path}.")
    print(f"Validation set succesfully saved at {args.validation_path}.")
    return 0


if __name__ == "__main__":
    main()