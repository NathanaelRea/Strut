from os import abort

from solver import run_case
from strut_io import load_json, parse_args


def main() raises:
    var (input_path, output_path) = parse_args()
    if input_path == "" or output_path == "":
        abort("usage: strut.mojo -- --input <case.json> --output <dir>")

    var data = load_json(input_path)
    run_case(data, output_path)


if __name__ == "__main__":
    main()
