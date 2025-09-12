import os
import sys
from PIL import Image

TARGET_SIZE = (227, 227)
FORMAT_CONFIG = {
    'ppm': {
        'mode': 'RGB',
        'save_as': 'PPM'
    },
    'pgm': {
        'mode': 'L',
        'save_as': 'PGM'
    }
}

def convert_image(source_path, target_format):
    if not os.path.exists(source_path):
        print(f"\033[31mError: file '{source_path}' not found.\033[0m", file=sys.stderr)
        return None

    try:
        img = Image.open(source_path)
    except Exception as e:
        print(f"\033[31mError occur when trying to open img file '{source_path}'\n  - Reason: \033[1m{e}\033[0m", file=sys.stderr)
        return None

    config = FORMAT_CONFIG[target_format]
    converted_img = img.convert(config['mode'])

    print(f"Resizing to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")
    converted_img = converted_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    file_name, _ = os.path.splitext(source_path)
    output_path = file_name + f".{target_format}"

    print(f"Converting '{source_path}' -> '{output_path}'")
    converted_img.save(output_path, config['save_as'])

    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"\033[1;31mUsage:\033[0m python {sys.argv[0]} <format> <file1.png> <file2.jpg> ...", file=sys.stderr)
        print(f"       python {sys.argv[0]} ppm *.png", file=sys.stderr)
        print(f"       python {sys.argv[0]} pgm *.jpg", file=sys.stderr)
        print(f"\033[1;31mSupported formats:\033[0m {', '.join(FORMAT_CONFIG.keys())}", file=sys.stderr)
        sys.exit(1)

    output_format = sys.argv[1].lower()
    target = sys.argv[2:]

    if output_format not in FORMAT_CONFIG:
        print(f"\033[1;31mError:\033[0m Unsupported format '{output_format}'.", file=sys.stderr)
        print(f"Supported formats are: {', '.join(FORMAT_CONFIG.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"Total: {len(target)} file(s) to convert to {output_format.upper()}")
    cnt = 0

    for file_path in target:
        if convert_image(file_path, output_format):
            cnt += 1

    print("-" * 50)
    if cnt == len(target):
        print(f"finished successfully")
    else:
        print(f"Only {cnt} file(s) successfully converted, which is less than {len(target)}")
        sys.exit(1)
