import argparse
from pathlib import Path

def split_diode_lists(data_root, txt_in):
    data_root = Path(data_root)
    txt_in = Path(txt_in)
    if not txt_in.is_absolute():
        txt_in = data_root / txt_in

    if not txt_in.exists():
        raise FileNotFoundError(f"Input file not found: {txt_in}")

    # Example: diode_val_all_filename_list_N1000.txt
    stem = txt_in.stem  # diode_val_all_filename_list_N1000
    suffix = txt_in.suffix  # .txt

    indoor_out = txt_in.with_name(stem.replace("all", "indoor") + suffix)
    outdoor_out = txt_in.with_name(stem.replace("all", "outdoor") + suffix)

    with open(txt_in, "r") as f:
        lines = f.readlines()

    indoor_lines = [l for l in lines if l.startswith("indoors/")]
    outdoor_lines = [l for l in lines if l.startswith("outdoor/")]

    with open(indoor_out, "w") as f:
        f.writelines(indoor_lines)
    with open(outdoor_out, "w") as f:
        f.writelines(outdoor_lines)

    print(f"Split complete:")
    print(f"  {len(indoor_lines)} indoor lines -> {indoor_out.name}")
    print(f"  {len(outdoor_lines)} outdoor lines -> {outdoor_out.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split combined DIODE filename list into indoor/outdoor subsets.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing the diode_val dataset and list files")
    parser.add_argument("--txt_in", type=str, required=True,
                        help="Full or relative path to the combined filename list (e.g. diode_val_all_filename_list_N1000.txt)")
    args = parser.parse_args()

    split_diode_lists(args.data_root, args.txt_in)
