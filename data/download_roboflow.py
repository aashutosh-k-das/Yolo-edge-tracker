"""Download a Roboflow dataset version in YOLOv8 format (alternative to the web "Download zip").

    export ROBOFLOW_API_KEY=...            # Roboflow > Settings > API Keys  (never commit it)
    python data/download_roboflow.py --workspace my-workspace --project aerial-objects --version 1

The workspace / project ids are in the project URL: app.roboflow.com/<workspace>/<project>/<version>
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--out", default="data/raw/roboflow")
    a = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set the ROBOFLOW_API_KEY environment variable first.")

    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ds = rf.workspace(a.workspace).project(a.project).version(a.version).download("yolov8", location=a.out)
    print("Downloaded to", ds.location)
    print("Next: python data/prepare_dataset.py --roboflow {} --dry-run".format(ds.location))


if __name__ == "__main__":
    main()
