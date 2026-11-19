"""Pull new rover run logs from the Pi into runs/rover/ and optionally render ROI previews for each new run."""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import cv2

LOCAL_DIR = Path("runs/rover")
DEFAULT_HOST = os.environ.get("ROVER_HOST")
DEFAULT_REMOTE_REPO = os.environ.get("ROVER_REPO", "~/bumblebee")


def pull(host: str, remote_repo: str) -> list[Path]:
    """Rsync the Pi's runs/rover/ into LOCAL_DIR and return run directories that were not here before."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in LOCAL_DIR.iterdir() if p.is_dir()}
    cmd = ["rsync", "-az", f"{host}:{remote_repo}/runs/rover/", f"{LOCAL_DIR}/"]
    print(" ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("rsync failed; check ROVER_HOST / ROVER_REPO and that the Pi is reachable over ssh")
    return sorted(p for p in LOCAL_DIR.iterdir() if p.is_dir() and p.name not in before)


def preview(run_dir: Path, profile: str) -> int:
    """Write ROI-preview overlays for every saved frame of a run into <run>/preview/ and return how many were written."""
    from vision.lane_detection import LaneTracker
    from vision.roi_preview import render

    out_dir = run_dir / "preview"
    out_dir.mkdir(exist_ok=True)
    count = 0
    tracker = LaneTracker()
    change_ends = _lane_change_ends(run_dir / "log.csv")
    for path in sorted((run_dir / "frames").glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        out, _ = render(frame, profile, tracker)
        if int(path.stem) in change_ends:
            tracker = LaneTracker()
        cv2.imwrite(str(out_dir / path.name), out)
        count += 1
    return count


def _lane_change_ends(log_path: Path) -> set[int]:
    """Frames where a lane change finished, so the replay resets its tracker like the rover did."""
    if not log_path.exists():
        return set()
    ends, before = set(), "in_lane"
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            phase = row.get("lc_phase")
            if phase == "in_lane" and before == "changing":
                ends.add(int(row["frame"]))
            before = phase
    return ends


def main():
    parser = argparse.ArgumentParser(description="Pull rover run logs from the Pi.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="ssh target, e.g. pi@raspberrypi.local (or set ROVER_HOST)")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO, help="repo path on the Pi (or set ROVER_REPO)")
    parser.add_argument("--preview", action="store_true", help="render ROI overlays for newly pulled runs")
    parser.add_argument("--roi", default="rover", help="lane profile for --preview")
    args = parser.parse_args()
    if not args.host:
        parser.error("no Pi host: pass --host or set ROVER_HOST")

    new_runs = pull(args.host, args.remote_repo)
    if not new_runs:
        print("no new runs")
    for run_dir in new_runs:
        line = f"new run: {run_dir}"
        if args.preview:
            line += f"  ({preview(run_dir, args.roi)} preview frames -> {run_dir / 'preview'})"
        print(line)


if __name__ == "__main__":
    main()
