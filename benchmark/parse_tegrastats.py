"""Summarise a tegrastats log: power, GPU load, RAM, temperatures -> numbers for the report.

    sudo tegrastats --interval 500 --logfile power.log &      # start logging
    python3 -m deploy.run ... --max-frames 1000               # run the workload
    sudo pkill tegrastats
    python3 -m benchmark.parse_tegrastats power.log --fps 24.5

Works with Jetson Nano (POM_5V_IN) and Xavier/Orin (VDD_IN) log formats.
With --fps it also reports energy efficiency (FPS per watt, mJ per frame).
"""
import argparse
import re

import numpy as np

PATTERNS = {
    "power_mw": re.compile(r"(?:POM_5V_IN|VDD_IN) (\d+)(?:mW)?/(\d+)"),
    "gpu_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "ram_mb": re.compile(r"RAM (\d+)/(\d+)MB"),
    "swap_mb": re.compile(r"SWAP (\d+)/(\d+)MB"),
    "cpu_pct": re.compile(r"CPU \[([^\]]+)\]"),
    "gpu_temp": re.compile(r"GPU@([\d.]+)C", re.IGNORECASE),
    "cpu_temp": re.compile(r"CPU@([\d.]+)C", re.IGNORECASE),
}


def parse(lines):
    rows = []
    for line in lines:
        r = {}
        m = PATTERNS["power_mw"].search(line)
        if m:
            r["power_mw"] = float(m.group(1))
        for k in ("gpu_pct", "gpu_temp", "cpu_temp"):
            m = PATTERNS[k].search(line)
            if m:
                r[k] = float(m.group(1))
        for k in ("ram_mb", "swap_mb"):
            m = PATTERNS[k].search(line)
            if m:
                r[k], r[k + "_total"] = float(m.group(1)), float(m.group(2))
        m = PATTERNS["cpu_pct"].search(line)
        if m:
            loads = [float(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
            if loads:
                r["cpu_pct"] = float(np.mean(loads))
        if r:
            rows.append(r)
    return rows


def summarise(rows, fps=None):
    out = {"samples": len(rows)}
    for k in ("power_mw", "gpu_pct", "cpu_pct", "ram_mb", "swap_mb", "gpu_temp", "cpu_temp"):
        v = [r[k] for r in rows if k in r]
        if v:
            out[k] = {"mean": round(float(np.mean(v)), 1), "max": round(float(np.max(v)), 1)}
    if rows and "ram_mb_total" in rows[0]:
        out["ram_total_mb"] = rows[0]["ram_mb_total"]
    if fps and "power_mw" in out:
        w = out["power_mw"]["mean"] / 1000.0
        out["fps_per_watt"] = round(fps / w, 2)
        out["mj_per_frame"] = round(1000.0 * w / fps, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--fps", type=float, default=None, help="measured FPS of the workload")
    a = ap.parse_args()
    with open(a.log) as f:
        s = summarise(parse(f), a.fps)
    for k, v in s.items():
        print("{:<14} {}".format(k, v))
    if "gpu_temp" in s and s["gpu_temp"]["max"] > 80:
        print("WARNING: GPU reached {} C, so it may be thermally throttling. Add a fan.".format(s["gpu_temp"]["max"]))
    if "swap_mb" in s and s["swap_mb"]["max"] > 100:
        print("NOTE: swap was used ({} MB). Free RAM (headless mode) or memory pressure will hurt FPS.".format(
            s["swap_mb"]["max"]))


if __name__ == "__main__":
    main()
