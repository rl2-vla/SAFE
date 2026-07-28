"""
Restructures 20260508_NEW → 20260508_NEW_RESTRUCTURED to match the 20260320 layout:

  OLD (20260508_NEW):
    SEED-X_W_-.../widowx_task/episode_files

  TARGET (20260508_NEW_RESTRUCTURED):
    cover_pi_zero_simpler_latents_tasks_0-3_260508_seed0-7-42_intact/
      widowx_task/
        widowx_task_seedX/
          episode_files

Files are hard-linked (no extra disk space, instant).
"""

import os
import re
import shutil
from pathlib import Path

SRC = Path("/mnt/hdd/SAFE_ds/training_latents/SAVED/cover_pi_zero_simpler/20260508_NEW")
DST = Path("/mnt/hdd/SAFE_ds/training_latents/SAVED/cover_pi_zero_simpler/20260508_NEW_RESTRUCTURED")
TOP_FOLDER = "cover_pi_zero_simpler_latents_tasks_0-3_260508_seed0-7-42_intact"

SEED_RE = re.compile(r"^SEED-(\d+)_")


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    total = 0
    for seed_dir in sorted(SRC.iterdir()):
        if not seed_dir.is_dir():
            continue
        m = SEED_RE.match(seed_dir.name)
        if not m:
            print(f"Skipping (no seed prefix): {seed_dir.name}")
            continue
        seed_num = m.group(1)  # e.g. "0", "7", "42"

        for task_dir in sorted(seed_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name  # e.g. widowx_carrot_on_plate

            seed_subfolder = DST / TOP_FOLDER / task_name / f"{task_name}_seed{seed_num}"

            for episode_file in sorted(task_dir.iterdir()):
                if not episode_file.is_file():
                    continue
                dst_file = seed_subfolder / episode_file.name
                hardlink_or_copy(episode_file, dst_file)
                total += 1

    print(f"Done. Linked {total} files to {DST / TOP_FOLDER}")
