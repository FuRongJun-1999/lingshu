"""Run the COLMAP sequential pair graph in bounded batches of CPU matches."""

from __future__ import annotations

from pathlib import Path
import sys

import pycolmap

from .config import positive_int


def match_pairs(database: Path, pair_list: Path, threads: int) -> None:
    positive_int(threads, "threads")
    with pycolmap.Database.open(database) as store:
        names = {image.image_id: image.name for image in store.read_all_images()}
        generator = pycolmap.SequentialPairGenerator(pycolmap.SequentialPairingOptions(overlap=10), store)
        with pair_list.open("w", encoding="utf-8", newline="\n") as stream:
            while not generator.has_finished():
                stream.writelines(f"{names[a]} {names[b]}\n" for a, b in generator.next())
        del generator

    pycolmap.match_image_pairs(
        database,
        matching_options=pycolmap.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
        pairing_options=pycolmap.ImportedPairingOptions(block_size=max(32, min(96, 4 * threads)),
                                                      match_list_path=pair_list),
        device=pycolmap.Device.cpu,
    )


if __name__ == "__main__":
    match_pairs(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
