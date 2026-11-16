"""READ-ONLY identity-graph report for a Celeb-DF v2 test list.

    python -m offline.celebdf_graph --list /path/to/List_of_testing_videos.txt [--json out.json]

Prints the number of connected identity components, the largest component, and how many
fake videos would span splits (and so be discarded) under a random identity-level split.
It reads only the list file and writes only the optional JSON report. It assigns no
splits and drops nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from offline.celebdf import ListFormatError, graph_report, parse_test_list

DEFAULT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    parser.add_argument("--seeds", type=int, default=20, help="seeds for the split simulation")
    args = parser.parse_args(argv)

    try:
        entries = parse_test_list(args.list_path.read_text())
    except ListFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = graph_report(entries, DEFAULT_RATIOS, range(args.seeds))
    report["list_file"] = str(args.list_path)

    largest = report["largest_component"]
    sim = report["identity_level_split_simulation"]
    assert isinstance(largest, dict) and isinstance(sim, dict)
    print(f"videos: {report['videos']}  by label: {report['videos_by_label']}")
    print(
        f"identities: {report['identities']}  videos without identity: "
        f"{report['videos_without_identity']}"
    )
    print(
        f"connected components: {report['components_total']} "
        f"({report['identity_components']} with identities, "
        f"{report['singleton_components_without_identity']} identity-less singletons)"
    )
    print(
        f"largest component: {largest['videos']} videos "
        f"({largest['share_of_videos']:.1%}), {largest['identities']} identities"
    )
    print(f"identity-level split 60/20/20, seed 0: {sim['seed_0']}")
    print(
        f"discarded videos over {sim['discarded_over_seeds']['seeds']} seeds: "
        f"{sim['discarded_over_seeds']}"
    )
    print("Analysis only: no split was assigned and nothing was dropped.")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
