import json

import pytest

from offline import celebdf_graph
from offline.celebdf import (
    ListFormatError,
    UnionFind,
    graph_report,
    identity_components,
    parse_test_list,
    simulate_identity_split,
)
from offline.fixture import fixture_videos, fixture_list_text

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_parse_labels_identities_and_ids():
    entries = parse_test_list(
        "1 YouTube-real/00170.mp4\n0 Celeb-synthesis/id0_id16_0005.mp4\n1 Celeb-real/id3_0002.mp4\n"
    )
    yt, fake, real = entries
    assert (yt.label, yt.identities, yt.y) == ("real", (), 0)
    assert (fake.label, fake.identities, fake.y) == ("fake", ("id0", "id16"), 1)
    assert real.identities == ("id3",)
    assert fake.video_id == "Celeb-synthesis__id0_id16_0005"


@pytest.mark.parametrize(
    "text, message",
    [
        ("0 Celeb-real/id0_0000.mp4\n", "contradicts folder"),
        ("1 Celeb-synthesis/id0_id1_0000.mp4\n", "contradicts folder"),
        ("2 Celeb-real/id0_0000.mp4\n", "expected"),
        ("1 Other/id0_0000.mp4\n", "unexpected path"),
        ("1 Celeb-real/alice_0000.mp4\n", "unexpected file name"),
        ("1 Celeb-real/id0_0000.mp4\n1 Celeb-real/id0_0000.mp4\n", "duplicate"),
        ("\n\n", "empty"),
    ],
)
def test_parse_refuses_bad_lists(text, message):
    with pytest.raises(ListFormatError, match=message):
        parse_test_list(text)


def test_union_find():
    uf = UnionFind()
    for x in "abcde":
        uf.add(x)
    uf.union("a", "b")
    uf.union("d", "c")
    uf.union("b", "c")
    assert len({uf.find(x) for x in "abcd"}) == 1 and uf.find("e") == "e"
    assert uf.find("d") == "a"  # smallest root wins: deterministic


def test_components_join_both_identities_of_a_fake():
    entries = parse_test_list(
        "1 Celeb-real/id0_0000.mp4\n"
        "1 Celeb-real/id1_0000.mp4\n"
        "0 Celeb-synthesis/id0_id1_0000.mp4\n"
        "1 Celeb-real/id2_0000.mp4\n"
        "1 YouTube-real/00001.mp4\n"
        "1 YouTube-real/00002.mp4\n"
    )
    comps = identity_components(entries)
    assert [len(c.videos) for c in comps] == [3, 1, 1, 1]
    assert comps[0].identities == ("id0", "id1")
    assert sum(1 for c in comps if c.group_id.startswith("solo:")) == 2


def test_simulation_counts_spanning_fakes_without_dropping():
    # A chain id0-id1-...-id9: one giant component; random identity splits must cut fakes.
    lines = [f"1 Celeb-real/id{i}_0000.mp4" for i in range(10)]
    lines += [f"0 Celeb-synthesis/id{i}_id{i + 1}_0000.mp4" for i in range(9)]
    entries = parse_test_list("\n".join(lines))
    sim = simulate_identity_split(entries, RATIOS, seed=0)
    kept = sum(sim["videos_kept_per_split"].values())
    assert sim["spanning_fake_videos"] == sim["videos_discarded"] > 0
    assert kept + sim["videos_discarded"] == len(entries)
    assert simulate_identity_split(entries, RATIOS, seed=0) == sim  # deterministic
    report = graph_report(entries, RATIOS, range(5))
    assert report["components_total"] == 1
    assert report["largest_component"]["share_of_videos"] == 1.0


def test_fixture_graph_report(tmp_path, capsys):
    path = tmp_path / "List_of_testing_videos.txt"
    path.write_text(fixture_list_text())
    out = tmp_path / "report.json"
    assert celebdf_graph.main(["--list", str(path), "--json", str(out), "--seeds", "5"]) == 0
    report = json.loads(out.read_text())
    assert report["videos"] == len(fixture_videos()) == 10
    assert report["videos_by_label"] == {"real": 5, "fake": 5}
    assert report["components_total"] == 5 and report["identities"] == 10
    assert report["largest_component"]["videos"] == 2
    assert "nothing was dropped" in capsys.readouterr().out
    # Reading the list is the only file access: nothing else appears next to it.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "List_of_testing_videos.txt",
        "report.json",
    ]


def test_cli_rejects_a_malformed_list(tmp_path, capsys):
    path = tmp_path / "bad.txt"
    path.write_text("0 Celeb-real/id0_0000.mp4\n")
    assert celebdf_graph.main(["--list", str(path)]) == 2
    assert "contradicts" in capsys.readouterr().err
