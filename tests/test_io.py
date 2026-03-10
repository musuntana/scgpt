from __future__ import annotations

from src.data.io import compute_file_md5, file_matches_md5


def test_compute_file_md5_and_match(tmp_path):
    sample_path = tmp_path / "sample.txt"
    sample_path.write_text("perturbscope-gpt\n", encoding="utf-8")

    expected_md5 = compute_file_md5(sample_path)

    assert expected_md5
    assert file_matches_md5(sample_path, expected_md5)
    assert not file_matches_md5(sample_path, "deadbeef")
