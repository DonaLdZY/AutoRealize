from pathlib import Path

from autorealize.models import FileSamplingReviewItem
from autorealize.modules.data_cognition import _apply_sampling_review


def _make_candidate(root: Path) -> dict:
    files = []
    for name in [
        "carrier_01_cost.xlsx",
        "carrier_02_cost.xlsx",
        "carrier_03_cost.xlsx",
        "carrier_99_cost.xlsx",
    ]:
        path = root / name
        path.write_text("stub", encoding="utf-8")
        files.append(path)

    samples = files[:2]
    skipped = files[2:]
    return {
        "pattern_id": "pat_test",
        "directory": ".",
        "pattern": "{id}__cost.xlsx",
        "total": len(files),
        "sampled": [p.name for p in samples],
        "skipped": [p.name for p in skipped],
        "_files": files,
        "_sample_paths": samples,
        "_skipped_paths": skipped,
    }


def test_sampling_review_can_add_extra_representative_file(tmp_path: Path) -> None:
    candidate = _make_candidate(tmp_path)
    review = FileSamplingReviewItem(
        pattern_id="pat_test",
        accept_sampling=True,
        extra_sample_files=["carrier_99_cost.xlsx"],
        reason="Read a tail carrier id as a boundary representative.",
    )

    out = _apply_sampling_review(candidate, review, tmp_path)

    assert out["review"]["decision"] == "accept_sampling_with_extra_files"
    assert out["sampled"] == [
        "carrier_01_cost.xlsx",
        "carrier_02_cost.xlsx",
        "carrier_99_cost.xlsx",
    ]
    assert out["skipped"] == ["carrier_03_cost.xlsx"]


def test_sampling_review_can_force_full_read(tmp_path: Path) -> None:
    candidate = _make_candidate(tmp_path)
    review = FileSamplingReviewItem(
        pattern_id="pat_test",
        accept_sampling=False,
        force_full_read=True,
        reason="Names may hide distinct business meanings.",
    )

    out = _apply_sampling_review(candidate, review, tmp_path)

    assert out["review"]["decision"] == "force_full_read"
    assert out["sampled"] == [
        "carrier_01_cost.xlsx",
        "carrier_02_cost.xlsx",
        "carrier_03_cost.xlsx",
        "carrier_99_cost.xlsx",
    ]
    assert out["skipped"] == []


def test_missing_sampling_review_falls_back_to_full_read(tmp_path: Path) -> None:
    candidate = _make_candidate(tmp_path)

    out = _apply_sampling_review(candidate, None, tmp_path)

    assert out["review"]["decision"] == "force_full_read_missing_review"
    assert out["sampled"] == [
        "carrier_01_cost.xlsx",
        "carrier_02_cost.xlsx",
        "carrier_03_cost.xlsx",
        "carrier_99_cost.xlsx",
    ]
    assert out["skipped"] == []
