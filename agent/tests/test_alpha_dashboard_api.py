from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.alpha_routes import BenchRequest, _result_for_wire


def test_bench_request_accepts_and_deduplicates_optional_alpha_ids() -> None:
    request = BenchRequest(
        zoo="alpha101",
        universe="sp500",
        period="2020-2025",
        alpha_ids=["alpha101_001", "alpha101_001", "alpha101_002"],
    )
    assert request.alpha_ids == ["alpha101_001", "alpha101_002"]


def test_bench_request_rejects_malformed_alpha_id() -> None:
    with pytest.raises(ValidationError):
        BenchRequest(
            zoo="alpha101",
            universe="sp500",
            period="2020-2025",
            alpha_ids=["../../not-an-alpha"],
        )


def test_result_for_wire_exposes_only_dashboard_safe_row_fields() -> None:
    wire = _result_for_wire(
        {
            "alive": 1,
            "reversed": 0,
            "dead": 0,
            "n_alphas_tested": 1,
            "n_skipped": 0,
            "rows": [
                {
                    "id": "alpha101_001",
                    "ic_mean": 0.04,
                    "ic_std": 0.08,
                    "ir": 0.5,
                    "ic_positive_ratio": 0.61,
                    "ic_count": 242,
                    "theme": ["momentum"],
                    "formula_latex": "secret-not-needed-by-dashboard",
                    "_category": "alive",
                }
            ],
        }
    )

    assert wire["skipped"] == 0
    assert wire["rows"] == [
        {
            "id": "alpha101_001",
            "ic_mean": 0.04,
            "ic_std": 0.08,
            "ir": 0.5,
            "ic_positive_ratio": 0.61,
            "ic_count": 242,
            "theme": ["momentum"],
            "category": "alive",
        }
    ]
