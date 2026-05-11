"""
Unit tests for ClaimLedger — claim tracking, deduplication, coverage reporting.
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.synthesis.claim_ledger import ClaimLedger


def test_add_claim():
    """Adding a claim creates a record with required fields."""
    ledger = ClaimLedger()
    record = ledger.add_claim(
        "IL-6 increases in obese mice (@avery2024)",
        section="Results",
    )

    assert record["claim_id"]
    assert len(record["claim_id"]) == 16
    assert record["section"] == "Results"
    assert "avery2024" in record["citations"]
    assert record["grounded"] is True
    assert len(ledger) == 1


def test_add_claim_parses_citations():
    """Inline @citations are automatically extracted."""
    ledger = ClaimLedger()
    ledger.add_claim(
        "MSCs are recruited via SDF-1 (@avery2023) and proliferate in response to BMP-2 (@smith2025).",
        section="Introduction",
    )
    assert len(ledger) == 1
    c = ledger.claims[0]
    assert "avery2023" in c["citations"]
    assert "smith2025" in c["citations"]


def test_deduplication():
    """Identical claims produce the same ID and are detected as duplicates."""
    ledger = ClaimLedger()
    text = "CD4+ T cell deficiency reduces macrophage M1 polarization (@avery2024)"

    ledger.add_claim(text, section="Results")
    assert ledger.is_duplicate(text)
    assert len(ledger.find_duplicates(text)) == 1


def test_deduplication_normalizes():
    """Different whitespace/casing produces the same claim ID."""
    ledger = ClaimLedger()
    ledger.add_claim("  IL-6 IS ELEVATED   in obese mice.  ", section="Results")
    assert ledger.is_duplicate("IL-6 is elevated in obese mice.")
    assert ledger.is_duplicate("il-6 is elevated in obese mice.")


def test_filter_new_claims():
    """filter_new_claims returns only unseen claims."""
    ledger = ClaimLedger()
    claims = [
        "Claim A is new (@avery2024)",
        "Claim B is also new (@avery2024)",
    ]
    ledger.add_claim(claims[0], section="Results")

    new = ledger.filter_new_claims(claims)
    assert new == [claims[1]]


def test_get_used_citations():
    """get_used_citations returns all citation keys across all claims."""
    ledger = ClaimLedger()
    ledger.add_claim("Claim 1 (@avery2024)", section="Intro")
    ledger.add_claim("Claim 2 (@smith2025)", section="Results")
    ledger.add_claim("Claim 3 (@avery2024)", section="Discussion")

    used = ledger.get_used_citations()
    assert "avery2024" in used
    assert "smith2025" in used
    assert len(used) == 2


def test_coverage_report():
    """Coverage report includes per-section breakdown."""
    ledger = ClaimLedger()
    ledger.add_claim("Claim 1 (@avery2024)", section="Introduction")
    ledger.add_claim("Claim 2 (@smith2025)", section="Results")
    ledger.add_claim("Claim 3 (@smith2025)", section="Results")
    ledger.add_claim("Claim 4 (@jones2023)", section="Discussion")

    report = ledger.coverage_report(available_citations={"avery2024", "smith2025", "jones2023", "lee2022"})

    assert report["total_claims"] == 4
    assert report["unique_citations_used"] == 3
    assert report["coverage_rate"] == 0.75  # 3/4 used
    assert report["unused_citations"] == 1  # lee2022 unused
    assert report["per_section"]["Results"]["claims"] == 2


def test_coverage_report_no_available():
    """Coverage report without available citations still works."""
    ledger = ClaimLedger()
    ledger.add_claim("Claim 1 (@avery2024)", section="Results")
    report = ledger.coverage_report()
    assert report["coverage_rate"] is None


def test_validate_section():
    """Section validation catches ungrounded and duplicate claims."""
    ledger = ClaimLedger()
    ledger.add_claim("Ungrounded claim with no citation", section="Results", grounded=False, citations=[])
    ledger.add_claim("Good claim (@avery2024)", section="Results")
    # Add duplicate
    ledger.add_claim("Good claim (@avery2024)", section="Results")

    warnings = ledger.validate_section("Results")
    assert len(warnings) > 0
    assert any("ungrounded" in w.lower() for w in warnings)
    assert any("duplicate" in w.lower() for w in warnings)


def test_persistence(tmp_path):
    """Ledger saves and loads from JSON."""
    path = tmp_path / "ledger.json"

    ledger1 = ClaimLedger(ledger_path=path)
    ledger1.add_claim("Claim 1 (@avery2024)", section="Intro")
    ledger1.save()

    ledger2 = ClaimLedger(ledger_path=path)
    assert len(ledger2) == 1
    assert ledger2.claims[0]["claim_text"] == "Claim 1 (@avery2024)"


def test_clear():
    """clear() resets the ledger."""
    ledger = ClaimLedger()
    ledger.add_claim("Claim 1 (@avery2024)", section="Intro")
    assert len(ledger) == 1
    ledger.clear()
    assert len(ledger) == 0


def test_batch_add():
    """add_claims adds multiple claims at once."""
    ledger = ClaimLedger()
    texts = [
        "Claim A (@avery2024)",
        "Claim B (@smith2025)",
        "Claim C (@jones2023)",
    ]
    records = ledger.add_claims(texts, section="Results")
    assert len(records) == 3
    assert len(ledger) == 3
    assert all(r["section"] == "Results" for r in records)


def test_ungrounded_tracking():
    """Ungrounded claims are tracked separately."""
    ledger = ClaimLedger()
    ledger.add_claim("Grounded (@avery2024)", section="Results", grounded=True)
    ledger.add_claim("Ungrounded", section="Results", grounded=False, citations=[])

    ungrounded = ledger.get_ungrounded_claims()
    assert len(ungrounded) == 1
    assert ungrounded[0]["claim_text"] == "Ungrounded"


def test_get_claims_by_section():
    """Claims can be queried by section."""
    ledger = ClaimLedger()
    ledger.add_claim("Intro claim (@avery2024)", section="Introduction")
    ledger.add_claim("Result claim (@smith2025)", section="Results")

    intro = ledger.get_claims_by_section("Introduction")
    assert len(intro) == 1
    assert intro[0]["section"] == "Introduction"

    results = ledger.get_claims_by_section("Results")
    assert len(results) == 1

    empty = ledger.get_claims_by_section("Nonexistent")
    assert len(empty) == 0
