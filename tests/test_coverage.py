"""Unit tests for coverage diagnostic matching logic."""
import pytest
from src.retrieval.coverage import (
    _match_pmcids_to_s2_results,
    _normalize_for_match,
    _titles_overlap,
)


class TestNormalizeForMatch:
    def test_lowercase_and_strip_punctuation(self):
        result = _normalize_for_match("Titanium Implant, Macrophage Polarization!")
        assert result == "titanium implant macrophage polarization"

    def test_collapse_whitespace(self):
        result = _normalize_for_match("titanium   implant    macrophage")
        assert result == "titanium implant macrophage"

    def test_empty_string(self):
        assert _normalize_for_match("") == ""
        assert _normalize_for_match("  !@#$  ") == ""


class TestTitlesOverlap:
    def test_exact_match(self):
        score = _titles_overlap(
            "Titanium implant macrophage polarization",
            "Titanium implant macrophage polarization",
        )
        assert score >= 0.95

    def test_close_match(self):
        score = _titles_overlap(
            "Titanium implant macrophage polarization in obese mice",
            "Macrophage polarization on titanium implants in obesity",
        )
        assert score > 0.3  # partial overlap

    def test_no_overlap(self):
        score = _titles_overlap(
            "Titanium implant macrophage polarization",
            "Zebra fish embryogenesis development",
        )
        assert score < 0.3

    def test_empty_titles(self):
        assert _titles_overlap("", "something") == 0.0
        assert _titles_overlap("something", "") == 0.0
        assert _titles_overlap("", "") == 0.0


class TestMatchPmcids:
    def test_exact_doi_match(self):
        s2_results = [
            {"doi": "10.1016/j.test.2021.01", "title": "Test paper A"},
        ]
        epmc_results = [
            {"doi": "10.1016/j.test.2021.01", "pmcid": "PMC12345", "title": "Test paper A"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is True
        assert s2_results[0]["matched_pmcid"] == "PMC12345"
        assert s2_results[0]["match_method"] == "doi_exact"

    def test_doi_with_url_prefix(self):
        s2_results = [
            {"doi": "https://doi.org/10.1016/j.test.2021.01", "title": "Test B"},
        ]
        epmc_results = [
            {"doi": "10.1016/j.test.2021.01", "pmcid": "PMC67890", "title": "Test B"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is True
        assert s2_results[0]["matched_pmcid"] == "PMC67890"
        assert s2_results[0]["match_method"] == "doi_clean"

    def test_title_fuzzy_match(self):
        s2_results = [
            {"doi": "", "title": "Macrophage polarization on titanium implants in obesity models"},
        ]
        epmc_results = [
            {"doi": "10.1016/other.2021", "pmcid": "PMC99999",
             "title": "Macrophage polarization on titanium implants in obese mice"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is True
        assert s2_results[0]["match_method"].startswith("title_fuzzy")

    def test_no_match(self):
        s2_results = [
            {"doi": "10.1016/no.match.001", "title": "Unique zebra fish paper no one cited"},
        ]
        epmc_results = [
            {"doi": "10.1016/something.else", "pmcid": "PMC001",
             "title": "Completely different research area"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is False
        assert s2_results[0]["match_method"] == ""

    def test_case_insensitive_doi(self):
        s2_results = [
            {"doi": "10.1016/J.TEST.2021.01", "title": "Case test"},
        ]
        epmc_results = [
            {"doi": "10.1016/j.test.2021.01", "pmcid": "PMC_CASE", "title": "Case test"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is True

    def test_multiple_s2_multiple_epmc(self):
        s2_results = [
            {"doi": "10.1000/match.one", "title": "Match one"},
            {"doi": "10.1000/match.two", "title": "Match two"},
            {"doi": "10.1000/no.match", "title": "No match"},
        ]
        epmc_results = [
            {"doi": "10.1000/match.one", "pmcid": "PMC1", "title": "Match one"},
            {"doi": "10.1000/match.two", "pmcid": "PMC2", "title": "Match two"},
        ]
        _match_pmcids_to_s2_results(s2_results, epmc_results)
        assert s2_results[0]["in_pmc"] is True
        assert s2_results[1]["in_pmc"] is True
        assert s2_results[2]["in_pmc"] is False
