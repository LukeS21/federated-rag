"""Unit tests for gap resolver parsing."""
import pytest
from src.agents.gap_resolver import _parse_gaps_to_queries, _is_false_positive_gap


class TestIsFalsePositiveGap:
    def test_null_finding_no_significant_difference(self):
        assert _is_false_positive_gap(
            "The results showed no significant difference between groups."
        )

    def test_null_finding_no_statistically_significant(self):
        assert _is_false_positive_gap(
            "No statistically significant change was detected."
        )

    def test_null_finding_no_effect_observed(self):
        assert _is_false_positive_gap(
            "No effect was observed at the 24-hour time point."
        )

    def test_null_finding_no_difference_observed(self):
        assert _is_false_positive_gap(
            "No difference was observed between treatment and control."
        )

    def test_real_gap_not_false_positive(self):
        assert not _is_false_positive_gap(
            "No osteoblast data exists in obese Ti models."
        )

    def test_real_gap_missing_data(self):
        assert not _is_false_positive_gap(
            "Missing data on macrophage polarization kinetics."
        )


class TestParseGapsToQueries:
    def test_numbered_gaps(self):
        text = (
            "1. No osteoblast data in obese Ti models.\n\n"
            "2. Missing IL-17A role in bone formation.\n\n"
            "3. Insufficient data on macrophage polarization."
        )
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 3

    def test_filters_null_finding(self):
        text = (
            "1. No osteoblast data in obese models.\n\n"
            "No significant difference was observed."
        )
        queries = _parse_gaps_to_queries(text)
        # Should only get the first one; the null finding is not a separate numbered item
        # so it becomes part of block 1's context, filtered by false-positive check on title
        assert len(queries) == 1

    def test_empty_text(self):
        assert _parse_gaps_to_queries("") == []
        assert _parse_gaps_to_queries("   ") == []

    def test_no_gap_keywords(self):
        text = "Titanium implants show good osseointegration in murine models."
        assert _parse_gaps_to_queries(text) == []

    def test_multiple_middle_words_in_no_data_pattern(self):
        """'no osteoblast activity data' should match (multiple words between no and data)."""
        text = "1. no osteoblast activity data exists in obese models."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1

    def test_simple_no_data(self):
        text = "1. no data on cytokine kinetics in diabetic models."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1

    def test_hyphen_in_compound_words_not_split(self):
        """Hyphens in IL-17A, Ti-6Al-4V should not be treated as bullet markers."""
        text = "IL-17A levels were measured. Ti-6Al-4V showed osseointegration. Gap: no data on kinetics."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) >= 1
        # The query should contain the gap content, not split fragments
        query_text = " ".join(q["query"] for q in queries)
        assert "kinetics" in query_text.lower()

    def test_gap_prefix_stripping(self):
        text = "Gap: No study has examined the role of leptin in peri-implant bone remodeling."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1
        assert "No study has" not in queries[0]["query"]

    def test_insufficient_data_keyword(self):
        text = "Insufficient data on macrophage polarization kinetics in diabetic models."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1

    def test_unknown_keyword(self):
        text = "The role of IL-17A in peri-implant bone formation remains unknown."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1

    def test_unexplored_keyword(self):
        text = "The interaction between surface roughness and immune response is unexplored."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1

    def test_short_text_skipped(self):
        text = "1. No data."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 0  # too short (< 15 chars)

    def test_context_preserved(self):
        text = "1. Missing data on macrophage polarization. Most studies focus on neutrophils."
        queries = _parse_gaps_to_queries(text)
        assert len(queries) == 1
        assert "neutrophils" in queries[0]["context"]
