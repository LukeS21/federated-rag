"""
Claim/citation ledger for cross-section tracking in multi-turn section writing.

Tracks every claim→citation mapping across sections.  Prevents duplicate
claims, ensures citation coverage, and flags ungrounded assertions during
multi-section writing.

Usage::

    ledger = ClaimLedger()
    ledger.add_claim(
        claim_text="IL-6 is elevated in obese mice post-implantation (@avery2024)",
        section="Results",
        citations=["avery2024"],
    )
    dupes = ledger.find_duplicates("IL-6 is elevated...")
    coverage = ledger.coverage_report()
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"@[\w-]+")


class ClaimLedger:
    """Tracks claims, citations, and sections during multi-turn synthesis.

    Each claim is assigned a stable ID (SHA-256 of normalized text) for
    deduplication.  Citations are parsed from ``@author2025`` inline keys.
    """

    def __init__(self, ledger_path: str | Path | None = None):
        """
        Args:
            ledger_path: Path to persist the ledger to disk (JSON).
        """
        self.claims: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.ledger_path = Path(ledger_path) if ledger_path else None
        if self.ledger_path and self.ledger_path.exists():
            self.load()

    def _claim_id(self, text: str) -> str:
        """Generate a stable, deterministic claim ID from normalized text."""
        normalized = " ".join(scrub_unicode(text).lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _extract_citations(self, text: str) -> List[str]:
        """Extract @citation keys from claim text."""
        return sorted(set(
            c.lstrip("@") for c in CITATION_PATTERN.findall(text)
        ))

    def add_claim(
        self,
        claim_text: str,
        section: str,
        citations: List[str] | None = None,
        grounded: bool = True,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Add a claim to the ledger.

        Args:
            claim_text: The claim text with inline @citations.
            section: Section name (Introduction, Methods, Results, Discussion).
            citations: Explicit citation list. If None, parsed from text.
            grounded: Whether the claim is evidence-grounded.
            metadata: Arbitrary key-value metadata.

        Returns:
            The claim record added to the ledger.
        """
        if citations is None:
            citations = self._extract_citations(claim_text)

        claim_id = self._claim_id(claim_text)
        record = {
            "claim_id": claim_id,
            "claim_text": scrub_unicode(claim_text),
            "section": section,
            "citations": sorted(set(citations)),
            "grounded": grounded,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.claims.append(record)
        self._by_id[claim_id] = record
        logger.debug("Ledger: added claim %s [section=%s, %d citations]",
                      claim_id, section, len(citations))
        return record

    def add_claims(
        self,
        claim_texts: List[str],
        section: str,
        grounded: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch-add claims from a list of text strings."""
        return [self.add_claim(t, section, grounded=grounded) for t in claim_texts]

    def find_duplicates(self, claim_text: str) -> List[Dict[str, Any]]:
        """Find existing claims with the same stable ID. O(1) lookup."""
        cid = self._claim_id(claim_text)
        claim = self._by_id.get(cid)
        return [claim] if claim else []

    def is_duplicate(self, claim_text: str) -> bool:
        """Return True if this claim text already exists in the ledger. O(1)."""
        cid = self._claim_id(claim_text)
        return cid in self._by_id

    def filter_new_claims(self, claim_texts: List[str]) -> List[str]:
        """Return only claim texts that are not yet in the ledger."""
        return [t for t in claim_texts if not self.is_duplicate(t)]

    def get_claims_by_section(self, section: str) -> List[Dict[str, Any]]:
        """Return all claims for a given section."""
        return [c for c in self.claims if c["section"] == section]

    def get_used_citations(self) -> Set[str]:
        """Return the set of all citation keys used across all claims."""
        used: Set[str] = set()
        for c in self.claims:
            used.update(c.get("citations", []))
        return used

    def get_ungrounded_claims(self) -> List[Dict[str, Any]]:
        """Return all claims flagged as ungrounded."""
        return [c for c in self.claims if not c.get("grounded", True)]

    def coverage_report(self, available_citations: Set[str] | None = None) -> Dict[str, Any]:
        """Generate a citation coverage report.

        Args:
            available_citations: Set of all citation keys in the corpus.
                                 If None, only counts used citations.

        Returns:
            Dict with: total_claims, used_citations, coverage_rate,
                       ungrounded_count, duplicate_count.
        """
        used = self.get_used_citations()
        total_claims = len(self.claims)
        unique_claim_ids = len({c["claim_id"] for c in self.claims})
        duplicate_count = total_claims - unique_claim_ids
        ungrounded = self.get_ungrounded_claims()

        report = {
            "total_claims": total_claims,
            "unique_claims": unique_claim_ids,
            "duplicate_count": duplicate_count,
            "unique_citations_used": len(used),
            "citation_keys_used": sorted(used),
            "ungrounded_count": len(ungrounded),
        }

        if available_citations:
            unused = available_citations - used
            report["available_citations"] = len(available_citations)
            report["unused_citations"] = len(unused)
            report["coverage_rate"] = round(
                len(used) / max(len(available_citations), 1), 3,
            )
            report["unused_citation_keys"] = sorted(unused)
        else:
            report["coverage_rate"] = None

        # Per-section breakdown
        sections: Dict[str, Dict] = {}
        for c in self.claims:
            sec = c["section"]
            if sec not in sections:
                sections[sec] = {"claims": 0, "citations": set(), "ungrounded": 0}
            sections[sec]["claims"] += 1
            sections[sec]["citations"].update(c.get("citations", []))
            if not c.get("grounded", True):
                sections[sec]["ungrounded"] += 1

        report["per_section"] = {
            sec: {
                "claims": d["claims"],
                "unique_citations": len(d["citations"]),
                "ungrounded": d["ungrounded"],
            }
            for sec, d in sections.items()
        }

        return report

    def validate_section(self, section: str, min_citations: int = 1) -> List[str]:
        """Validate a section's claims and return warnings.

        Checks:
          - Minimum citation count per claim
          - Ungrounded claims
          - Duplicate claims within the section
        """
        warnings: List[str] = []
        section_claims = self.get_claims_by_section(section)

        for c in section_claims:
            if len(c.get("citations", [])) < min_citations:
                warnings.append(
                    f"Claim '{c['claim_text'][:80]}...' has fewer than "
                    f"{min_citations} citation(s)"
                )
            if not c.get("grounded", True):
                warnings.append(
                    f"Claim '{c['claim_text'][:80]}...' is flagged as ungrounded"
                )

        # Duplicate detection within section
        seen_ids: Set[str] = set()
        for c in section_claims:
            if c["claim_id"] in seen_ids:
                warnings.append(
                    f"Duplicate claim: '{c['claim_text'][:80]}...'"
                )
            seen_ids.add(c["claim_id"])

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the ledger to a plain dict."""
        return {
            "claims": self.claims,
            "updated_at": time.time(),
        }

    def save(self, path: str | Path | None = None) -> None:
        """Persist ledger to disk as JSON."""
        dest = Path(path) if path else self.ledger_path
        if dest is None:
            raise ValueError("No ledger_path set — provide a path argument or set it at init")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Ledger saved: %d claims to %s", len(self.claims), dest)

    def load(self, path: str | Path | None = None) -> "ClaimLedger":
        """Load ledger from disk."""
        src = Path(path) if path else self.ledger_path
        if src is None:
            raise ValueError("No ledger_path set")
        if not src.exists():
            logger.warning("Ledger file not found: %s — starting fresh", src)
            return self

        data = json.loads(src.read_text(encoding="utf-8"))
        self.claims = data.get("claims", [])
        self._by_id = {c["claim_id"]: c for c in self.claims}
        logger.info("Ledger loaded: %d claims from %s", len(self.claims), src)
        return self

    def clear(self) -> None:
        """Reset the ledger to empty."""
        self.claims = []
        self._by_id = {}

    def __len__(self) -> int:
        return len(self.claims)
