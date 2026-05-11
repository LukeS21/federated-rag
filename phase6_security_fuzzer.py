#!/usr/bin/env python3
"""Phase 6 Security Scrubber Fuzzer — GLiNER-PII & BoundaryScrubber stress test.

Generates 1000+ randomized PHI-like strings across all detection categories
(person names, SSNs, emails, phone numbers, MRNs, API keys, IPs, dates,
medical conditions, etc.) and measures:

  * Regex Layer 1 detection rate (BoundaryScrubber patterns)
  * GLiNER-PII Layer 2 detection rate (context-dependent AI model)
  * False-positive rate on clean biomedical text
  * Overlap between regex and GLiNER detections
  * Performance (mean latency per sample)

GLiNER-PII testing is optional — skipped when model unavailable or
``GLINER_PRIVACY_ENABLED=0``.  Outputs a JSON scorecard and prints a
colour-coded report.

Usage:
    python phase6_security_fuzzer.py                        # Full fuzz run
    python phase6_security_fuzzer.py --samples 200           # Smaller run
    python phase6_security_fuzzer.py --no-gliner             # Regex only
    python -m pytest phase6_security_fuzzer.py -v            # Regression guard
"""

from __future__ import annotations

import json
import os
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Category definitions for fuzzing ──────────────────────────────────────
# Each category has: name, generator functions that produce positive examples,
# and the expected redaction tag.

CATEGORIES: Dict[str, Dict[str, Any]] = {
    "ssn": {
        "tag": "[REDACTED-SSN]",
        "generators": [
            lambda: f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(100, 999)}{random.randint(10, 99)}{random.randint(1000, 9999)}",
            lambda: f"SSN {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        ],
    },
    "email": {
        "tag": "[REDACTED-EMAIL]",
        "generators": [
            lambda: f"{_random_word(6)}.{_random_word(5)}@{_random_word(6)}.{_random_tld()}",
            lambda: f"{_random_word(8)}@{_random_word(5)}.org",
            lambda: f"contact {_random_word(5)}@{_random_word(7)}.edu for details",
        ],
    },
    "phone": {
        "tag": "[REDACTED-PHONE]",
        "generators": [
            lambda: f"({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            lambda: f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            lambda: f"Call {random.randint(200,999)}.{random.randint(100,999)}.{random.randint(1000,9999)} for appointment",
        ],
    },
    "mrn": {
        "tag": "[REDACTED-MRN]",
        "generators": [
            lambda: f"MRN MR#{random.randint(100000, 999999999999)}",
            lambda: f"MR #{random.randint(100000, 999999)}",
            lambda: f"MR#{random.randint(100000, 999999999999)} admitted on",
        ],
    },
    "api_key": {
        "tag": "[REDACTED-KEY]",
        "generators": [
            lambda: f"sk-{_random_alnum(48)}",
            lambda: f"pk-{_random_alnum(32)}",
            lambda: f"api-{_random_alnum(40)}",
        ],
    },
    "grant": {
        "tag": "[REDACTED-GRANT]",
        "generators": [
            lambda: f"R{random.randint(0,9)}{random.randint(0,9)}HL{random.randint(100000, 999999)}",
            lambda: f"K23CA{random.randint(100000, 999999)}",
            lambda: f"U01DK{random.randint(100000, 999999)}",
            lambda: f"Grant #NIH-{random.randint(2020,2026)}-{_random_alnum(4).upper()}",
        ],
    },
    "ip": {
        "tag": "[REDACTED-IP]",
        "generators": [
            lambda: f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
            lambda: f"Server at {random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
        ],
    },
    "dob": {
        "tag": "[REDACTED-DOB]",
        "generators": [
            lambda: f"DOB: {random.randint(1,12)}/{random.randint(1,28)}/{random.randint(1940,2010)}",
            lambda: f"DOB {random.randint(1950,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        ],
    },
    "hospital": {
        "tag": "[REDACTED-FACILITY]",
        "generators": [
            lambda: f"{_random_name()} Hospital",
            lambda: f"{_random_name()} Medical Center",
            lambda: f"{_random_name()} Clinic",
        ],
    },
    "project_code": {
        "tag": "[REDACTED-PROJECT]",
        "generators": [
            lambda: f"PROJ-{random.randint(2020,2026)}-{random.randint(1,999):03d}",
            lambda: f"PROJ {random.randint(2020,2026)} {random.randint(1,999):03d}",
        ],
    },
}

# Clean biomedical text snippets for false-positive testing
_CLEAN_SNIPPETS: List[str] = [
    "The study evaluated 150 patients with type 2 diabetes in a randomized controlled trial.",
    "TNF-alpha and IL-6 levels were measured by ELISA at baseline and week 12.",
    "Bone marrow-derived macrophages were cultured in RPMI 1640 with 10% FBS.",
    "Surface roughness (Ra) was measured by atomic force microscopy on Ti-6Al-4V disks.",
    "C57BL/6J mice were fed a high-fat diet (45 kcal% fat) for 12 weeks.",
    "Flow cytometry analysis revealed increased CD4+ T cell infiltration at day 7.",
    "Histomorphometric analysis showed 23.4% bone-to-implant contact in the test group.",
    "The p-value was 0.003 with 95% confidence interval for the primary endpoint.",
    "MicroCT scanning was performed at 10 um resolution using a Scanco Medical system.",
    "Leptin receptor expression was quantified by qRT-PCR and normalized to GAPDH.",
    "Neutrophil elastase activity was significantly elevated in obese mice post-implantation.",
    "Osteocalcin and CTX-1 were assessed as markers of bone formation and resorption.",
    "The implant surface was characterized by XPS, SEM, and contact angle goniometry.",
    "Statistical analysis was performed using GraphPad Prism 9 with two-way ANOVA.",
    "All procedures were approved by the Institutional Animal Care and Use Committee.",
    "Cell viability exceeded 90% in all groups as measured by MTT assay at 24 and 72 hours.",
    "The rough-hydrophilic surface showed increased protein adsorption compared to machined controls.",
    "Immunohistochemistry revealed elevated F4/80+ macrophage infiltration at the implant site.",
    "Obese mice exhibited significantly elevated serum leptin and C-reactive protein levels.",
    "RNA sequencing was performed on day 7 peri-implant tissue with 30M reads per sample.",
]

# Person names for GLiNER testing (regex cannot catch these in free text)
_PERSON_NAMES: List[str] = [
    "John Smith", "Maria Garcia", "Wei Zhang", "Sarah Johnson",
    "David Chen", "Emily Brown", "Michael Wilson", "Lisa Anderson",
    "Robert Thompson", "Jennifer Martinez", "James Davis", "Patricia Lee",
    "Thomas White", "Barbara Harris", "Daniel Clark", "Elizabeth Moore",
    "Matthew Taylor", "Susan Jackson", "Andrew Martin", "Karen Rodriguez",
]

# Medical conditions for GLiNER testing
_MEDICAL_CONDITIONS: List[str] = [
    "type 2 diabetes", "hypertension", "coronary artery disease",
    "breast cancer", "Alzheimer's disease", "rheumatoid arthritis",
    "chronic kidney disease", "COPD", "multiple sclerosis",
    "Parkinson's disease",
]

# GLiNER-taggable organizations
_ORGANIZATIONS: List[str] = [
    "Mayo Clinic", "Johns Hopkins", "Massachusetts General",
    "Cleveland Clinic", "Stanford Medical Center", "UCLA Health",
    "MD Anderson", "Mount Sinai Hospital",
]

# Dates for GLiNER testing
_DATES: List[str] = [
    "January 15 2024", "March 3 2023", "12/05/2022",
    "2021-07-19", "October 2020", "June 10 2019",
]


def _random_word(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def _random_alnum(length: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _random_name() -> str:
    first = random.choice(["North", "Central", "St", "Mercy", "General", "Community",
                            "University", "Regional", "County", "Memorial"])
    return first


def _random_tld() -> str:
    return random.choice(["com", "org", "edu", "net", "gov"])


def _embed_phi(text: str, phi: str) -> str:
    """Embed a PHI segment within biomedical text for context-dependent testing."""
    templates = [
        f"Patient {phi} was enrolled in the study after informed consent.",
        f"Data from {phi} showed elevated inflammatory markers.",
        f"Samples were processed at {phi} according to standard protocols.",
        f"The subject {phi} completed all follow-up visits.",
        f"Records for {phi} were de-identified before analysis.",
        f"Contact {phi} for additional clinical data.",
        f"Specimens from {phi} underwent histopathological examination.",
        f"Funding from {phi} supported the biomarker analysis.",
    ]
    return random.choice(templates)


def _generate_clean_samples(n: int) -> List[Tuple[str, str]]:
    """Generate *n* clean biomedical text samples (negative examples).

    Returns list of ``(text, "clean")`` tuples.
    """
    samples: List[Tuple[str, str]] = []
    for _ in range(n):
        snippet = random.choice(_CLEAN_SNIPPETS)
        # Randomly add numbers and abbreviations that could confuse detectors
        variants = [
            snippet,
            f"{snippet} (n={random.randint(50, 500)})",
            f"p={random.uniform(0.001, 0.05):.3f}, {snippet}",
            snippet.replace(" mice", f" n={random.randint(8,15)}/group mice"),
        ]
        samples.append((random.choice(variants), "clean"))
    return samples


def _generate_positive_samples(n_per_category: int = 15) -> List[Tuple[str, str, str]]:
    """Generate positive samples (known PHI) for each detection category.

    Returns list of ``(text, category_name, expected_tag)`` tuples.
    """
    samples: List[Tuple[str, str, str]] = []
    for category_name, info in CATEGORIES.items():
        for _ in range(n_per_category):
            gen = random.choice(info["generators"])
            phi_text = gen()
            # 50% chance: embed in biomedical context for realism
            if random.random() < 0.5:
                text = _embed_phi(phi_text, phi_text)
            else:
                text = phi_text
            samples.append((text, category_name, info["tag"]))
    return samples


def _generate_gliner_positives(n: int) -> List[Tuple[str, str]]:
    """Generate *n* positive samples for GLiNER-only categories.

    These are context-dependent PII that regex patterns cannot reliably catch:
    person names in free text, organizations, dates in prose, and medical
    conditions mentioned without MRN/SSN formatting.

    Returns list of ``(text, category_label)`` tuples.
    """
    samples: List[Tuple[str, str]] = []
    categories = {
        "PERSON": _PERSON_NAMES,
        "MEDICAL_CONDITION": _MEDICAL_CONDITIONS,
        "ORGANIZATION": _ORGANIZATIONS,
        "DATE": _DATES,
    }
    for label, items in categories.items():
        for _ in range(max(n // len(categories), 5)):
            item = random.choice(items)
            templates = [
                f"Patient {item} was admitted with acute symptoms.",
                f"The study by {item} et al. demonstrated significant results.",
                f"Treatment for {item} was initiated per protocol.",
                f"Data were collected at {item} between 2022 and 2024.",
                f"Samples from patient {item} showed elevated CRP levels.",
                f"On {item}, the subject presented for follow-up.",
            ]
            text = random.choice(templates).replace("{}", item).replace("{item}", item, 1)
            if item in text or any(w in text for w in item.split()):
                pass  # item embedded correctly
            else:
                text = f"Subject {item} was included in the analysis."
            samples.append((text, label))
    return samples


# ── Scoring & reporting ───────────────────────────────────────────────────

@dataclass
class FuzzResult:
    """Results for a single fuzz sample."""
    text: str
    expected_category: str
    expected_tag: str
    regex_detected: bool = False
    regex_tag: str = ""
    gliner_detected: bool = False
    gliner_category: str = ""
    gliner_elapsed_ms: float = 0.0


@dataclass
class FuzzScorecard:
    """Aggregated fuzzer results."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_positive_samples: int = 0
    total_negative_samples: int = 0

    # Regex Layer 1
    regex_true_positives: int = 0
    regex_false_negatives: int = 0
    regex_false_positives: int = 0
    regex_true_negatives: int = 0
    regex_detection_rate: float = 0.0
    regex_false_positive_rate: float = 0.0

    # GLiNER Layer 2
    gliner_available: bool = False
    gliner_true_positives: int = 0
    gliner_false_negatives: int = 0
    gliner_false_positives: int = 0
    gliner_true_negatives: int = 0
    gliner_detection_rate: float = 0.0
    gliner_false_positive_rate: float = 0.0
    gliner_mean_latency_ms: float = 0.0
    gliner_max_latency_ms: float = 0.0

    # Per-category breakdown
    category_details: Dict[str, Any] = field(default_factory=dict)

    # Overlap
    regex_only_count: int = 0
    gliner_only_count: int = 0
    both_count: int = 0
    neither_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_positive_samples": self.total_positive_samples,
            "total_negative_samples": self.total_negative_samples,
            "regex": {
                "detection_rate": round(self.regex_detection_rate, 4),
                "false_positive_rate": round(self.regex_false_positive_rate, 4),
                "true_positives": self.regex_true_positives,
                "false_negatives": self.regex_false_negatives,
                "false_positives": self.regex_false_positives,
                "true_negatives": self.regex_true_negatives,
            },
            "gliner": {
                "available": self.gliner_available,
                "detection_rate": round(self.gliner_detection_rate, 4),
                "false_positive_rate": round(self.gliner_false_positive_rate, 4),
                "true_positives": self.gliner_true_positives,
                "false_negatives": self.gliner_false_negatives,
                "false_positives": self.gliner_false_positives,
                "true_negatives": self.gliner_true_negatives,
                "mean_latency_ms": round(self.gliner_mean_latency_ms, 2),
                "max_latency_ms": round(self.gliner_max_latency_ms, 2),
            },
            "overlap": {
                "regex_only": self.regex_only_count,
                "gliner_only": self.gliner_only_count,
                "both_detected": self.both_count,
                "neither_detected": self.neither_count,
            },
            "category_details": self.category_details,
        }


def _colour(label: str, condition: str) -> str:
    """ANSI colour codes for terminal output."""
    codes = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "INFO": "\033[96m"}
    reset = "\033[0m"
    return f"{codes.get(condition, '')}{label}{reset}"


def _grade(value: float, pass_thresh: float = 0.90, warn_thresh: float = 0.70) -> str:
    if value >= pass_thresh:
        return "PASS"
    elif value >= warn_thresh:
        return "WARN"
    return "FAIL"


def _grade_negative(value: float, fail_thresh: float = 0.05, warn_thresh: float = 0.10) -> str:
    """For false positive rate — lower is better."""
    if value <= fail_thresh:
        return "PASS"
    elif value <= warn_thresh:
        return "WARN"
    return "FAIL"


# ── Main fuzz runner ──────────────────────────────────────────────────────

def _is_gliner_available() -> bool:
    """Check if GLiNER can be imported."""
    try:
        import gliner  # noqa: F401
        return True
    except ImportError:
        return False


def run_fuzzer(
    samples_per_category: int = 15,
    clean_samples: int = 100,
    gliner_positive_samples: int = 40,
    skip_gliner: bool = False,
) -> FuzzScorecard:
    """Run the full security fuzzer.

    Args:
        samples_per_category: Positive samples per regex category.
        clean_samples: Number of clean biomedical text samples.
        gliner_positive_samples: Positive samples for GLiNER-only categories.
        skip_gliner: Skip GLiNER testing even if available.

    Returns:
        Aggregated ``FuzzScorecard`` with all metrics.
    """
    scorecard = FuzzScorecard()

    # ── Generate test corpus ───────────────────────────────────────────
    regex_positives: List[FuzzResult] = []
    gliner_positives_raw: List[Tuple[str, str]] = []

    # Regex-catchable positives
    for text, cat, tag in _generate_positive_samples(samples_per_category):
        regex_positives.append(FuzzResult(text=text, expected_category=cat, expected_tag=tag))
    scorecard.total_positive_samples = len(regex_positives)

    # Clean negatives
    clean_texts = _generate_clean_samples(clean_samples)
    scorecard.total_negative_samples = len(clean_texts)

    # GLiNER-only positives (context-dependent)
    gliner_positives_raw = _generate_gliner_positives(gliner_positive_samples)

    # ── Layer 1: Regex BoundaryScrubber ─────────────────────────────────
    from src.security.boundary_scrubber import BoundaryScrubber

    scrubber = BoundaryScrubber(privacy_model=None)

    # Test positives
    for result in regex_positives:
        scrubbed = scrubber.scrub(result.text)
        result.regex_detected = result.expected_tag in scrubbed
        # Determine which tag(s) matched
        for cat_info in CATEGORIES.values():
            if cat_info["tag"] in scrubbed:
                result.regex_tag = cat_info["tag"]
                break

    scorecard.regex_true_positives = sum(1 for r in regex_positives if r.regex_detected)
    scorecard.regex_false_negatives = scorecard.total_positive_samples - scorecard.regex_true_positives
    scorecard.regex_detection_rate = (
        scorecard.regex_true_positives / scorecard.total_positive_samples
        if scorecard.total_positive_samples else 0.0
    )

    # Test negatives (false positives)
    for text, _label in clean_texts:
        before = text
        scrubbed = scrubber.scrub(text)
        if scrubbed != before:
            scorecard.regex_false_positives += 1
    scorecard.regex_true_negatives = clean_samples - scorecard.regex_false_positives
    scorecard.regex_false_positive_rate = (
        scorecard.regex_false_positives / clean_samples if clean_samples else 0.0
    )

    # Per-category breakdown for regex
    for cat in CATEGORIES:
        cat_samples = [r for r in regex_positives if r.expected_category == cat]
        if cat_samples:
            detected = sum(1 for r in cat_samples if r.regex_detected)
            scorecard.category_details[f"regex_{cat}"] = {
                "samples": len(cat_samples),
                "detected": detected,
                "rate": round(detected / len(cat_samples), 4),
            }

    # ── Layer 2: GLiNER-PII ─────────────────────────────────────────────
    gliner_available = _is_gliner_available() and not skip_gliner
    gliner_enabled = os.getenv("GLINER_PRIVACY_ENABLED", "1").strip().lower() in ("1", "true", "yes")

    if gliner_available and gliner_enabled:
        scorecard.gliner_available = True
        from src.security.gliner_privacy import GlinerPrivacyModel

        # We use the module-level singleton for timing — first call may be slow
        print("  Loading GLiNER-PII model (first call may download ~1 GB)...")
        gliner_model = GlinerPrivacyModel()
        latencies: List[float] = []

        # Test positives (GLiNER-specific context-dependent PHI)
        gliner_positives_matched = 0
        for text, expected_label in gliner_positives_raw:
            start = time.perf_counter()
            detections = gliner_model.detect(text)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            found = any(d[3] == expected_label for d in detections)
            if found:
                gliner_positives_matched += 1

        scorecard.gliner_true_positives = gliner_positives_matched
        scorecard.gliner_false_negatives = len(gliner_positives_raw) - gliner_positives_matched
        scorecard.gliner_detection_rate = (
            gliner_positives_matched / len(gliner_positives_raw)
            if gliner_positives_raw else 0.0
        )

        # Test negatives (false positives on clean text)
        for text, _label in clean_texts:
            start = time.perf_counter()
            detections = gliner_model.detect(text)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            if detections:
                scorecard.gliner_false_positives += 1
        scorecard.gliner_true_negatives = clean_samples - scorecard.gliner_false_positives
        scorecard.gliner_false_positive_rate = (
            scorecard.gliner_false_positives / clean_samples if clean_samples else 0.0
        )

        if latencies:
            scorecard.gliner_mean_latency_ms = sum(latencies) / len(latencies)
            scorecard.gliner_max_latency_ms = max(latencies)

        # Overlap analysis: test the same samples with both layers
        combined_samples = [(t, c, e) for t, c, e in _generate_positive_samples(10)]
        for text, cat, tag in combined_samples:
            regex_hit = tag in scrubber.scrub(text)
            gliner_hit = len(gliner_model.detect(text)) > 0

            if regex_hit and gliner_hit:
                scorecard.both_count += 1
            elif regex_hit and not gliner_hit:
                scorecard.regex_only_count += 1
            elif gliner_hit and not regex_hit:
                scorecard.gliner_only_count += 1
            else:
                scorecard.neither_count += 1

    return scorecard


def print_report(scorecard: FuzzScorecard) -> None:
    """Print a colour-coded fuzz test report."""
    print()
    print("=" * 70)
    print("  PHASE 6 SECURITY SCRUBBER FUZZER — RESULTS")
    print("=" * 70)
    print(f"  Timestamp:  {scorecard.timestamp}")
    print(f"  Samples:    {scorecard.total_positive_samples} positive  "
          f"|  {scorecard.total_negative_samples} negative (clean)")
    print()

    # Regex Layer 1
    print(f"  {_colour('REGEX LAYER 1 (BoundaryScrubber)', 'INFO')}")
    dr = scorecard.regex_detection_rate
    grade_dr = _grade(dr)
    print(f"    Detection rate:       {_colour(f'{dr:.2%}', grade_dr)}  "
          f"({scorecard.regex_true_positives}/{scorecard.total_positive_samples})")
    print(f"    False negatives:      {scorecard.regex_false_negatives}")

    fpr = scorecard.regex_false_positive_rate
    grade_fpr = _grade_negative(fpr, fail_thresh=0.03, warn_thresh=0.08)
    print(f"    False positive rate:  {_colour(f'{fpr:.2%}', grade_fpr)}  "
          f"({scorecard.regex_false_positives}/{scorecard.total_negative_samples})")

    # Per-category breakdown
    print(f"    Per-category detection:")
    for cat_name in sorted(CATEGORIES.keys()):
        key = f"regex_{cat_name}"
        if key in scorecard.category_details:
            d = scorecard.category_details[key]
            rate = d["rate"]
            g = _grade(rate, pass_thresh=0.90, warn_thresh=0.60)
            bar = "\u2588" * int(rate * 20) + "\u2591" * (20 - int(rate * 20))
            print(f"      {cat_name:>16s}  {_colour(f'{rate:.0%}', g)}  {bar}  "
                  f"({d['detected']}/{d['samples']})")

    print()

    # GLiNER Layer 2
    if scorecard.gliner_available:
        print(f"  {_colour('GLiNER-PII LAYER 2', 'INFO')}")
        dr = scorecard.gliner_detection_rate
        grade_dr = _grade(dr, pass_thresh=0.70, warn_thresh=0.50)
        print(f"    Detection rate:       {_colour(f'{dr:.2%}', grade_dr)}  "
              f"({scorecard.gliner_true_positives}/{scorecard.gliner_true_positives + scorecard.gliner_false_negatives})")
        print(f"    False negatives:      {scorecard.gliner_false_negatives}")

        fpr = scorecard.gliner_false_positive_rate
        grade_fpr = _grade_negative(fpr, fail_thresh=0.05, warn_thresh=0.12)
        print(f"    False positive rate:  {_colour(f'{fpr:.2%}', grade_fpr)}  "
              f"({scorecard.gliner_false_positives}/{scorecard.total_negative_samples})")
        print(f"    Mean latency:         {scorecard.gliner_mean_latency_ms:.1f} ms")
        print(f"    Max latency:          {scorecard.gliner_max_latency_ms:.1f} ms")
        print()
        print(f"  {_colour('OVERLAP ANALYSIS', 'INFO')}")
        total_overlap = (scorecard.both_count + scorecard.regex_only_count +
                         scorecard.gliner_only_count + scorecard.neither_count)
        if total_overlap:
            print(f"    Regex only:           {scorecard.regex_only_count} "
                  f"({scorecard.regex_only_count/total_overlap:.0%})")
            print(f"    GLiNER only:          {scorecard.gliner_only_count} "
                  f"({scorecard.gliner_only_count/total_overlap:.0%})")
            print(f"    Both detected:        {scorecard.both_count} "
                  f"({scorecard.both_count/total_overlap:.0%})")
            print(f"    Neither detected:     {scorecard.neither_count} "
                  f"({scorecard.neither_count/total_overlap:.0%})")
    else:
        print(f"  {_colour('GLiNER-PII LAYER 2: SKIPPED (model not available)', 'WARN')}")
        print(f"    Install: pip install gliner")
        print(f"    Or set GLINER_PRIVACY_ENABLED=1 with gliner installed.")

    print()
    print("=" * 70)


def save_scorecard(scorecard: FuzzScorecard, path: str = "projects/default/fuzzer_scorecard.json") -> None:
    """Save scorecard to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Scorecard saved to {p}")


# ── Pytest integration ───────────────────────────────────────────────────

def test_regex_ssn_detection() -> None:
    """All standard-format SSNs must be detected by regex layer."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "123-45-6789",
        "987654321",
        "SSN 111-22-3333",
        f"Patient SSN: {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-SSN]" in result



def test_regex_email_detection() -> None:
    """All standard-format emails must be detected by regex layer."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "john.doe@hospital.org",
        "contact research.lab@university.edu",
        "dr_smith@clinic.com",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-EMAIL]" in result


def test_regex_phone_detection() -> None:
    """US and international phone formats must be detected."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "(555) 123-4567",
        "555-123-4567",
        "+1-555-123-4567",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-PHONE]" in result


def test_regex_api_key_detection() -> None:
    """API key patterns (sk-, pk-, api-) must be detected."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    long_random = _random_alnum(32)
    patterns = [
        f"sk-{long_random}",
        f"pk-{long_random}",
        f"api-{long_random}",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-KEY]" in result


def test_regex_mrn_detection() -> None:
    """Medical record number formats must be detected."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "MRN MR#123456789",
        "MR #987654",
        "MR#123456789012 admitted",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-MRN]" in result


def test_regex_grant_detection() -> None:
    """NIH grant identifiers must be detected."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "R01HL123456",
        "K23CA987654",
        "U01DK555555",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-GRANT]" in result


def test_regex_ip_detection() -> None:
    """IPv4 addresses must be detected."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    patterns = [
        "192.168.1.1",
        "10.0.0.255",
        "Server at 172.16.0.100",
    ]
    for text in patterns:
        result = scrubber.scrub(text)
        assert "[REDACTED-IP]" in result


def test_clean_text_no_false_positive() -> None:
    """Biomedical text without PHI must not be redacted."""
    from src.security.boundary_scrubber import BoundaryScrubber
    scrubber = BoundaryScrubber()
    for text in _CLEAN_SNIPPETS:
        result = scrubber.scrub(text)
        assert result == text, f"False positive on: {text[:60]}..."


def test_fuzzer_regression() -> None:
    """Full fuzzer run — detection rate must be above thresholds."""
    scorecard = run_fuzzer(samples_per_category=5, clean_samples=20,
                           gliner_positive_samples=10, skip_gliner=True)
    # Regex must detect >= 85% of structured PHI
    assert scorecard.regex_detection_rate >= 0.85, (
        f"Regex detection rate {scorecard.regex_detection_rate:.2%} below 85% threshold"
    )
    # False positive rate must be <= 10% on clean biomedical text
    assert scorecard.regex_false_positive_rate <= 0.10, (
        f"Regex false positive rate {scorecard.regex_false_positive_rate:.2%} exceeds 10%"
    )


# ── CLI entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6 Security Scrubber Fuzzer")
    parser.add_argument("--samples", type=int, default=15,
                        help="Positive samples per regex category (default: 15)")
    parser.add_argument("--clean", type=int, default=100,
                        help="Clean biomedical text samples (default: 100)")
    parser.add_argument("--gliner-samples", type=int, default=40,
                        help="GLiNER-positive samples (default: 40)")
    parser.add_argument("--no-gliner", action="store_true",
                        help="Skip GLiNER testing even if model is available")
    parser.add_argument("--output", type=str, default="projects/default/fuzzer_scorecard.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("Phase 6 Security Scrubber Fuzzer")
    print(f"  Generating {args.samples} samples per category "
          f"({args.samples * len(CATEGORIES)} total positive)")
    print(f"  Generating {args.clean} clean biomedical samples")
    if not args.no_gliner:
        print(f"  Generating {args.gliner_samples} GLiNER-positive samples")

    scorecard = run_fuzzer(
        samples_per_category=args.samples,
        clean_samples=args.clean,
        gliner_positive_samples=args.gliner_samples,
        skip_gliner=args.no_gliner,
    )

    print_report(scorecard)
    save_scorecard(scorecard, path=args.output)
