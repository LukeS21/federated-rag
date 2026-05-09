"""Security hardening modules for Phase 5.

Provides:
  - BoundaryScrubber: regex redaction at secure-public boundary
  - PrivacyModel: abstract interface for AI-based PII detection (Phase 6)
  - AuditLogger: timestamped security event logging
  - ScopeRouter: query_scope-based routing decisions for LangGraph
"""

from src.security.boundary_scrubber import BoundaryScrubber, default_boundary_scrubber
from src.security.audit_log import AuditLogger, get_audit_logger
from src.security.privacy_model import PrivacyModel, NoOpPrivacyModel

__all__ = [
    "BoundaryScrubber",
    "default_boundary_scrubber",
    "AuditLogger",
    "get_audit_logger",
    "PrivacyModel",
    "NoOpPrivacyModel",
]
