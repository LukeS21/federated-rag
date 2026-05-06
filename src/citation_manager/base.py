from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class AbstractCitationManager(ABC):
    @abstractmethod
    def add_item(self, metadata: Dict) -> str:
        """Add a new reference, return its citation key."""
        pass

    @abstractmethod
    def download_pdf(self, item_key: str, save_path: str) -> Optional[str]:
        """Download PDF if available, return file path or None."""
        pass

    @abstractmethod
    def format_citation_key(self, item_key: str, style: str = "inline") -> str:
        """Return formatted citation (e.g., @smith2023)."""
        pass
