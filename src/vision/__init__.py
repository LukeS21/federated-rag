"""Phase 7a: Vision Pipeline — figure extraction, smart filtering, vision model
integration, and figure-to-text embedding for cross-modal retrieval.
"""

from src.vision.figure_extractor import FigureExtractor
from src.vision.figure_filter import FigureFilter
from src.vision.vision_descriptor import VisionDescriptor
from src.vision.figure_embedder import FigureEmbedder

__all__ = ["FigureExtractor", "FigureFilter", "VisionDescriptor", "FigureEmbedder"]
