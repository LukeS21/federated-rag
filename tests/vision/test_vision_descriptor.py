"""
Tests for VisionDescriptor — model rotation and figure description via Ollama API.
"""
import json
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from src.vision.vision_descriptor import VisionDescriptor


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def descriptor():
    """VisionDescriptor with default settings."""
    return VisionDescriptor(model="llava:7b", ollama_host="http://localhost:11434")


@pytest.fixture
def sample_image():
    """A minimal 10×10 RGB PIL Image."""
    return Image.new("RGB", (10, 10), color="red")


def mock_requests_post_ok(*args, **kwargs):
    """Return a successful Ollama generate response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": "A bar chart showing IL-6 levels across three experimental groups."}
    return resp


def mock_requests_post_empty(*args, **kwargs):
    """Return an empty response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": ""}
    return resp


def mock_requests_post_pull_ok(*args, **kwargs):
    """Return successful pull response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"status": "success"}
    return resp


# ── Unit tests ──────────────────────────────────────────────────────────────

def test_default_model():
    """Default vision model is gemma4:e4b (best biomedical accuracy, already loaded)."""
    vd = VisionDescriptor()
    assert vd.model == "gemma4:e4b"


def test_custom_model():
    """Custom model name is respected."""
    vd = VisionDescriptor(model="minicpm-v:8b")
    assert vd.model == "minicpm-v:8b"


def test_api_url_construction(descriptor):
    """API URLs are constructed correctly."""
    assert descriptor._api_url("generate") == "http://localhost:11434/api/generate"
    assert descriptor._api_url("pull") == "http://localhost:11434/api/pull"
    assert descriptor._api_url("/chat") == "http://localhost:11434/api/chat"


def test_encode_image(descriptor, sample_image):
    """Image encoding produces valid base64 for Ollama."""
    b64 = descriptor._encode_image(sample_image)
    assert isinstance(b64, str)
    assert len(b64) > 20, "Base64 string too short"
    # Should be decodable
    import base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_scrub_text_removes_non_ascii(descriptor):
    """ASCII scrubbing removes non-ASCII characters."""
    text = "IL\u20106 levels in \u03b1\u03b2 T cells"
    cleaned = descriptor._scrub_text(text)
    assert all(ord(c) < 128 for c in cleaned)


@patch("requests.post")
def test_describe_success(mock_post, descriptor, sample_image):
    """describe() returns the generated text from Ollama."""
    mock_post.return_value = mock_requests_post_ok()

    desc = descriptor.describe(sample_image)

    assert "bar chart" in desc.lower()
    assert "il-6" in desc.lower()
    assert all(ord(c) < 128 for c in desc)


@patch("requests.post")
def test_describe_uses_custom_prompt(mock_post, descriptor, sample_image):
    """Custom prompt is passed to the API."""
    mock_post.return_value = mock_requests_post_ok()

    custom = "Summarize this microscopy image."
    descriptor.describe(sample_image, prompt=custom)

    call_args = mock_post.call_args
    sent_json = call_args[1]["json"]
    assert custom in sent_json["prompt"]


@patch("requests.post")
def test_describe_returns_empty_on_error(mock_post, descriptor, sample_image):
    """HTTP errors return empty string, not exception."""
    mock_post.side_effect = ConnectionError("Ollama not running")

    desc = descriptor.describe(sample_image)
    assert desc == ""


@patch("requests.post")
def test_unload_model(mock_post, descriptor):
    """unload_model sends keep_alive=0."""
    mock_post.return_value = mock_requests_post_ok()

    result = descriptor.unload_model("gemma4:e4b")
    assert result is True

    call_args = mock_post.call_args
    sent_json = call_args[1]["json"]
    assert sent_json["keep_alive"] == 0
    assert sent_json["model"] == "gemma4:e4b"


@patch("requests.post")
def test_pull_model(mock_post, descriptor):
    """pull_model sends pull request."""
    mock_post.return_value = mock_requests_post_pull_ok()

    result = descriptor.pull_model()
    assert result is True


@patch("requests.post")
def test_describe_figures(mock_post, descriptor, sample_image):
    """describe_figures processes a list of figures."""
    mock_post.return_value = mock_requests_post_ok()

    figures = [
        {"image": sample_image, "caption": "Figure 1.", "page_no": 3, "file_path": "/tmp/f1.png"},
        {"image": sample_image, "caption": "Figure 2.", "page_no": 4, "file_path": "/tmp/f2.png"},
        {"image": None, "caption": "Figure 3.", "page_no": 5, "file_path": ""},
    ]

    result = descriptor.describe_figures(
        figures,
        unload_first="gemma4:e4b",
        reload_after=False,
    )

    assert len(result) == 3
    for fig in result:
        assert "description" in fig
        assert fig["described_by"] in (descriptor.model, "skipped")
    # Figures 0 and 1 were described; figure 2 was skipped (no image)
    assert result[0]["described_by"] == descriptor.model
    assert result[1]["described_by"] == descriptor.model

    # Figure with no image and no file_path falls back to caption
    assert result[2]["description"] == "Figure 3."


@patch("requests.post")
def test_describe_figures_fallback_to_caption(mock_post, descriptor, sample_image):
    """Empty vision model response falls back to caption."""
    mock_post.return_value = mock_requests_post_empty()

    figures = [
        {"image": sample_image, "caption": "Figure 1: Bar chart.", "page_no": 3, "file_path": "/tmp/f1.png"},
    ]

    result = descriptor.describe_figures(figures, fallback_to_caption=True, reload_after=False)
    assert result[0]["description"] == "Figure 1: Bar chart."


def test_load_save_descriptions(tmp_path):
    """Descriptions can be saved to and loaded from JSON."""
    cache = tmp_path / "descriptions.json"

    data = {
        "/tmp/fig1.png": "A bar chart of cytokine levels",
        "/tmp/fig2.png": "Microscopy showing H&E staining",
    }
    VisionDescriptor.save_descriptions(cache, data)

    loaded = VisionDescriptor.load_descriptions(cache)
    assert loaded == data


def test_load_descriptions_missing():
    """Loading a non-existent cache returns empty dict."""
    loaded = VisionDescriptor.load_descriptions("/nonexistent/path.json")
    assert loaded == {}
