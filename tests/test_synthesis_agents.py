"""Unit tests for synthesis agents (Ollama calls mocked)."""

from types import SimpleNamespace
from unittest.mock import patch

from src.agents.arbiter import Arbiter
from src.agents.socratic_critic import SocraticCritic
from src.agents.synthesis_drafter import SynthesisDrafter


def test_synthesis_drafter_scrubs_output():
    # TiO₂ should be scrubbed to TiO2
    with patch("src.agents.synthesis_drafter.ChatOllama") as MockChat:
        MockChat.return_value.invoke.return_value = SimpleNamespace(content="TiO₂ result @smith2025")
        drafter = SynthesisDrafter()
        out = drafter.draft(
            query="q",
            entities={"x": []},
            chunks=[{"text": "evidence"}],
            citations=["@smith2025"],
            kg_context={},
        )
        assert "TiO2" in out


def test_socratic_critic_returns_mocked_no_critique():
    with patch("src.agents.socratic_critic.ChatOllama") as MockChat:
        MockChat.return_value.invoke.return_value = SimpleNamespace(
            content="NO_CRITIQUE: All claims are evidence-grounded."
        )
        critic = SocraticCritic()
        out = critic.critique(draft="d", chunks=[{"text": "e"}], entities={"x": []})
        assert out == "NO_CRITIQUE: All claims are evidence-grounded."


def test_arbiter_scrubs_output():
    with patch("src.agents.arbiter.ChatOllama") as MockChat:
        MockChat.return_value.invoke.return_value = SimpleNamespace(content="Revised TiO₂ paragraph.")
        arbiter = Arbiter()
        out = arbiter.revise(draft="d", critique="c", chunks=[{"text": "e"}])
        assert "TiO2" in out

