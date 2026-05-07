# tests/test_synthesis_agents.py

from unittest.mock import MagicMock, patch

from src.agents.synthesis_drafter import SynthesisDrafter
from src.agents.socratic_critic import SocraticCritic
from src.agents.arbiter import Arbiter


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _mock_ollama_invoke(return_ascii: str = "Draft with citation @smith2024"):
    """Return a MagicMock that, when called, returns an object with .content"""
    mock_response = MagicMock()
    mock_response.content = return_ascii
    return mock_response


# ---------------------------------------------------------------------------
# Drafter
# ---------------------------------------------------------------------------

def test_drafter_constructs_prompt_and_scrubs_output():
    drafter = SynthesisDrafter(model_name="fake-model")
    entities = {"cytokine": [{"entity": "IL-6", "evidence": "..."}]}
    chunks = [{"text": "IL-6 increased in obese mice.", "metadata": {}}]
    citations = ["@avery2025"]
    kg_context = {"nodes": []}

    # Simulate LLM returning a string with a non-ASCII character (Greek alpha)
    raw_output = "Draft with α and β."
    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw_output)) as mock_invoke:
        result = drafter.draft(
            query="Test?",
            entities=entities,
            chunks=chunks,
            citations=citations,
            kg_context=kg_context,
        )

    # The output must be plain ASCII; α→alpha, β→beta via scrub_unicode
    assert "α" not in result
    assert "β" not in result
    assert "alpha" in result.lower()
    assert "beta" in result.lower()
    # Check that the prompt included the query, citations, etc.
    messages = mock_invoke.call_args[0][0]
    system_msg = messages[0].content
    user_msg = messages[1].content
    assert "biomedical literature synthesis drafter" in system_msg.lower()
    assert "Test?" in user_msg
    assert "@avery2025" in user_msg


def test_drafter_no_citations():
    drafter = SynthesisDrafter(model_name="fake-model")
    entities = {}
    chunks = []
    citations = []  # empty
    kg_context = {}
    raw_output = "Simple draft."

    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw_output)) as mock_invoke:
        result = drafter.draft("Q", entities, chunks, citations, kg_context)
    assert result == "Simple draft."
    # The user prompt should mention citations as "none provided"
    user_msg = mock_invoke.call_args[0][0][1].content
    assert "none provided" in user_msg.lower()


# ---------------------------------------------------------------------------
# Socratic Critic
# ---------------------------------------------------------------------------

def test_critic_returns_no_critique():
    critic = SocraticCritic(model_name="fake-model")
    draft = "All claims are perfect."
    chunks = [{"text": "Evidence.", "metadata": {}}]
    entities = {}

    # LLM returns the exact NO_CRITIQUE string (with possible non-ASCII)
    raw = "NO_CRITIQUE: All claims are evidence-grounded."
    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw)) as mock_invoke:
        result = critic.critique(draft, chunks, entities)
    assert result == raw  # after scrubbing it's already ASCII
    assert "NO_CRITIQUE" in result

    # Verify prompt instructions
    system_msg = mock_invoke.call_args[0][0][0].content
    assert "NEVER propose alternative text" in system_msg


def test_critic_with_unicode_scrub():
    critic = SocraticCritic(model_name="fake-model")
    draft = "Some text."
    chunks = [{"text": "data."}]
    entities = {}
    raw = "Claim μ is unsupported."  # mu
    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw)):
        result = critic.critique(draft, chunks, entities)
    assert "μ" not in result
    # In the current mapping, μ becomes plain ASCII "u"
    assert "claim u is unsupported." in result.lower()


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------

def test_arbiter_revise_scrubs_and_includes_critique():
    arbiter = Arbiter(model_name="fake-model")
    draft = "Original draft."
    critique = "Claim X lacks evidence."
    chunks = [{"text": "Evidence for X.", "metadata": {}}]

    raw = "Revised draft with σ symbol."  # sigma
    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw)) as mock_invoke:
        result = arbiter.revise(draft, critique, chunks)
    assert "σ" not in result
    # scrub_unicode maps σ→sigma
    assert "sigma" in result

    # Check that critique text is present in the prompt
    user_msg = mock_invoke.call_args[0][0][1].content
    assert critique in user_msg
    assert "Original draft." in user_msg


def test_arbiter_no_critique():
    arbiter = Arbiter(model_name="fake-model")
    draft = "Good draft."
    critique = "NO_CRITIQUE: All claims are evidence-grounded."
    chunks = [{"text": "Evidence."}]
    raw = "Good draft unchanged."
    with patch("langchain_ollama.ChatOllama.invoke", return_value=_mock_ollama_invoke(raw)):
        result = arbiter.revise(draft, critique, chunks)
    assert result == raw