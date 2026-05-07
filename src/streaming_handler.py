"""LangChain callback handler for real-time token streaming."""

from langchain_core.callbacks import BaseCallbackHandler


class TokenStreamHandler(BaseCallbackHandler):
    """Prints tokens as they arrive from the LLM, one line per call."""

    def __init__(self) -> None:
        self.current_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)
        self.current_text += token

    def on_llm_end(self, response, **kwargs) -> None:
        print()

