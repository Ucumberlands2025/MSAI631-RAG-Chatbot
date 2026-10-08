"""Gradio UI for the coursework RAG assistant."""
from functools import lru_cache

import gradio as gr

from rag import format_evidence, load_assistant


@lru_cache(maxsize=1)
def get_assistant():
    # Lazy model loading: first question may take time and download weights.
    return load_assistant()


def respond(message: str, history: list[dict] | None = None) -> str:
    del history  # No cross-session or cross-turn memory by design.
    try:
        result = get_assistant().ask(message)
    except Exception as exc:
        # Do not present failure as an AI-generated fact.
        return (
            "**Setup or inference error.** The chatbot could not load its models or index. "
            "Check the Space logs or local terminal; see SETUP_AND_SUBMISSION.md.\n\n"
            f"Error type: `{type(exc).__name__}`."
        )
    answer = result["answer"]
    if result["sources"]:
        answer += "\n\n---\n**Retrieved source excerpts (verify the answer):**\n\n"
        answer += format_evidence(result["sources"])
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Course FAQ Assistant — RAG Demonstration",
    description=(
        "Ask a question about the SAMPLE course documents. Answers are generated with "
        "Flan-T5-small from retrieved passages; source excerpts appear below the answer. "
        "Sample content is fictional and must not be relied on as real university policy."
    ),
    examples=[
        "When are demonstration assignments due?",
        "How should I ask for a sample extension?",
        "What is the sample participation requirement?",
        "What is the weather forecast tomorrow?",
    ],
)

if __name__ == "__main__":
    demo.launch()
