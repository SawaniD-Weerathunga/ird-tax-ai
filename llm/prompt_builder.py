from typing import List, Dict

def build_user_prompt(question: str, contexts: List[Dict]) -> str:
    parts = []
    parts.append(f"QUESTION:\n{question}\n")
    parts.append("CONTEXT:")

    for i, c in enumerate(contexts, start=1):
        doc = c.get("document", "unknown")
        pages = c.get("page_range", "unknown")
        section = c.get("section", "Not specified")
        text = c.get("content", "")[:1500]

        parts.append(
            f"\n[CONTEXT {i}]\n"
            f"Document: {doc}\n"
            f"Pages: {pages}\n"
            f"Section: {section}\n"
            f"Text: {text}\n"
        )

    parts.append(
        "\nINSTRUCTIONS:\n"
        "- Use ONLY the CONTEXT above.\n"
        "- Provide the answer + citations.\n"
    )
    return "\n".join(parts)
