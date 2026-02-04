from typing import List, Dict

def build_system_prompt() -> str:
    return """You are an IRD Tax Assistant for Sri Lanka.

STRICT RULES:
1) Answer ONLY using the provided CONTEXT.
2) Do NOT guess or add outside knowledge.
3) If the answer is not in the CONTEXT, say: "Unavailable in provided documents."
4) Always include citations for every claim using:
   - Document name
   - Page range
   - Section
5) If multiple documents disagree, prefer Public Notice (PN) over Guides.
6) Keep the answer short and clear.

Output format:
Answer:
- <your answer>

Citations:
- <document>, pages <x-y>, section <section>
"""

def build_user_prompt(question: str, contexts: List[Dict]) -> str:
    """
    contexts is a list of dicts like:
    {
      "document": "...",
      "page_range": [x,y],
      "section": "...",
      "content": "..."
    }
    """
    parts = []
    parts.append("QUESTION:")
    parts.append(question.strip())
    parts.append("\nCONTEXT:\n")

    for i, c in enumerate(contexts, start=1):
        parts.append(f"[CONTEXT {i}]")
        parts.append(f"Document: {c.get('document')}")
        parts.append(f"Pages: {c.get('page_range')}")
        parts.append(f"Section: {c.get('section', 'Not specified')}")
        parts.append(f"Text: {c.get('content', '')}".strip())
        parts.append("")  # blank line

    parts.append("INSTRUCTIONS:")
    parts.append("- Use ONLY the CONTEXT above.")
    parts.append("- Provide the answer + citations.")
    return "\n".join(parts)
