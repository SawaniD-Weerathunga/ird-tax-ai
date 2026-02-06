# llm/disclaimer.py

DISCLAIMER_TEXT = "This response is based solely on IRD-published documents and is not professional tax advice."

def append_disclaimer(answer: str) -> str:
    ans = (answer or "").rstrip()
    if not ans:
        return DISCLAIMER_TEXT
    # avoid duplicating disclaimer
    if DISCLAIMER_TEXT.lower() in ans.lower():
        return ans
    return ans + "\n\n" + DISCLAIMER_TEXT
