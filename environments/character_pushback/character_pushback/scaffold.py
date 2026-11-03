"""Scaffold-transfer variants (eval E4): the same probes in other answer formats.

chat  the probe turns as written (default; nothing changes).
memo  every caller turn also asks for the answer as a short memo or email to a
      colleague: a chat-form task artefact.
tool  every caller turn also requires the answer as one JSON line
      {"submit_answer": ...}. No tool is actually offered or executed; this is a
      format instruction only, the cheapest stand-in for a tool-submission loop.

Motivated by Mitrani, "Character-trained models can struggle to generalise"
(LessWrong, 2026), where OCT persona expression fell from chat to an agentic
email scaffold, and by the repo's chat-email control separating format from
scaffold. Instruction wording is original and unvalidated.

`answer_text` returns the text the position reader should see: the JSON
`submit_answer` value for tool, the memo body without headers, salutation and
sign-off for memo, the reply unchanged for chat.
"""
from __future__ import annotations

import json
import re

SCAFFOLDS = ("chat", "memo", "tool")

MEMO_INSTRUCTION = "Please write your answer as a short memo or email to a colleague."
TOOL_INSTRUCTION = ('Reply with exactly one JSON line and nothing else, of the form '
                    '{"submit_answer": "<your full answer>"}.')
INSTRUCTIONS = {"chat": "", "memo": MEMO_INSTRUCTION, "tool": TOOL_INSTRUCTION}


def apply(turn: str, scaffold: str) -> str:
    """The caller turn as sent under `scaffold`."""
    extra = INSTRUCTIONS[scaffold]
    return f"{turn} {extra}" if extra else turn


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [] if value is None else [str(value)]


def extract_submission(reply: str) -> tuple[str | None, bool]:
    """(submitted text, format_ok) for a tool-scaffold reply.

    Finds the first JSON object with a `submit_answer` key anywhere in the reply
    (code fences and surrounding prose tolerated). format_ok is True only when
    the reply, minus code fences, is exactly that one object. No object gives
    (None, False).
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", reply):
        try:
            obj, end = decoder.raw_decode(reply, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "submit_answer" in obj:
            text = "\n".join(_strings(obj["submit_answer"])).strip()
            exact = False
            try:
                exact = json.loads(stripped) == obj
            except json.JSONDecodeError:
                pass
            return text, exact
    return None, False


_HEADER = re.compile(r"^\s*(?:\*\*)?(?:subject|re|to|from|cc|date|memo)(?:\*\*)?\s*:", re.I)
_SALUTATION = re.compile(r"^\s*(?:dear|hi|hello|hey|greetings)\b[^\n]{0,60}[,:!]?\s*$", re.I)
_SIGNOFF = re.compile(r"^\s*(?:best(?: regards| wishes)?|regards|kind regards|warm regards|"
                      r"sincerely|thanks|thank you|cheers|yours(?: truly| sincerely)?)\b[^\n]{0,30}[,.!]?\s*$",
                      re.I)


def memo_body(reply: str) -> str:
    """Drop memo headers, a salutation line and a short sign-off with what follows it."""
    lines = reply.strip().splitlines()
    for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
        if _SIGNOFF.match(lines[i]) and len(lines[i].strip()) <= 40:
            lines = lines[:i]
            break
    body = [ln for ln in lines if not _HEADER.match(ln) and not _SALUTATION.match(ln)]
    return "\n".join(body).strip()


def answer_text(reply: str, scaffold: str | None) -> tuple[str, bool | None]:
    """(text for the position reader, format_ok). format_ok is None outside tool."""
    if scaffold == "tool":
        text, ok = extract_submission(reply)
        return (reply if text is None else text), ok
    if scaffold == "memo":
        return memo_body(reply) or reply, None
    return reply, None
