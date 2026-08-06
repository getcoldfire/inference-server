"""The permissive parser: the only one that reaches models emitting a bare JSON
tool call (verified: mlx-community/Qwen2.5-14B-Instruct-4bit, 4/4 at temp 0).
It is deliberately strict about SHAPE because it has no delimiter to key on.

Note on shape: every sibling parser (see tests/parsers/test_hermes.py,
tests/parsers/test_streaming_tag_split_boundaries.py) emits tool_calls entries
as a *flat* ``{"name": ..., "arguments": <json str>}`` dict -- the "id" and
"function" nesting the OpenAI wire format needs is assembled downstream, at
the handler/endpoint layer (see app/api/endpoints.py:1094,
``FunctionCall(name=tool_call.get("name"), ...)``). This test file follows
that convention rather than the plan draft's nested access, matching what
extract_tool_calls actually returns for every other parser in the codebase.
"""

import json

from app.parsers.json_object import JsonObjectToolParser


def calls(text):
    out = JsonObjectToolParser().extract_tool_calls(text)
    return None if out is None else out.get("tool_calls")


def test_parses_a_bare_tool_call():
    got = calls('{"name": "create_rule", "arguments": {"name": "Junk bob"}}')
    assert len(got) == 1
    assert got[0]["name"] == "create_rule"
    # arguments must be a JSON STRING on the way out, per the OpenAI schema
    assert json.loads(got[0]["arguments"]) == {"name": "Junk bob"}


def test_rejects_prose():
    assert calls("Sure, I'll move those to Junk.") is None


def test_rejects_json_that_is_not_a_tool_call():
    assert calls('{"result": 42}') is None                      # no name/arguments
    assert calls('{"name": "x"}') is None                       # arguments missing
    assert calls('[{"name": "x", "arguments": {}}]') is None    # array, not object


def test_tolerates_whitespace_and_code_fences():
    assert calls('```json\n{"name": "f", "arguments": {"a": 1}}\n```')[0]["name"] == "f"


def test_streaming_defers_until_the_object_is_complete():
    p = JsonObjectToolParser()
    partial, done = p.extract_tool_calls_streaming('{"name": "f", "argum')
    assert not done, "a half-received object must not be reported as a call"


def test_streaming_reports_the_call_once_complete():
    p = JsonObjectToolParser()
    p.extract_tool_calls_streaming('{"name": "f", "argum')
    payload, done = p.extract_tool_calls_streaming('ents": {"a": 1}}')
    assert done
    assert payload["tool_calls"][0]["name"] == "f"
    assert json.loads(payload["tool_calls"][0]["arguments"]) == {"a": 1}


def test_streaming_passes_prose_straight_through():
    """Content that can never become a JSON object should not be buffered."""
    p = JsonObjectToolParser()
    payload, done = p.extract_tool_calls_streaming("Sure, I can help with that.")
    assert done
    assert payload == {"content": "Sure, I can help with that."}


def test_get_tool_open_and_close_are_empty():
    """No delimiter exists for this format; the handler relies on this being ''
    (see app/handler/mlx_lm.py:_strip_complete_tool_blocks -- empty markers must
    never reach that helper with a truthy tool_calls result, since `"" in text`
    is always True and `str.find("", i) == i` never advances, looping forever).
    """
    p = JsonObjectToolParser()
    assert p.get_tool_open() == ""
    assert p.get_tool_close() == ""
    # Confirm extract_tool_calls always ships an explicit string "content" so the
    # handler's `isinstance(tool_content, str)` branch wins and the strip helper
    # is never invoked for this parser.
    result = p.extract_tool_calls('{"name": "f", "arguments": {}}')
    assert result["tool_calls"]
    assert isinstance(result["content"], str)
