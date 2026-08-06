"""Tool parser for models that emit a bare JSON object with no delimiter.

Qwen2.5 and Llama models emit a tool call as a plain JSON object -- no
``<tool_call>`` wrapper, no ``<function=...>`` tag, nothing to key a state
machine on. Verified on hardware: ``mlx-community/Qwen2.5-14B-Instruct-4bit``
produced ``{"name": "create_rule", "arguments": {...}}`` four times out of
four at temperature 0, even with the Hermes tool-call instruction inlined
verbatim in the system prompt. None of the other parsers in
``TOOL_PARSER_MAP`` read that shape.

Because there is no delimiter, this parser cannot tell "a tool call is
starting" the way every sibling parser can (by finding ``tool_open`` in the
stream). Two consequences follow:

* It is strict about SHAPE instead: only a top-level JSON object carrying
  both ``name`` (a non-empty string) and ``arguments`` (an object) is a call.
  Arrays, missing keys, and wrong-typed values are ordinary content.
* It must never run against a response the request didn't offer tools for --
  a model legitimately answering a question with a JSON object would
  otherwise be misread as calling a function. That gate can't live in the
  parser (it's constructed per model at load time, before any request
  exists), so it sets ``requires_tools = True`` and the handler
  (``app/handler/mlx_lm.py``) checks that flag against ``request.tools``
  before ever handing it text.
"""

from __future__ import annotations

import json
import re

from .abstract_parser import AbstractToolParser, ToolParserState

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class JsonObjectToolParser(AbstractToolParser):
    """Tool parser for a bare ``{"name": ..., "arguments": {...}}`` payload.

    Handles model output that is exactly a top-level JSON object (optionally
    wrapped in a ```` ```json ```` / ```` ``` ```` fence), for example::

        {"name": "get_weather", "arguments": {"city": "NYC"}}

    Only run this parser when the request actually sent ``tools`` -- see the
    module docstring and ``requires_tools`` below.
    """

    #: Only this parser sets this. The handler gates on it to avoid ever
    #: running a delimiter-less parser against a response to a tools-less
    #: request.
    requires_tools = True

    def __init__(self, tool_open: str = "", tool_close: str = "") -> None:
        """Initialize the parser. There is no wire-format delimiter, so both
        markers are the empty string (see the module docstring)."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)

    def get_tool_open(self) -> str:
        """Return the empty string: this format has no opening marker."""
        return self.tool_open

    def get_tool_close(self) -> str:
        """Return the empty string: this format has no closing marker."""
        return self.tool_close

    @staticmethod
    def _strip_fence(text: str) -> str:
        """Strip surrounding whitespace and an optional ``` / ```json fence."""
        stripped = text.strip()
        match = _FENCE_RE.match(stripped)
        if match:
            return match.group(1).strip()
        return stripped

    @staticmethod
    def _as_tool_call(candidate: str) -> dict[str, str] | None:
        """Parse ``candidate`` and return a flat tool-call dict, or None.

        A match requires a top-level JSON *object* (not an array or scalar)
        carrying both a non-empty string ``name`` and an object
        ``arguments``. Anything else -- malformed JSON, missing keys, wrong
        types -- is not a tool call.
        """
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        arguments = data.get("arguments")
        if not isinstance(name, str) or not name:
            return None
        if not isinstance(arguments, dict):
            return None
        return {"name": name, "arguments": json.dumps(arguments)}

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract a tool call from complete model output.

        Parameters
        ----------
        model_output : str
            Complete model output, expected to be a bare JSON object
            (optionally fenced).

        Returns
        -------
        dict[str, list] | None
            ``{"tool_calls": [...], "content": ""}`` when ``model_output`` is
            a strictly-shaped tool call. ``{"content": model_output}``
            (verbatim, unmangled) otherwise -- this includes prose and any
            JSON that isn't a well-formed call.
        """
        call = self._as_tool_call(self._strip_fence(model_output))
        if call is None:
            return {"content": model_output}
        # Emit an explicit string "content" alongside the call. The handler
        # (app/handler/mlx_lm.py) only falls back to `_strip_complete_tool_blocks`
        # when "content" is *not* a string; with empty open/close markers that
        # helper never advances (`str.find("", i)` always matches at `i`) and
        # would spin forever. Setting content="" here keeps that path dead.
        return {"tool_calls": [call], "content": ""}

    def extract_tool_calls_streaming(self, chunk: str) -> tuple[dict | str | None, bool]:
        """Extract tool calls from streaming chunks.

        With no delimiter to key on, this buffers everything that could
        still become a tool call and only reports once the buffer resolves
        one way or the other:

        * Buffer content whose first non-whitespace character is ``{`` or a
          fence backtick is held back until it parses as complete JSON
          (matching the strict call shape or not) -- reporting a half
          received object as a call would be wrong, and reporting a
          half-received object as plain content would truncate it.
        * Anything else is passed through immediately: it can never resolve
          to a JSON object, so there is no reason to withhold it from the
          stream.

        Parameters
        ----------
        chunk : str
            Chunk of model output to process.

        Returns
        -------
        tuple[dict | str | None, bool]
            ``(payload, is_complete)`` -- ``payload`` carries ``tool_calls``
            and/or ``content``; ``is_complete`` is True once the buffered
            text has been fully resolved (as a call, as non-matching
            content, or as definitely-not-JSON content), False while still
            buffering.
        """
        self.buffer += chunk

        looks_like_json = bool(self.buffer.strip()) and self.buffer.strip()[0] in "{`"
        if not looks_like_json:
            passthrough = self.buffer
            self.buffer = ""
            self.state = ToolParserState.NORMAL
            if passthrough:
                return {"content": passthrough}, True
            return None, False

        candidate = self._strip_fence(self.buffer)
        call = self._as_tool_call(candidate)
        if call is not None:
            self.buffer = ""
            self.state = ToolParserState.NORMAL
            return {"tool_calls": [call], "content": ""}, True

        # Still ambiguous: either an incomplete object, or complete-but-
        # invalid JSON/shape. Only resolve the "complete but not a call"
        # case once the buffer itself parses as valid JSON -- otherwise we
        # cannot distinguish "not done yet" from "never going to match".
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            self.state = ToolParserState.FOUND_PREFIX
            return None, False

        content = self.buffer
        self.buffer = ""
        self.state = ToolParserState.NORMAL
        return {"content": content}, True
