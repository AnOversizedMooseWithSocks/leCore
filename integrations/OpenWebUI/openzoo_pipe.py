"""
title: openzoo
author: leCore / openzoo
author_url: https://openzoo.fun
funding_url: https://openzoo.fun
version: 0.1.0
license: MIT
description: Route chats through openzoo.fun - leCore-backed inference, 480+ models, effectively unlimited context, pay-per-call. Works with the local x402 proxy (npx openzoo) today and the hosted endpoint when available.
requirements: requests
"""

# WHY this exists: openzoo exposes an OpenAI-compatible endpoint (locally at
# http://localhost:8402/v1 via `npx openzoo`, or hosted). OpenWebUI "manifold"
# pipe functions can surface an entire external provider as entries in the
# model dropdown. This file is a pure HTTP forwarder: no payment logic lives
# here - the local proxy pays x402 invoices from its burner wallet, and a
# hosted endpoint would use ordinary API keys. Keeping payment out of the
# plugin means one file works for both rails, and there is nothing here that
# can lose anyone's money.
#
# Kept negative: we deliberately do NOT implement corpus-spill client-side.
# The leCore spill (bind a huge corpus, feed the model only a few k tokens)
# belongs server-side at openzoo so every client gets it with zero code.
# A client-side spill was considered and rejected: it would fork behavior
# between harnesses and double-bill oversized bodies.

import json
import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        # Default is the local proxy started by `npx openzoo`. The hosted endpoint is
        # https://x402-tokens.fly.dev/v1 (api.openzoo.fun/v1 404s -- it is the
        # website, not the gateway).
        # NOTE if you run open-webui in Docker (the documented install): set
        # this to http://host.docker.internal:8402/v1 -- inside a container
        # "localhost" is the container, and the proxy is on the HOST.
        OPENZOO_BASE_URL: str = Field(
            default="http://localhost:8402/v1",
            description="OpenAI-compatible base URL for openzoo (local proxy or hosted).",
        )
        # The local proxy ignores the key (payment happens via x402); a hosted
        # endpoint will require a real one. Either way the header is sent.
        OPENZOO_API_KEY: str = Field(
            default="sk-openzoo",
            description="API key. Any value works for the local proxy; real key for hosted.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=600,
            description="Per-request timeout in seconds (long, for big-corpus calls).",
        )
        MODEL_PREFIX: str = Field(
            default="zoo/",
            description="Display prefix for zoo models in the model dropdown.",
        )
        SHOW_RECEIPTS: bool = Field(
            default=True,
            description="Append openzoo billing receipts (billedUsd, savings) to responses when present.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------ helpers

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.valves.OPENZOO_API_KEY}",
            "Content-Type": "application/json",
        }

    def _strip_owui_prefix(self, model_id: str) -> str:
        # OpenWebUI namespaces manifold models as "<function_id>.<model_id>".
        # WHY the guards: zoo model ids themselves contain dots
        # ("nvidia/nemotron-3.5-lightning"), so blindly splitting on the first
        # "." would mangle a bare id. We strip only when the head looks like a
        # function id: the first "." comes before any "/", and the head has no
        # digits (function ids are names like "openzoo"; version-y heads like
        # "gpt-3" are part of a real model id and must be kept).
        if "." not in model_id:
            return model_id
        head, tail = model_id.split(".", 1)
        dot_before_slash = ("/" not in model_id) or (model_id.index(".") < model_id.index("/"))
        head_is_function_id = not any(c.isdigit() for c in head)
        if tail and dot_before_slash and head_is_function_id:
            return tail
        return model_id

    # Only forward fields the OpenAI chat-completions API defines. OpenWebUI
    # injects harness bookkeeping (user, metadata, chat_id, session_id, files,
    # features, tool_ids, ...) that a strict upstream may reject with a 400.
    _OPENAI_CHAT_FIELDS = frozenset({
        "model", "messages", "stream", "stream_options", "temperature", "top_p",
        "max_tokens", "max_completion_tokens", "stop", "n", "presence_penalty",
        "frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "seed",
        "response_format", "tools", "tool_choice", "parallel_tool_calls", "user",
    })

    @staticmethod
    def _format_receipt(data: dict) -> str:
        # FIXED (was reading the wrong object): billing does NOT live in
        # `usage`. Verified against the live gateway, a response carries a
        # TOP-LEVEL "x402" block:
        #   x402: {billedUsd, cogsUsd, directUsd, savesVsDirect,
        #          subscription: {tier, cogsUsd, wouldHaveBilled, invoiced}}
        # while `usage` holds only OpenAI/OpenRouter fields (prompt_tokens,
        # completion_tokens, cost, cost_details, is_byok). The previous
        # version looked for billedUsd/savesVsDirect inside `usage`, never
        # found them, and silently rendered a token count and nothing else --
        # i.e. the receipt, which is the whole reason this plugin exists, was
        # dead code.
        x = data.get("x402") or {}
        usage = data.get("usage") or {}
        sub = x.get("subscription") or {}
        parts = []

        billed = x.get("billedUsd")
        if isinstance(billed, (int, float)):
            parts.append(f"billed ${billed:.6f}".rstrip("0").rstrip("."))

        # DELIBERATELY NOT rendering `savesVsDirect` as a percentage saved.
        # Measured: billedUsd 0.00700472 / wouldHaveBilled 0.02101416 =
        # 0.3333 == savesVsDirect. So the field is the RATIO PAID, not the
        # fraction saved -- printing "saved 0.33" claims a third when the
        # real saving is two thirds. Show both absolute numbers instead;
        # they cannot be misread.
        would = sub.get("wouldHaveBilled") or x.get("directUsd")
        if isinstance(would, (int, float)) and isinstance(billed, (int, float)) \
                and would > billed > 0:
            parts.append(f"vs ${would:.6f}".rstrip("0").rstrip(".") + " direct"
                         f" ({would / billed:.1f}x)")

        if sub.get("tier"):
            parts.append(f"{sub['tier']} subscription")
        elif x.get("paid"):
            parts.append(str(x["paid"]))

        pt = usage.get("prompt_tokens")
        if pt:
            parts.append(f"{pt} tokens read")
        return ("\n\n---\n*openzoo: " + " · ".join(parts) + "*") if parts else ""

    # ------------------------------------------------------------------ pipes

    def pipes(self):
        """Populate OpenWebUI's model dropdown from the zoo's live model list."""
        try:
            r = requests.get(
                f"{self.valves.OPENZOO_BASE_URL}/models",
                headers=self._headers(),
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                return [{"id": "openzoo-unavailable", "name": "openzoo: no models returned"}]
            return [
                {"id": m["id"], "name": f"{self.valves.MODEL_PREFIX}{m['id']}"}
                for m in data
            ]
        except Exception as e:
            # A visible error entry beats a silent empty dropdown: the most
            # common failure is simply that `npx openzoo` isn't running.
            return [{
                "id": "openzoo-error",
                "name": f"openzoo unreachable ({type(e).__name__}) - is `npx openzoo` running?",
            }]

    def pipe(self, body: dict):
        """Forward a chat completion to openzoo, streaming or not."""
        model = self._strip_owui_prefix(body.get("model", ""))
        if model.startswith("openzoo-"):
            return (
                "openzoo is not reachable. Start the local proxy with `npx openzoo` "
                f"(expected at {self.valves.OPENZOO_BASE_URL}) or set the hosted URL "
                "in this function's Valves."
            )

        payload = {k: v for k, v in body.items() if k in self._OPENAI_CHAT_FIELDS}
        payload["model"] = model
        stream = bool(payload.get("stream", False))

        try:
            r = requests.post(
                f"{self.valves.OPENZOO_BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
                stream=stream,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            return (
                "Could not connect to openzoo at "
                f"{self.valves.OPENZOO_BASE_URL}. Start it with `npx openzoo`, "
                "or point OPENZOO_BASE_URL at the hosted endpoint."
            )
        except requests.exceptions.Timeout:
            return "openzoo request timed out. Large-corpus calls can be slow; raise REQUEST_TIMEOUT if needed."

        # x402: an unfunded local wallet surfaces as 402 from the proxy.
        if r.status_code == 402:
            return (
                "openzoo returned 402 Payment Required - the local burner wallet "
                "is unfunded. Run `npx openzoo address` to get the funding address "
                "and `npx openzoo balance` to check it."
            )
        if r.status_code == 401:
            return "openzoo returned 401 - check OPENZOO_API_KEY in this function's Valves."
        if not r.ok:
            try:
                detail = r.json().get("error", {}).get("message", r.text[:300])
            except Exception:
                detail = r.text[:300]
            return f"openzoo error {r.status_code}: {detail}"

        if stream:
            # Pass SSE lines through untouched; OpenWebUI parses them natively.
            return r.iter_lines()

        data = r.json()
        if self.valves.SHOW_RECEIPTS:
            receipt = self._format_receipt(data)   # whole body: x402 is top-level
            if receipt:
                try:
                    data["choices"][0]["message"]["content"] += receipt
                except (KeyError, IndexError, TypeError):
                    pass  # never let receipt cosmetics break a good response
        return data


# ---------------------------------------------------------------------------
# Self-test: exercises the pure logic (prefix stripping, receipt formatting)
# without a network. Run: python3 integrations/OpenWebUI/openzoo_pipe.py
# ---------------------------------------------------------------------------
def _selftest():
    p = Pipe()
    # prefix stripping: OpenWebUI's "<function>.<model>" form must round-trip
    assert p._strip_owui_prefix("openzoo.deepseek-v4") == "deepseek-v4"
    assert p._strip_owui_prefix("plain-model") == "plain-model"
    # KEPT NEGATIVE (bug fixed): zoo ids contain dots; a bare dotted id must
    # NOT be mangled by prefix stripping. The old split-on-first-dot code
    # returned "5-lightning" here.
    assert p._strip_owui_prefix("nvidia/nemotron-3.5-lightning") == "nvidia/nemotron-3.5-lightning"
    assert p._strip_owui_prefix("openzoo.nvidia/nemotron-3.5-lightning") == "nvidia/nemotron-3.5-lightning"
    assert p._strip_owui_prefix("gpt-3.5-turbo") == "gpt-3.5-turbo"  # digit head = model, keep
    assert p._strip_owui_prefix("openzoo.gpt-3.5-turbo") == "gpt-3.5-turbo"
    # payload sanitization: OpenWebUI bookkeeping must not reach upstream
    junk = {"model": "m", "messages": [], "stream": True,
            "chat_id": "x", "session_id": "y", "metadata": {}, "features": {}}
    clean = {k: v for k, v in junk.items() if k in Pipe._OPENAI_CHAT_FIELDS}
    assert set(clean) == {"model", "messages", "stream"}, clean
    # receipt formatting -- REGRESSION TEST for the bug this file used to have.
    # These are the exact shapes a live gateway returns; the old code read
    # `usage` for billedUsd/savesVsDirect and therefore rendered nothing.
    live = {
        "usage": {"prompt_tokens": 3100, "completion_tokens": 20, "cost": 0.007},
        "x402": {"billedUsd": 0.00700472, "savesVsDirect": 0.3333333333333333,
                 "cogsUsd": 0.00700472, "directUsd": 0.00700472,
                 "paid": "subscription",
                 "subscription": {"tier": "pro", "wouldHaveBilled": 0.02101416,
                                  "invoiced": "stripe"}},
    }
    full = Pipe._format_receipt(live)
    assert "billed $0.007" in full, full
    assert "vs $0.021" in full and "3.0x" in full, full   # 0.021/0.007
    assert "pro subscription" in full, full
    assert "3100 tokens read" in full, full
    # and the ratio must NOT be printed as a savings percentage
    assert "saved" not in full, full
    # a usage-only body (no x402 -- e.g. prepaid credit, no 402 emitted) still
    # renders the token count rather than blowing up
    assert "3100 tokens read" in Pipe._format_receipt({"usage": {"prompt_tokens": 3100}})
    assert Pipe._format_receipt({}) == ""
    # valves defaults are the local-proxy contract
    assert p.valves.OPENZOO_BASE_URL.endswith("/v1")
    print("openzoo_pipe selftest OK")


if __name__ == "__main__":
    _selftest()
