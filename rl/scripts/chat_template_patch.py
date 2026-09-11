"""Resolve a chat_template PATH into its CONTENT before constructing OpenAIServingChat.

WHY
---
`OpenAIServingChat` expects the chat template's **content**. vLLM's own
`api_server` resolves a path first, via `load_chat_template()`. NeMo RL splats
`http_server_serving_chat_kwargs` straight into the constructor
(vllm_worker_async.py:~494) and never calls it, so a path is used verbatim AS the
template.

The result is a template that renders to a constant string -- the path itself:

    AssertionError: Found possibly non-monotonically increasing trajectory!
      Template prefix repr (detokenized):
        '/config/rl/configs/qwen3_5_retain_thinking.jinja'
      Template repr (detokenized):
        '/config/rl/configs/qwen3_5_retain_thinking.jinja'

Because every render is identical, `len(template_token_ids)` equals
`len(template_prefix_token_ids)`, the assert in `_replace_prefix_tokens` fires and
the request 500s.

This went unnoticed for eleven runs because that assert sits behind
`if not model_prefix_token_ids: return template_token_ids` -- it is only reachable
once the agent takes a SECOND turn, which never happened until the reasoning
parser was fixed.

WHY IT MATTERS FOR THIS MODEL SPECIFICALLY
------------------------------------------
Qwen3.6's stock template has a `preserve_thinking` flag:

    {%- if (preserve_thinking is defined and preserve_thinking is true)
           or (loop.index0 > ns.last_query_index) %}

Qwen3.5's does not -- it has only the `loop.index0 > ns.last_query_index` branch,
which strips <think> from every prior assistant turn once a plain user message is
appended. `qwen3_5_retain_thinking.jinja` hand-patches that flag into 3.5. So on
Qwen3.5 a working custom template is REQUIRED; on Qwen3.6 it can be dropped in
favour of `chat_template_kwargs: {preserve_thinking: true}`.

Deliberately does NOT import vLLM's `load_chat_template`: its module path has
moved between vLLM versions, and a plain read is all that is needed. A value that
is not an existing file is passed through untouched, so inline template content
keeps working.
"""

import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

anchor = """        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
"""
if anchor not in src:
    raise SystemExit(
        "serving_chat_kwargs anchor not found; upstream changed "
        "nemo_rl/models/generation/vllm/vllm_worker_async.py:~484."
    )

patched = (
    anchor
    + """
        # HYPOTEST_CHAT_TEMPLATE_PATCH: a path here is a PATH, not a template.
        # OpenAIServingChat wants the content; vLLM's api_server calls
        # load_chat_template() first, and this code path does not. Without this,
        # the path string itself becomes the template and every render is the
        # same constant, tripping the monotonicity assert in
        # _replace_prefix_tokens with a 500 on the second agent turn.
        _ct = serving_chat_kwargs.get("chat_template")
        if isinstance(_ct, str) and _ct:
            import os as _os

            if _os.path.isfile(_ct):
                with open(_ct) as _fh:
                    serving_chat_kwargs["chat_template"] = _fh.read()
                print(
                    "[HYPOTEST_CHAT_TEMPLATE] loaded %s (%d chars)"
                    % (_ct, len(serving_chat_kwargs["chat_template"])),
                    flush=True,
                )
            else:
                # Not a file: assume it is already inline template content. Warn
                # if it looks like a path anyway, since that is the failure this
                # patch exists to prevent.
                if _ct.endswith((".jinja", ".jinja2", ".j2")) or _os.sep in _ct[:200]:
                    print(
                        "[HYPOTEST_CHAT_TEMPLATE] WARNING: chat_template looks like a "
                        "path but is not a readable file: %r -- it will be used as "
                        "literal template content and the run will almost certainly "
                        "fail on the second agent turn." % (_ct[:200],),
                        flush=True,
                    )
                else:
                    print(
                        "[HYPOTEST_CHAT_TEMPLATE] using inline content (%d chars)"
                        % (len(_ct),),
                        flush=True,
                    )
"""
)

open(path, "w", encoding="utf-8").write(src.replace(anchor, patched, 1))
print("patched", path)
