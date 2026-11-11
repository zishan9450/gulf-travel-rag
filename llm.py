"""One place for Anthropic and OpenAI. Both keys can stay filled in.

The project reads `.env` on every call, so editing that file or saving the
Keys panel applies to the next question. No restart required.

  LLM_PROVIDER=anthropic   or   openai
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_FAST_MODEL = "gpt-5.4-nano"
OPENAI_JUDGE_MODEL = "gpt-5.4"

_anthropic = None
_anthropic_key = None
_openai = None
_openai_key = None


class LLMConfigError(RuntimeError):
    """Raised when the selected provider has no API key."""


class LLMError(RuntimeError):
    """Raised when the provider rejects the request."""


def read_dotenv():
    values = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name):
    """A value in .env wins, so the file is the switch. An empty line falls through to the shell."""
    value = read_dotenv().get(name, "")
    if value:
        return value
    return os.environ.get(name, "")


def update_settings(updates):
    """Write provider and keys. A blank value leaves the saved key alone."""
    global _anthropic, _anthropic_key, _openai, _openai_key
    current = read_dotenv()
    for key, value in updates.items():
        if value:
            current[key] = value.strip()
    lines = [
        "# Both keys can stay filled in. Change LLM_PROVIDER to switch on the next question.",
        f"LLM_PROVIDER={current.get('LLM_PROVIDER', 'anthropic')}",
        f"ANTHROPIC_API_KEY={current.get('ANTHROPIC_API_KEY', '')}",
        f"OPENAI_API_KEY={current.get('OPENAI_API_KEY', '')}",
        f"OPENAI_MODEL={current.get('OPENAI_MODEL', OPENAI_MODEL)}",
        f"OPENAI_FAST_MODEL={current.get('OPENAI_FAST_MODEL', OPENAI_FAST_MODEL)}",
        f"OPENAI_JUDGE_MODEL={current.get('OPENAI_JUDGE_MODEL', OPENAI_JUDGE_MODEL)}",
    ]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _anthropic = _openai = None
    _anthropic_key = _openai_key = None
    return public_settings()


def active_provider():
    choice = setting("LLM_PROVIDER").lower()
    if choice in ("openai", "anthropic"):
        return choice
    if setting("OPENAI_API_KEY") and not setting("ANTHROPIC_API_KEY"):
        return "openai"
    if setting("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise LLMConfigError(
        "Add an Anthropic key, an OpenAI key, or both in the Keys panel (saved to .env)."
    )


def public_settings():
    try:
        provider = active_provider()
    except LLMConfigError:
        provider = setting("LLM_PROVIDER") or "anthropic"
    return {
        "provider": provider,
        "anthropic_key_set": bool(setting("ANTHROPIC_API_KEY")),
        "openai_key_set": bool(setting("OPENAI_API_KEY")),
        "openai_model": setting("OPENAI_MODEL") or OPENAI_MODEL,
    }


def _anthropic_client():
    global _anthropic, _anthropic_key
    key = setting("ANTHROPIC_API_KEY")
    if not key:
        raise LLMConfigError("ANTHROPIC_API_KEY is empty. Add it in the Keys panel or in .env.")
    if _anthropic is None or _anthropic_key != key:
        import anthropic
        _anthropic = anthropic.Anthropic(api_key=key)
        _anthropic_key = key
    return _anthropic


def _openai_client():
    global _openai, _openai_key
    key = setting("OPENAI_API_KEY")
    if not key:
        raise LLMConfigError("OPENAI_API_KEY is empty. Add it in the Keys panel or in .env.")
    if _openai is None or _openai_key != key:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMConfigError("The OpenAI package is not installed. Run: pip install openai") from e
        _openai = OpenAI(api_key=key)
        _openai_key = key
    return _openai


def message_text(msg):
    """Claude replies can start with a thinking block. Keep the text blocks."""
    parts = [
        block.text
        for block in msg.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"Model returned no text (stop_reason={msg.stop_reason}).")
    return text


def _openai_model(model):
    if model.startswith("gpt-") or model.startswith("o"):
        return model
    if model.startswith("claude-haiku"):
        return setting("OPENAI_FAST_MODEL") or OPENAI_FAST_MODEL
    if model.startswith("claude-opus"):
        return setting("OPENAI_JUDGE_MODEL") or OPENAI_JUDGE_MODEL
    return setting("OPENAI_MODEL") or OPENAI_MODEL


def _openai_text(message):
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        bits = []
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                bits.append(text)
        return "\n".join(bits).strip()
    return ""


def _raise_provider(provider, exc):
    detail = str(exc).replace("\n", " ")
    if len(detail) > 300:
        detail = detail[:300] + "..."
    raise LLMError(f"{provider} error: {exc.__class__.__name__}: {detail}") from exc


def _complete_anthropic(model, system, user, max_tokens):
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # These models think by default and count that against max_tokens.
    if model.startswith("claude-haiku"):
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["output_config"] = {"effort": "low"}
    elif model.startswith("claude-sonnet"):
        kwargs["thinking"] = {"type": "between_tools"}
        kwargs["output_config"] = {"effort": "low"}
    else:
        kwargs["output_config"] = {"effort": "medium"}
    try:
        return message_text(_anthropic_client().messages.create(**kwargs))
    except LLMConfigError:
        raise
    except Exception as exc:
        if (exc.__class__.__module__ or "").startswith("anthropic"):
            _raise_provider("Anthropic", exc)
        raise


def _complete_openai(model, system, user, max_tokens):
    name = _openai_model(model)
    kwargs = {
        "model": name,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith("claude-opus"):
        kwargs["reasoning_effort"] = "medium"
    elif not name.startswith(("gpt-4", "gpt-3")):
        kwargs["reasoning_effort"] = "low"
    client = _openai_client()
    try:
        response = client.chat.completions.create(**kwargs)
    except LLMConfigError:
        raise
    except Exception as exc:
        module = exc.__class__.__module__ or ""
        if "reasoning_effort" in kwargs and module.startswith("openai") and "reasoning" in str(exc).lower():
            kwargs.pop("reasoning_effort")
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as again:
                if (again.__class__.__module__ or "").startswith("openai"):
                    _raise_provider("OpenAI", again)
                raise
        elif module.startswith("openai"):
            _raise_provider("OpenAI", exc)
        else:
            raise
    text = _openai_text(response.choices[0].message)
    if not text:
        raise RuntimeError(f"Model returned no text (finish_reason={response.choices[0].finish_reason}).")
    return text


def complete(model, system, user, max_tokens):
    if active_provider() == "openai":
        return _complete_openai(model, system, user, max_tokens)
    return _complete_anthropic(model, system, user, max_tokens)
