"""Small engine protocol adapters. Requests use IDs, avoiding tokenizer drift."""

import httpx


class ProtocolError(ValueError):
    pass


class Backend:
    def __init__(self, engine, model):
        if engine not in ("vllm", "sglang"):
            raise ValueError("engine must be vllm or sglang")
        self.engine, self.model = engine, model

    async def identity(self, client):
        response = await client.get("/v1/models")
        response.raise_for_status()
        try:
            names = [item["id"] for item in response.json()["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed /v1/models response") from exc
        if self.model not in names:
            raise ProtocolError(f"Configured served model {self.model!r} is absent from /v1/models")
        return {"served_models": names, "revision_verified_by_api": False}

    def request(self, ids, output):
        if self.engine == "vllm":
            return "/v1/completions", {
                "model": self.model, "prompt": ids, "max_tokens": output,
                "min_tokens": output, "ignore_eos": True, "temperature": 0,
                "stream": False, "seed": 0,
            }
        return "/generate", {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": output, "min_new_tokens": output,
                                "ignore_eos": True, "temperature": 0},
            "stream": False,
        }

    def usage(self, data):
        try:
            if self.engine == "vllm":
                prompt, output = data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]
                reason = data["choices"][0]["finish_reason"]
            else:
                prompt, output = data["meta_info"]["prompt_tokens"], data["meta_info"]["completion_tokens"]
                reason = data["meta_info"].get("finish_reason")
            if any(type(value) is not int or value < 0 for value in (prompt, output)):
                raise ValueError("invalid token counts")
            return {"prompt_tokens": prompt, "output_tokens": output, "finish_reason": reason}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProtocolError("Response lacks valid per-request token usage") from exc


def endpoint(value):
    url = httpx.URL(value)
    if url.scheme not in ("http", "https") or not url.host or url.username or url.password or url.query or url.fragment:
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query or fragment")
    if url.path not in ("", "/"):
        raise ValueError("Use the server root URL, without /v1 or other path prefixes")
    return str(url).rstrip("/")
