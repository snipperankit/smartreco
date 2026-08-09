# Corporate proxy uses a self-signed CA — disable SSL verification globally.
import os as _os
import ssl as _ssl

_os.environ.setdefault("CURL_CA_BUNDLE", "")
_os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
_ssl._create_default_https_context = _ssl._create_unverified_context

# httpx (used by ChromaDB and OpenAI) ignores Python's global SSL context.
# Monkey-patch httpx to default verify=False in this corporate environment.
import httpx as _httpx

_orig_client_init = _httpx.Client.__init__


def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _orig_client_init(self, *args, **kwargs)


_httpx.Client.__init__ = _patched_client_init

_orig_async_init = _httpx.AsyncClient.__init__


def _patched_async_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _orig_async_init(self, *args, **kwargs)


_httpx.AsyncClient.__init__ = _patched_async_init

# ChromaDB uses httpx.stream() (module-level, not Client) for ONNX model download.
_orig_stream = _httpx.stream


def _patched_stream(*args, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 300.0)
    return _orig_stream(*args, **kwargs)


_httpx.stream = _patched_stream
