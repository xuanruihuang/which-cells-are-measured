"""Locations used by the scripts.

Values come from `config.env` at the repository root (lines KEY=value; ${KEY} refers to an earlier key or to an
environment variable) or from environment variables of the same name, which take precedence. See config.example.env
for the keys. Only file and folder locations are read here; no analysis setting depends on the configuration."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = None


def _load():
    vals = {}
    path = os.environ.get("FATE_CONFIG", os.path.join(ROOT, "config.env"))
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            v = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), vals.get(m.group(1), "")), v)
            vals[k] = v
    return vals


def get(key):
    """The location set for `key`; stops with a message if it is not set."""
    global _cache
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    if _cache is None:
        _cache = _load()
    if _cache.get(key):
        return _cache[key]
    raise SystemExit("%s is not set: copy config.example.env to config.env and set it, or export %s." % (key, key))
