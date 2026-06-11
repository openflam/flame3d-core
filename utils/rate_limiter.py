"""Redis-backed request scheduler for LLM API calls.

``utils.llm_call.LLMCaller`` runs many requests concurrently (e.g. the object
identifier fans out a frame batch across a thread pool), which trips provider
rate limits. This module gives the caller a small **distributed token-bucket
limiter**: before every ``litellm.completion`` it asks the limiter for a slot
and blocks until one is free.

The state lives in Redis (the same instance compose already runs as the Celery
result backend), so the limit is enforced **across threads, processes, and
containers** — every worker thread and every pipeline subprocess draws from the
same per-model buckets.

Each model has two independent buckets, matching how providers express their
limits:

* **RPM** — requests per minute
* **TPM** — tokens per minute (prompt + reserved completion tokens)

Limits are configured per model, with a ``"default"`` entry applied to any model
not listed explicitly::

    "rate_limits": {
      "default":     { "requests_per_minute": 60,  "tokens_per_minute": 100000 },
      "gpt-4o-mini": { "requests_per_minute": 200, "tokens_per_minute": 400000 }
    }

Either field may be omitted to leave that dimension unlimited. The limiter
resolves its config from (in order): an explicit dict passed by the caller, the
JSON config at ``$FLAME3D_CONFIG_PATH``, or the shipped
``server/default_config.json``. If Redis is unreachable or a model has no
configured limit, the limiter **fails open** (it never blocks the pipeline on
infrastructure problems).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Atomic dual token-bucket step, evaluated server-side in Redis so concurrent
# callers can't race. Refills both a request bucket (KEYS[1]) and a token bucket
# (KEYS[2]) based on elapsed time, then consumes from BOTH only if both can
# satisfy the demand — otherwise it consumes from neither and reports how long
# to wait. Atomicity across the two buckets avoids spending a request slot while
# blocked on tokens (or vice versa).
#
#   KEYS[1] = request bucket   KEYS[2] = token bucket
#   ARGV[1] = req refill rate (tokens/sec)   ARGV[2] = req capacity
#   ARGV[3] = tok refill rate (tokens/sec)   ARGV[4] = tok capacity
#   ARGV[5] = now (unix seconds, float)      ARGV[6] = tokens requested
#
# A capacity <= 0 marks that bucket disabled (unlimited). Returns
# {allowed (0|1), wait_ms (int)}.
_DUAL_BUCKET_LUA = """
local function refill(key, rate, cap, now)
  local state = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(state[1])
  local ts = tonumber(state[2])
  if tokens == nil then tokens = cap; ts = now end
  local delta = now - ts
  if delta < 0 then delta = 0 end
  return math.min(cap, tokens + delta * rate)
end

local function persist(key, tokens, rate, cap, now)
  redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
  redis.call('PEXPIRE', key, math.ceil((cap / rate) * 1000) + 10000)
end

local req_rate, req_cap = tonumber(ARGV[1]), tonumber(ARGV[2])
local tok_rate, tok_cap = tonumber(ARGV[3]), tonumber(ARGV[4])
local now = tonumber(ARGV[5])
local want = tonumber(ARGV[6])

local req_on = req_cap > 0
local tok_on = tok_cap > 0

local req_tokens = req_on and refill(KEYS[1], req_rate, req_cap, now) or 0
local tok_tokens = tok_on and refill(KEYS[2], tok_rate, tok_cap, now) or 0

local req_ok = (not req_on) or (req_tokens >= 1)
local tok_ok = (not tok_on) or (tok_tokens >= want)

if req_ok and tok_ok then
  if req_on then persist(KEYS[1], req_tokens - 1, req_rate, req_cap, now) end
  if tok_on then persist(KEYS[2], tok_tokens - want, tok_rate, tok_cap, now) end
  return {1, 0}
end

-- Blocked: persist the refilled state (no consumption) and report the wait.
local wait_ms = 0
if req_on then
  persist(KEYS[1], req_tokens, req_rate, req_cap, now)
  if not req_ok then
    wait_ms = math.max(wait_ms, math.ceil(((1 - req_tokens) / req_rate) * 1000))
  end
end
if tok_on then
  persist(KEYS[2], tok_tokens, tok_rate, tok_cap, now)
  if not tok_ok then
    wait_ms = math.max(wait_ms, math.ceil(((want - tok_tokens) / tok_rate) * 1000))
  end
end
return {0, wait_ms}
"""

# Key prefix for bucket hashes in Redis (":r" requests, ":t" tokens).
_KEY_PREFIX = "flame3d:llm_ratelimit:"

# Never sleep longer than this between re-checks, so a token freed early (or a
# limit change) is picked up promptly even when the computed wait is large.
_MAX_SLEEP_SECONDS = 2.0


def _repo_root() -> Path:
    """Repo root, i.e. the parent of the ``utils`` package directory."""
    return Path(__file__).resolve().parents[1]


def _load_rate_limits_from_config() -> Dict[str, Any]:
    """Load the ``rate_limits`` mapping from the active JSON config.

    Looks at ``$FLAME3D_CONFIG_PATH`` first, then the shipped
    ``server/default_config.json``. Any error yields an empty mapping (the
    limiter then fails open for every model).
    """
    candidates = []
    env_path = os.environ.get("FLAME3D_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_repo_root() / "server" / "default_config.json")

    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            limits = config.get("rate_limits")
            if isinstance(limits, dict):
                return limits
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def _redis_url() -> str:
    """Resolve the Redis URL, reusing the Celery result backend by default."""
    return (
        os.environ.get("LLM_RATE_LIMIT_REDIS_URL")
        or os.environ.get("CELERY_RESULT_BACKEND")
        or "redis://redis:6379/0"
    )


def _as_positive_float(value: Any) -> Optional[float]:
    """Coerce *value* to a positive float, or None if it isn't one."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


class RateLimiter:
    """Distributed dual token-bucket limiter keyed by model name."""

    def __init__(
        self,
        rate_limits: Optional[Dict[str, Any]] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self._rate_limits: Dict[str, Any] = (
            rate_limits if rate_limits is not None else _load_rate_limits_from_config()
        )
        self._redis_url = redis_url or _redis_url()
        self._lock = threading.Lock()
        self._redis = None
        self._script = None
        self._redis_failed = False
        self._connect()

    def set_rate_limits(self, rate_limits: Dict[str, Any]) -> None:
        """Replace the in-memory limit table (e.g. when a caller passes its own)."""
        if rate_limits is not None:
            self._rate_limits = rate_limits

    def _connect(self) -> None:
        """Open the Redis connection once. Failures switch the limiter to no-op."""
        try:
            import redis  # local import: keeps the dependency optional

            client = redis.Redis.from_url(self._redis_url)
            client.ping()
            self._redis = client
            self._script = client.register_script(_DUAL_BUCKET_LUA)
        except Exception as exc:  # pragma: no cover - infra dependent
            self._redis = None
            self._redis_failed = True
            print(
                f"[rate_limiter] Redis unavailable ({self._redis_url}): {exc}. "
                "Proceeding without rate limiting."
            )

    def _limits_for(self, model: str) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(rpm, tpm)`` for *model*; each is None when unlimited.

        Accepts a ``{"requests_per_minute": N, "tokens_per_minute": M}`` object,
        or a bare integer (treated as RPM only) for backward compatibility.
        """
        entry = self._rate_limits.get(model, self._rate_limits.get("default"))
        if entry is None:
            return None, None
        if isinstance(entry, dict):
            return (
                _as_positive_float(entry.get("requests_per_minute")),
                _as_positive_float(entry.get("tokens_per_minute")),
            )
        # Legacy shape: a bare RPM number.
        return _as_positive_float(entry), None

    def acquire(self, model: str, est_tokens: int = 0) -> None:
        """Block until both the request and token budgets allow a call.

        Args:
            model: model name, used to pick limits and bucket keys.
            est_tokens: estimated tokens this request will consume (prompt +
                reserved completion). Only used for the TPM bucket.

        No-op when the model has no configured limit or Redis is unreachable.
        """
        rpm, tpm = self._limits_for(model)
        if (rpm is None and tpm is None) or self._redis is None or self._script is None:
            return

        # tokens/sec refill, with a one-minute burst capacity per dimension. A
        # capacity of 0 disables that bucket on the Lua side (unlimited).
        req_rate = (rpm or 0.0) / 60.0
        req_cap = float(rpm) if rpm else 0.0
        tok_rate = (tpm or 0.0) / 60.0
        tok_cap = float(tpm) if tpm else 0.0

        # A single request can't demand more than the bucket holds, else it
        # would wait forever; clamp to capacity (it then waits for a full bucket).
        want = max(1, int(est_tokens))
        if tok_cap > 0:
            want = min(want, int(tok_cap))

        req_key = _KEY_PREFIX + model + ":r"
        tok_key = _KEY_PREFIX + model + ":t"

        while True:
            try:
                allowed, wait_ms = self._script(
                    keys=[req_key, tok_key],
                    args=[req_rate, req_cap, tok_rate, tok_cap, repr(time.time()), want],
                )
            except Exception as exc:  # pragma: no cover - infra dependent
                # Redis dropped mid-run: fail open rather than stall the pipeline.
                with self._lock:
                    if not self._redis_failed:
                        self._redis_failed = True
                        print(
                            f"[rate_limiter] Redis error during acquire: {exc}. "
                            "Disabling rate limiting for this process."
                        )
                    self._redis = None
                return

            if int(allowed) == 1:
                return

            time.sleep(min(_MAX_SLEEP_SECONDS, max(0.01, float(wait_ms) / 1000.0)))


# Process-wide singleton so every LLMCaller shares the same connection/buckets.
_LIMITER: Optional[RateLimiter] = None
_LIMITER_LOCK = threading.Lock()


def get_rate_limiter(rate_limits: Optional[Dict[str, Any]] = None) -> RateLimiter:
    """Return the process-wide :class:`RateLimiter`, creating it on first use.

    An explicit *rate_limits* dict (e.g. derived from the run's config) takes
    precedence over the auto-loaded defaults.
    """
    global _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None:
            _LIMITER = RateLimiter(rate_limits=rate_limits)
        elif rate_limits is not None:
            _LIMITER.set_rate_limits(rate_limits)
        return _LIMITER
