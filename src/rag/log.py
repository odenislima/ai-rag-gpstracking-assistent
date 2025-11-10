import os
import sys
from datetime import datetime

_TRUTHY = {"1", "true", "yes", "on"}

def rag_log(msg: str, *args, level: str = "INFO", ts: bool = False, end: str = "\n", flush: bool = True) -> None:
    """
    Print `msg` only if RAG_LOG_ENABLED is truthy (1/true/yes/on).
    Supports str.format(*args). Writes to stderr.
    """
    if os.getenv("RAG_LOG_ENABLED", "0").lower() not in _TRUTHY:
        return
    if args:
        msg = msg.format(*args)
    prefix = f"[{level}] "
    if ts:
        prefix = f"{datetime.now().strftime('%H:%M:%S')} {prefix}"
        
    print(prefix + msg, file=sys.stderr, end=end, flush=flush)
