import pickle
from functools import wraps
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import TypeVar

from config import DEBUG
from config import DEBUG_DIR
from config import RERUN

F = TypeVar("F", bound=Callable)


def _cache_file(cache_name: str, cache_dir: Optional[Path]) -> Path:
    target_dir = cache_dir or DEBUG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{cache_name}.pkl"


def _should_bypass_cache(persistent: bool) -> bool:
    return not DEBUG and not persistent


def _save_pickle(path: Path, value) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def cached(
    cache_name: str,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
    persistent: bool = False,
    rerun: bool = False,
):
    """Cache a function result to ``<cache_dir>/<cache_name>.pkl``.

    ``persistent=True`` keeps caching enabled even when ``config.DEBUG`` is off.
    ``rerun=True`` or ``config.RERUN`` forces recomputation while preserving the
    original cache path and file format.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _should_bypass_cache(persistent):
                if verbose:
                    print(f"[Cache] DEBUG=False, computing {cache_name}")
                return func(*args, **kwargs)

            cache_file = _cache_file(cache_name, cache_dir)

            if RERUN or rerun:
                if verbose:
                    print(f"[Cache] RERUN=True, recomputing {cache_name}")
                result = func(*args, **kwargs)
                _save_pickle(cache_file, result)
                if verbose:
                    print(f"[Cache] Saved to {cache_file}")
                return result

            if cache_file.exists():
                if verbose:
                    print(f"[Cache] Loading {cache_name} from {cache_file}")
                return _load_pickle(cache_file)

            if verbose:
                print(f"[Cache] Cache miss, computing {cache_name}")
            result = func(*args, **kwargs)
            _save_pickle(cache_file, result)
            if verbose:
                print(f"[Cache] Saved to {cache_file}")
            return result

        return wrapper

    return decorator
