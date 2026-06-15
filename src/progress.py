from __future__ import annotations

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **_):
        return iterable
