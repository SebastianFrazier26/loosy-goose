from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _encoder(encoding: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding)


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    if not text:
        return 0
    return len(_encoder(encoding).encode(text, disallowed_special=()))
