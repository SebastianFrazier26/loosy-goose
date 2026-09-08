from loosy_goose.tokens import count_tokens


def test_count_tokens_positive_and_monotone() -> None:
    short = "The quick brown fox."
    long = short + " It jumped over the lazy dog, twice, before lunch."
    assert count_tokens("") == 0
    assert count_tokens(short) > 0
    assert count_tokens(long) >= count_tokens(short)


def test_count_tokens_tolerates_special_token_text() -> None:
    assert count_tokens("<|endoftext|> should not raise") > 0
