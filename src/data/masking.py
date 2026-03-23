from abc import ABC, abstractmethod


class Masker(ABC):
    @abstractmethod
    def mask_one(self, biography: str, value: str) -> tuple:
        """
        Masks a single value in the biography text.

        Returns:
            (masked_biography, found): masked_biography has the value replaced
            with [BLANK]; found is False if the value wasn't located in the text.
        """
        ...


class AttributeMasker(Masker):
    """
    Replaces an attribute value with [BLANK] using simple substring matching.
    Case-sensitive; skips the attribute if the value is not found in the text.
    """

    def mask_one(self, biography: str, value: str) -> tuple:
        if not value or value not in biography:
            return biography, False
        masked = biography.replace(value, "[BLANK]", 1)
        return masked, True


class RandomSpanMasker(Masker):
    """
    Masks a contiguous span of whitespace-delimited tokens at a random position.
    Useful as a fallback when structured attribute values are unavailable.
    """

    def __init__(self, min_span: int = 1, max_span: int = 3):
        self.min_span = min_span
        self.max_span = max_span

    def mask_one(self, biography: str, value: str) -> tuple:
        import random
        tokens = biography.split()
        if not tokens:
            return biography, False
        span_len = random.randint(self.min_span, min(self.max_span, len(tokens)))
        start = random.randint(0, len(tokens) - span_len)
        masked_tokens = tokens[:start] + ["[BLANK]"] + tokens[start + span_len:]
        return " ".join(masked_tokens), True
