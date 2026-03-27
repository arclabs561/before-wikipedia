from __future__ import annotations

from collections import Counter
from itertools import groupby
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

import nltk
from bs4 import BeautifulSoup
from nltk.util import ngrams


class Predictor:
    def __init__(self) -> None:
        pass

    def urltext(self, url: str) -> str:
        print("retrieving text...")
        try:
            html = urlopen(url, timeout=10).read()
        except URLError as exc:
            print(f"failed to retrieve {url}: {exc}")
            return ""
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    def analyze(
        self,
        raw: str,
        back: int = 2,
        ahead: int = 1,
        normalize: Callable[[str], str] = str.lower,
    ) -> dict[tuple[str, ...], list[tuple[tuple[str, ...], int]]]:
        tokens = nltk.word_tokenize(normalize(raw))
        grams = ngrams(tokens, back + ahead)
        print("generating transitions...")
        trans = {
            k: Counter(t[-ahead:] for t in group).most_common()
            for k, group in groupby(sorted(grams), key=lambda x: x[:back])
        }
        return trans

    # TODO: implement persistence for tokenized output
    def save_tokens(self) -> None:
        pass


if __name__ == "__main__":
    p = Predictor()
    raw = p.urltext("http://www.masseiana.org/pliny.htm")
    d = p.analyze(raw)
