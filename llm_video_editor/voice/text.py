"""Prepare a narration line for speech: pronunciations, numbers and units.

The caption keeps the line as written ("380 km to Khajuraho"); the engine gets the
spoken form ("three hundred and eighty kilometres to Kha-joo-raa-ho").

Pronunciations live in ``lexicon.json`` files: ``<config>/lexicon.json`` (every
project) and ``<ws>/voice/lexicon.json`` (this one; wins). Keys match whole words,
case-insensitively; values are respellings the engine reads naturally.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}
UNITS = [  # (pattern after a number, spoken singular, spoken plural)
    (r"km/h|kmph|kph", "kilometre per hour", "kilometres per hour"),
    (r"km", "kilometre", "kilometres"), (r"m", "metre", "metres"),
    (r"kg", "kilogram", "kilograms"), (r"hrs?|h", "hour", "hours"),
    (r"mins?", "minute", "minutes"), (r"°C", "degree Celsius", "degrees Celsius"),
    (r"°", "degree", "degrees"), (r"%", "percent", "percent"),
    (r"₹|rs\.?|inr", "rupee", "rupees"),
]


def number_words(n: int) -> str:
    """English words for 0 <= n < 10**9, British style ("three hundred and eighty")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    for size, name in ((10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            head, rest = divmod(n, size)
            words = f"{number_words(head)} {name}"
            if rest:
                words += (" and " if rest < 100 else " ") + number_words(rest)
            return words
    return str(n)


def ordinal_words(n: int) -> str:
    w = number_words(n)
    last = re.split(r"[ -]", w)[-1]
    if last in _ORD:
        new = _ORD[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return w[: len(w) - len(last)] + new


def _year(n: int) -> str:
    """1999 -> nineteen ninety-nine; 2026 -> twenty twenty-six; 2000 -> two thousand."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return number_words(n) if hi % 10 == 0 else f"{number_words(hi)} hundred"
    if n < 2010 and n >= 2000:
        return number_words(n)
    return f"{number_words(hi)} {'oh-' + _ONES[lo] if lo < 10 else number_words(lo)}"


def speak_numbers(text: str) -> str:
    """Numbers, ordinals, years and common units as English words."""
    def unit(m: re.Match) -> str:
        num = float(m.group(1).replace(",", ""))
        val = m.group(1).replace(",", "")
        words = (number_words(int(num)) if num.is_integer()
                 else f"{number_words(int(num))} point {' '.join(_ONES[int(d)] for d in val.split('.')[1])}")
        for pat, one, many in UNITS:
            if re.fullmatch(pat, m.group(2), re.I):
                return f"{words} {one if num == 1 else many}"
        return words

    unit_alt = "|".join(p for p, _, _ in UNITS)
    text = re.sub(r"(?:₹|rs\.?\s?)(\d[\d,]*)", lambda m: f"{m.group(1)} rupees", text, flags=re.I)
    text = re.sub(rf"(\d[\d,]*(?:\.\d+)?)\s?({unit_alt})(?![a-z])", unit, text, flags=re.I)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1))), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _year(int(m.group(1))), text)
    text = re.sub(r"\b\d[\d,]*\b", lambda m: number_words(int(m.group(0).replace(",", ""))),
                  text)
    return text


def load_lexicon(*paths: Path) -> dict[str, str]:
    lex: dict[str, str] = {}
    for p in paths:
        if p and p.exists():
            lex.update({k.lower(): v for k, v in json.loads(p.read_text()).items()})
    return lex


def apply_lexicon(text: str, lex: dict[str, str]) -> str:
    if not lex:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(lex, key=len,
                                                                  reverse=True)) + r")\b",
                     re.I)
    return pat.sub(lambda m: lex[m.group(0).lower()], text)


def spoken(text: str, lang_code: str, lex: dict[str, str]) -> str:
    """The form sent to the engine."""
    out = apply_lexicon(text, lex)
    if lang_code.startswith("en"):
        out = speak_numbers(out)
    return re.sub(r"\s+", " ", out).strip()
