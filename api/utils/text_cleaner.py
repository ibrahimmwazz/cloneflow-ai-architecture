"""
Text Normalization Utility
Cleans user text before sending to TTS models.
"""

import re


def clean_text(text: str) -> str:
    """Normalize text for TTS synthesis."""

    # Currency: $5 → five dollars
    text = re.sub(
        r'\$(\d+(?:\.\d+)?)',
        lambda m: _number_to_words(float(m.group(1))) + ' dollars',
        text
    )

    # Percentages: 10% → ten percent
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*%',
        lambda m: _number_to_words(float(m.group(1))) + ' percent',
        text
    )

    # Common abbreviations
    abbreviations = {
        r'\bPM\b': 'P M',
        r'\bAM\b': 'A M',
        r'\bDr\.?\b': 'Doctor',
        r'\bMr\.?\b': 'Mister',
        r'\bMrs\.?\b': 'Misses',
        r'\bMs\.?\b': 'Miss',
        r'\bSt\.?\b': 'Street',
        r'\bAve\.?\b': 'Avenue',
        r'\betc\.?\b': 'etcetera',
        r'\be\.g\.?\b': 'for example',
        r'\bi\.e\.?\b': 'that is',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Strip double spaces
    text = re.sub(r' {2,}', ' ', text)

    # Remove incompatible emoji / special chars
    text = re.sub(r'[^\x00-\x7F\u00A0-\uFFFF\w\s.,!?;:\'\"\-\(\)\[\]]+', '', text)

    # Strip leading/trailing whitespace
    return text.strip()


def _number_to_words(n: float) -> str:
    """Convert a number to words (0-999)."""
    if n == int(n):
        n = int(n)

    ones = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
        'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'
    ]
    tens = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
        'eighty', 'ninety'
    ]

    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (' ' + ones[n % 10] if n % 10 else '')
    if n < 1000:
        return ones[n // 100] + ' hundred' + (' and ' + _number_to_words(n % 100) if n % 100 else '')
    return str(n)
