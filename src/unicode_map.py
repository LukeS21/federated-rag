# Mapping from common biomedical Unicode characters to plain ASCII equivalents
UNICODE_TO_PLAIN = {
    # Combined forms (must come before individual characters)
    'μg': 'mcg',           # micro sign + g  (U+00B5)
    'µg': 'mcg',           # mu + g          (U+03BC)
    'μg': 'mcg',           # just in case
    # Greek letters (lowercase)
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
    'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
    'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
    # Uppercase
    'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta', 'Ε': 'Epsilon',
    'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta', 'Ι': 'Iota', 'Κ': 'Kappa',
    'Λ': 'Lambda', 'Μ': 'Mu', 'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron',
    'Π': 'Pi', 'Ρ': 'Rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon',
    'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',
    # Subscripts
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
    '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    # Superscripts
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
    '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    # Special symbols
    '±': '+/-', '≥': '>=', '≤': '<=', '×': 'x', '°': 'deg',
    '′': "'", '″': '"', '∞': 'infinity', '≈': '~', '≠': '!=', '≡': '==',
    # The single μ must come AFTER the combined μg forms
    'μ': 'u',              # U+00B5 micro sign → 'u' (only if not part of μg)
    'µ': 'u',              # U+03BC Greek mu → 'u'
    # Additional biomedical symbols
    '–': '-',      # en dash
    '—': '--',     # em dash
    '‐': '-',      # hyphen
    '‑': '-',      # non-breaking hyphen
    '℃': 'degC',
    '℉': 'degF',
    '™': '(TM)',
    '®': '(R)',
    '©': '(C)',
    '±': '+/-',    # already there, ensure it's in
    '≤': '<=',
    '≥': '>=',
    'Δ': 'Delta',  # already there if uppercase delta
    'Δ': 'Delta',  # verify
    '‐': '-',
    '′': "'",      # prime
    '″': '"',      # double prime
    '→': '->',
    '←': '<-',
    '↔': '<->',
    '≈': '~',
    '≠': '!=',
    '·': '.',      # middle dot used in multiplication
    '×': 'x',      # multiplication sign
    '÷': '/',
}

import unicodedata

def scrub_unicode(text: str) -> str:
    # Step A: NFKC normalization (handle ligatures, fullwidth chars, etc.)
    text = unicodedata.normalize('NFKC', text)
    # Step B: Longest-match static mapping
    for k, v in sorted(UNICODE_TO_PLAIN.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, v)
    # Step C: Remove any remaining non-ASCII characters (catch-all)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    return text