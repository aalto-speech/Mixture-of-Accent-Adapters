"""
DHF: Deterministic Hallucination Filter
=======================================

Reference-free, rule-based post-processing that suppresses decoding artifacts of
autoregressive ASR decoders (Whisper), e.g.

  * looping / repeated n-grams          "i want to go i want to go i want to go ..."
  * runaway single-token repetitions    "the the the the the ..."
  * number-word spam tails              "... 17 90000 thousand thousand thousand ..."
  * filler / noise tokens               "sil", "uh", "um", "noise", ...
  * glued syllable spam                 "silalalalalalala..."
  * apostrophe garbage                  "''''''''''''''"

Algorithm (per hypothesis, no reference transcript is used):
  1. Detect:  `is_suspicious_hypothesis(h)` flags h using deterministic statistics
              (spam tokens, single-char tokens, longest token run, repeated 3-gram
              coverage, number ratio, lexical diversity).
  2. Propose: `generate_candidates(h)` builds a small set of repaired candidates
              (noise removal + run compression, number-tail cut, best-repeat-block
              compression, cut at second occurrence of a repeated phrase).
  3. Select:  `reference_free_quality_score` scores every candidate and the original;
              the best candidate replaces h only if it is at least +1.0 better
              (conservative replacement).

Only flagged hypotheses are modified; everything else is passed through unchanged.
The procedure is fully deterministic (no sampling, no model, no reference).
"""

import re
from typing import List

# ============================================================
# Vocabulary
# ============================================================
NUM_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
}

SPAM_TOKENS = {
    "sil", "sp", "uh", "um", "erm", "eh", "ah", "mm", "hmm",
    "noise", "background", "static",
}

HAS_ALPHA = re.compile(r"[a-z]", re.I)
NUMLIKE = re.compile(
    r"("
    r"\d{1,2}:\d{2}(?::\d{2})?"                         # time: 01:00 / 01:00:30
    r"|\d{4}-\d{2}-\d{2}"                               # date: 2024-01-31
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"                   # date: 9/10/1966 or 09-10-66
    r"|[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"     # numbers: 27555693, 1,000, 3.14, 10%
    r"|\d+(?:st|nd|rd|th)"                              # ordinals: 2nd, 3rd
    r")",
    re.I,
)


# ============================================================
# Normalization / tokenization
# ============================================================
def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _pre_clean_text(s: str) -> str:
    """Lowercase, normalize whitespace, split letter<->digit boundaries."""
    s = _normalize_ws((s or "").lower())
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    return s


def _tokens(s: str) -> List[str]:
    """Tokenize into words + digits + ordinals."""
    s = _pre_clean_text(s)
    return re.findall(r"[a-z]+|\d+(?:st|nd|rd|th)?", s)


def _detok(toks) -> str:
    return " ".join(toks).strip()


def _is_ordinal(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(st|nd|rd|th)", t))


def _is_numberish(t: str) -> bool:
    return t.isdigit() or t in NUM_WORDS or _is_ordinal(t)


# ============================================================
# Spam removal helpers
# ============================================================
def drop_apostrophe_garbage(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    s = raw.strip()
    if s.count("'") >= 12 and s.count(" ") <= 2:
        return ""
    return raw


def compress_token_runs(toks, keep_max=2):
    """Keep at most `keep_max` consecutive copies of the same token."""
    if not toks:
        return toks
    out = [toks[0]]
    run = 1
    for t in toks[1:]:
        if t == out[-1]:
            run += 1
            if run <= keep_max:
                out.append(t)
        else:
            run = 1
            out.append(t)
    return out


def number_spam_ratio(toks) -> float:
    if not toks:
        return 0.0
    return sum(1 for t in toks if _is_numberish(t)) / len(toks)


def _is_lalalala_spam_token(t: str) -> bool:
    """
    Catch glued syllable spam such as 'silalalalalal...', 'lalalalal...', 'alaalaala...'.
    Conservative: long token, very low character variety, strong (la/al/ala) repetition.
    """
    if not t or not t.isalpha():
        return False
    if len(t) < 12:
        return False
    if len(set(t)) > 5:
        return False
    if re.search(r"(?:la){5,}", t) or re.search(r"(?:al){5,}", t) or re.search(r"(?:ala){4,}", t):
        return True
    if t.startswith("sil") and (re.search(r"(?:la){4,}", t[3:]) or re.search(r"(?:ala){3,}", t[3:])):
        return True
    return False


def remove_noise_tokens(toks):
    """Drop known filler/noise tokens, single-letter tokens and glued syllable spam."""
    out = []
    for t in toks:
        if t in SPAM_TOKENS:
            continue
        if len(t) == 1 and t.isalpha():
            continue
        if _is_lalalala_spam_token(t):
            continue
        out.append(t)
    return out


# ============================================================
# Repetition detectors / fixers
# ============================================================
def find_best_consecutive_repeat(tokens, min_k=2, max_k=16, min_reps=3):
    """Find the consecutive repeated block (length k, >= min_reps copies) covering most tokens."""
    n = len(tokens)
    best = None  # (covered, k, -start, start, reps)
    for start in range(n):
        remaining = n - start
        max_k_eff = min(max_k, remaining // min_reps)
        for k in range(min_k, max_k_eff + 1):
            block = tokens[start:start + k]
            reps = 1
            i = start + k
            while i + k <= n and tokens[i:i + k] == block:
                reps += 1
                i += k
            if reps >= min_reps:
                covered = k * reps
                cand = (covered, k, -start, start, reps)
                if best is None or cand > best:
                    best = cand
    if best is None:
        return None
    _, k, _, start, reps = best
    return start, k, reps


def compress_best_repeat_block(tokens):
    """Keep the prefix and a single copy of the best repeated block; drop the rest."""
    best = find_best_consecutive_repeat(tokens, min_k=2, max_k=16, min_reps=3)
    if best is None:
        return tokens
    start, k, reps = best
    return tokens[:start] + tokens[start:start + k]


def cut_at_second_occurrence(tokens, Ls=(10, 9, 8, 7, 6, 5, 4, 3)):
    """Truncate at the second occurrence of the longest repeated phrase (length in Ls)."""
    n = len(tokens)
    for L in Ls:
        seen = {}
        for i in range(0, n - L + 1):
            ph = tuple(tokens[i:i + L])
            if ph in seen:
                return tokens[:i]
            seen[ph] = i
    return tokens


def cut_number_word_tail(tokens, trigger_words=("thousand", "hundred", "million", "billion", "trillion")):
    """Truncate where a 10-token window becomes dominated by number words."""
    if len(tokens) < 12:
        return tokens
    for i in range(10, len(tokens)):
        window = tokens[i - 10:i]
        num_ratio = number_spam_ratio(window)
        trig_cnt = sum(1 for t in window if t in trigger_words)
        if num_ratio >= 0.60 or trig_cnt >= 4:
            cut = max(3, i - 10)
            return tokens[:cut]
    return tokens


# ============================================================
# Candidate generation
# ============================================================
def generate_candidates(pred_text: str) -> List[str]:
    raw = pred_text or ""
    raw = drop_apostrophe_garbage(raw)
    if raw == "":
        return [""]

    toks0 = _tokens(raw)
    if not toks0:
        return [""]

    toks = remove_noise_tokens(toks0)
    toks = compress_token_runs(toks, keep_max=2)

    if not toks:
        return [""]

    cands = [
        _detok(toks),                               # A: cleaned
        _detok(cut_number_word_tail(toks)),         # B: cut number-spam tail
        _detok(compress_best_repeat_block(toks)),   # C: compress best repeat block
        _detok(cut_at_second_occurrence(toks)),     # D: cut at second occurrence
    ]

    if len(toks) >= 25 and number_spam_ratio(toks) >= 0.70:
        cands.append("")

    seen = set()
    uniq = []
    for c in cands:
        c = _normalize_ws(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ============================================================
# Reference-free scoring
# ============================================================
def longest_token_run(toks) -> int:
    if not toks:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def repeated_ngram_coverage(tokens, n=3) -> int:
    if len(tokens) < 2 * n:
        return 0
    seen = {}
    covered = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        if ng in seen:
            covered += n
        else:
            seen[ng] = i
    return covered


def reference_free_quality_score(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return -100.0

    L = len(toks)
    uniq_ratio = len(set(toks)) / max(1, L)
    num_ratio = number_spam_ratio(toks)
    spam_cnt = sum(1 for t in toks if t in SPAM_TOKENS or _is_lalalala_spam_token(t))
    single_char_cnt = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    long_run = longest_token_run(toks)
    rep3 = repeated_ngram_coverage(toks, n=3)
    rep4 = repeated_ngram_coverage(toks, n=4)

    score = 0.0
    # prefer normal-length outputs
    score -= 0.03 * max(0, L - 40)
    # penalize suspicious patterns
    score -= 2.0 * spam_cnt
    score -= 1.5 * single_char_cnt
    score -= 2.0 * max(0, long_run - 2)
    score -= 0.25 * rep3
    score -= 0.35 * rep4
    score -= 8.0 * max(0, num_ratio - 0.5)
    # reward lexical diversity a bit
    score += 2.0 * uniq_ratio
    # avoid degenerate ultra-short truncation
    if L <= 1:
        score -= 20.0
    elif L <= 2:
        score -= 8.0

    return score


def is_suspicious_hypothesis(pred_text: str) -> bool:
    toks = _tokens(pred_text)
    if not toks:
        return False

    L = len(toks)
    num_ratio = number_spam_ratio(toks)
    spam_cnt = sum(1 for t in toks if t in SPAM_TOKENS or _is_lalalala_spam_token(t))
    single_char_cnt = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    long_run = longest_token_run(toks)
    rep3 = repeated_ngram_coverage(toks, n=3)
    uniq_ratio = len(set(toks)) / max(1, L)

    if spam_cnt >= 1:
        return True
    if single_char_cnt >= 4:
        return True
    if long_run >= 4:
        return True
    if rep3 >= 6:
        return True
    if L >= 25 and num_ratio >= 0.70:
        return True
    if L >= 20 and uniq_ratio < 0.35:
        return True
    return False


def pick_best_candidate_reference_free(pred_text: str, min_gain: float = 1.0) -> str:
    orig = _normalize_ws(pred_text or "")
    candidates = generate_candidates(pred_text)

    if orig not in candidates:
        candidates = [orig] + candidates

    scored = []
    for cand in candidates:
        scored.append((reference_free_quality_score(cand), len(_tokens(cand)), cand))

    # higher score is better; if tied, prefer longer non-degenerate output
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, best_cand = scored[0]
    orig_score = reference_free_quality_score(orig)

    # conservative: only replace if clearly better
    if best_cand != orig and best_score >= orig_score + min_gain:
        return best_cand
    return orig


def apply_dhf(pred_text: str) -> str:
    """Apply DHF to a single hypothesis string."""
    text = "" if pred_text is None else str(pred_text)
    if is_suspicious_hypothesis(text):
        return pick_best_candidate_reference_free(text)
    return text


# ============================================================
# Analysis-only helper (uses text, not audio; not part of DHF itself)
# ============================================================
def flag_number_only_text(s) -> int:
    """1 if the string contains no letters and at least one numeric-like pattern."""
    s = "" if s is None else str(s).strip()
    if not s:
        return 0
    if HAS_ALPHA.search(s):
        return 0
    return 1 if NUMLIKE.search(s) else 0
