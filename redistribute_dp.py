#!/usr/bin/env python3
"""
DP-based text redistribution for parallel aligned texts.

Given N source (AG) lines and the full translation text with word-level
alignments, finds optimal line breaks that minimize misalignment.

The translation text is treated as a single stream of words. The DP
finds cut points that assign each contiguous span of words to a source
line, maximizing the number of words that end up on the same line as
their aligned AG counterpart.

Complexity: O(N * M^2) where N = source lines, M = translation words.
In practice M is bounded by passage size (~50 words), so this is fast.

Used by both redistribute_riordan.py (EN) and redistribute_polylas.py (MG).
"""


def knuth_plass(
    n_lines: int,
    words: list[str],
) -> list[int]:
    """Classic Knuth-Plass line breaking: distribute words across N lines
    minimizing the sum of squared deviations from ideal line length.

    This is a pure visual-balance pass with no alignment information.
    Used as the initial distribution before alignment-driven refinement.

    Returns N+1 cut points.
    """
    m = len(words)
    if n_lines <= 0 or m == 0:
        return [0, m]
    if n_lines == 1:
        return [0, m]
    if m <= n_lines:
        # Fewer words than lines: one word per line, rest empty
        cuts = list(range(m)) + [m] * (n_lines - m + 1)
        return cuts

    n = n_lines

    # Compute character lengths for each word (including a space separator)
    lengths = [len(w) + 1 for w in words]  # +1 for space
    # Prefix sum of lengths
    plen = [0] * (m + 1)
    for i in range(m):
        plen[i + 1] = plen[i] + lengths[i]

    ideal_len = plen[m] / n  # ideal character length per line

    def badness(start: int, end: int) -> float:
        """Squared deviation from ideal line length."""
        line_len = plen[end] - plen[start]
        dev = line_len - ideal_len
        return dev * dev

    NEG_INF = float("inf")  # we're minimizing
    dp = [[NEG_INF] * (m + 1) for _ in range(n)]
    parent = [[0] * (m + 1) for _ in range(n)]

    min_w = 1

    # First line
    for j in range(min_w, m - min_w * (n - 1) + 1):
        dp[0][j] = badness(0, j)

    for i in range(1, n):
        remaining = n - 1 - i
        for j in range(min_w * (i + 1), m - min_w * remaining + 1):
            k_min = min_w * i
            k_max = j - min_w
            for k in range(k_min, k_max + 1):
                if dp[i - 1][k] == NEG_INF:
                    continue
                val = dp[i - 1][k] + badness(k, j)
                if val < dp[i][j]:  # minimizing
                    dp[i][j] = val
                    parent[i][j] = k

    cuts = [0] * (n + 1)
    cuts[n] = m
    j = m
    for i in range(n - 1, 0, -1):
        cuts[i] = parent[i][j]
        j = cuts[i]

    return cuts


def redistribute(
    ag_lines: list[int],
    words: list[str],
    word_to_ag: dict[int, list[int]],
    bonds: set[int] | None = None,
) -> list[int]:
    """Find optimal cut points for distributing words across AG lines.

    Args:
        ag_lines: Sorted list of AG line numbers.
        words: All translation words in reading order.
        word_to_ag: Maps word index -> list of AG line numbers it aligns to.
        bonds: Set of word indices where cutting between word[i] and word[i+1]
               is penalized. Used to keep syntactic groups together (det+noun,
               prep+noun). Built from dependency parsing.

    Returns:
        List of N+1 cut points where ag_lines[i] gets words[cuts[i]:cuts[i+1]].
    """
    n = len(ag_lines)
    m = len(words)

    if n == 0:
        return [0, m]
    if n == 1:
        return [0, m]
    if m == 0:
        return [0] * (n + 1)

    line_idx = {ln: i for i, ln in enumerate(ag_lines)}

    # Precompute per-word score for each AG line assignment.
    # +2 for correct line, -1 for wrong line, 0 for unaligned.
    score = []
    for w in range(m):
        aligned = word_to_ag.get(w, [])
        indices = {line_idx[ln] for ln in aligned if ln in line_idx}
        ws = [0.0] * n
        if indices:
            for i in range(n):
                ws[i] = 2.0 if i in indices else -1.0
        score.append(ws)

    # Prefix sums for fast span scoring
    prefix = [[0.0] * (m + 1) for _ in range(n)]
    for i in range(n):
        for j in range(m):
            prefix[i][j + 1] = prefix[i][j] + score[j][i]

    # Line balance penalty: penalize lines that deviate from ideal length.
    # This prevents the alignment DP from creating 1-word lines.
    ideal = m / n

    # Bond penalty: penalize cuts that split syntactic groups.
    # A cut at position k splits words[k-1] from words[k].
    # If k-1 is in bonds, add a penalty to discourage this split.
    bond_set = bonds or set()
    BOND_PENALTY = -5.0

    MIN_WORDS = 2  # soft minimum words per line
    SHORT_PENALTY = -4.0  # per word below minimum

    def span_score(line_idx: int, start: int, end: int) -> float:
        align = prefix[line_idx][end] - prefix[line_idx][start]
        num_words = end - start
        if num_words == 0:
            return align - 10.0
        deviation = (num_words - ideal) / max(ideal, 1)
        balance = -0.3 * deviation * deviation * ideal
        # Penalize very short lines (1-word orphans)
        short = SHORT_PENALTY * max(0, MIN_WORDS - num_words) if num_words < MIN_WORDS else 0.0
        # Penalize if this span starts by breaking a bond
        bond = BOND_PENALTY if start > 0 and (start - 1) in bond_set else 0.0
        return align + balance + short + bond

    # DP
    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (m + 1) for _ in range(n)]
    parent = [[0] * (m + 1) for _ in range(n)]

    # Enforce minimum 1 word per line when we have enough words
    min_w = 1 if m >= n else 0

    # First line gets words[0:j], must have at least min_w words
    # and leave enough for remaining lines
    for j in range(min_w, m - min_w * (n - 1) + 1):
        dp[0][j] = span_score(0, 0, j)

    for i in range(1, n):
        remaining = n - 1 - i  # lines AFTER this one
        for j in range(min_w * (i + 1), m - min_w * remaining + 1):
            # k = start of this line's span; previous lines used words[0:k]
            # k must be >= min_w*i (previous i lines each got at least min_w)
            # j-k must be >= min_w (this line gets at least min_w)
            k_min = max(min_w * i, j - (m - min_w * remaining))  # don't take too many
            k_max = j - min_w  # leave at least min_w for this line
            for k in range(k_min, k_max + 1):
                prev = dp[i - 1][k]
                if prev == NEG_INF:
                    continue
                val = prev + span_score(i, k, j)
                if val > dp[i][j]:
                    dp[i][j] = val
                    parent[i][j] = k

    # Backtrack
    cuts = [0] * (n + 1)
    cuts[n] = m
    j = m
    for i in range(n - 1, 0, -1):
        cuts[i] = parent[i][j]
        j = cuts[i]

    return cuts


def build_word_to_ag(
    ag_lines: list[int],
    words: list[str],
    alignments: list[dict],
    translation_by_line: dict[int, str],
    word_offsets: dict[int, int],
    src_line_key: str = "ag_line",
    tgt_line_key: str = "en_line",
    tgt_idx_key: str = "en_word_idx",
) -> dict[int, list[int]]:
    """Build word_to_ag mapping handling both per-line and sentence-relative indices.

    Tries to match each alignment record's en_word text against the
    concatenated words array by position, with fallback to text matching.

    Returns {flat_word_index: [ag_line_numbers]}.
    """
    import re

    _SKIP = {"ᴹᵁᴿ", "⟨LEAF⟩"}

    # Build clean-token-to-flat mapping for each possible starting line.
    # alignments.json uses Crane-sentence-relative clean indices where
    # en_word_idx counts clean tokens from en_line forward.
    # For each starting line, build the sequence of clean tokens and
    # map their indices to flat positions in the words array.
    clean_maps: dict[int, dict[int, int]] = {}
    ag_set = set(ag_lines)
    all_tgt_lines = sorted(translation_by_line.keys())

    for start_ln in all_tgt_lines:
        if start_ln not in word_offsets:
            continue
        mapping = {}
        ci = 0
        for ln in all_tgt_lines:
            if ln < start_ln:
                continue
            if ln not in word_offsets:
                continue
            text = translation_by_line.get(ln, "")
            base = word_offsets[ln]
            for ri, w in enumerate(text.split()):
                if w in _SKIP:
                    continue
                cleaned = re.sub(r"^[^\w']+|[^\w']+$", "", w)
                if cleaned:
                    mapping[ci] = base + ri
                    ci += 1
        clean_maps[start_ln] = mapping

    # Also build per-line raw index mapping (for MG alignments which
    # use per-line raw indices)
    per_line_raw: dict[int, dict[int, int]] = {}
    for ln in all_tgt_lines:
        if ln not in word_offsets:
            continue
        base = word_offsets[ln]
        text = translation_by_line.get(ln, "")
        nwords = len(text.split()) if text else 0
        per_line_raw[ln] = {i: base + i for i in range(nwords)}

    # Map alignments
    word_to_ag: dict[int, list[int]] = {}
    for r in alignments:
        src_ln = r[src_line_key]
        tgt_ln = r[tgt_line_key]
        tgt_idx = r[tgt_idx_key]

        # Try sentence-relative clean mapping first
        pos = clean_maps.get(tgt_ln, {}).get(tgt_idx)

        # Fallback: per-line raw index
        if pos is None:
            pos = per_line_raw.get(tgt_ln, {}).get(tgt_idx)

        if pos is not None and 0 <= pos < len(words):
            word_to_ag.setdefault(pos, []).append(src_ln)

    return word_to_ag


def build_bonds(words: list[str], dp_results: list | None = None) -> set[int]:
    """Build syntactic bond set from dependency parsing.

    Returns a set of word indices i where words[i] and words[i+1] should
    not be split across lines (e.g. article+noun, preposition+noun).

    Uses gr-nlp-toolkit if available, falls back to simple heuristics.
    """
    bonds: set[int] = set()

    if dp_results:
        # dp_results is a list of (head_idx, deprel) per word (0-indexed)
        # Bond types:
        # 1. Adjacent head-dependent pairs (det+noun, case+noun, etc.)
        # 2. Single-word orphans: if a word's head is far away and it would
        #    be alone on a line, bond it to its neighbor. Catches vocatives
        #    like «Ω where the head is several words later.
        adjacent_deprels = {"det", "case", "amod", "nmod", "nummod", "flat",
                            "cc", "mark"}
        for i, (head_idx, deprel) in enumerate(dp_results):
            if deprel in adjacent_deprels:
                if head_idx == i + 1:
                    bonds.add(i)
                elif head_idx == i - 1 and i > 0:
                    bonds.add(i - 1)
                elif head_idx > i + 1 and i < len(words) - 1:
                    # Head is further right — bond to next word to prevent
                    # orphaning this word on its own line
                    bonds.add(i)
        return bonds

    # Fallback: simple heuristic - bond Greek articles/prepositions to next word
    import re
    _ARTICLES = {"ο", "η", "το", "τα", "οι", "των", "τον", "την", "τις",
                 "τους", "του", "της", "στο", "στη", "στον", "στην", "στα",
                 "στου", "στης", "στους", "στων", "στις"}
    for i, w in enumerate(words[:-1]):
        clean = re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower()
        if clean in _ARTICLES:
            bonds.add(i)

    return bonds


_polylas_tags: dict | None = None


def _load_polylas_tags():
    """Load polylas_tags.json once."""
    global _polylas_tags
    if _polylas_tags is not None:
        return
    import json
    from pathlib import Path
    tags_path = Path(__file__).resolve().parent / "output" / "polylas_tags.json"
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            _polylas_tags = json.load(f)
    else:
        _polylas_tags = {}


def _get_dp_results(
    ag_lines: list[int],
    words: list[str],
    translation_by_line: dict[int, str],
    word_offsets: dict[int, int],
) -> list | None:
    """Build dp_results list from polylas_tags.json for the given word stream.

    The word stream is the concatenated translation words (already flattened
    by redistribute_passage). We match each word to its DP tag by stripped
    form, building a flat list of (head_flat_idx, deprel) tuples.

    Returns [(head_idx, deprel), ...] per word in the flattened stream,
    or None if tags aren't available.
    """
    _load_polylas_tags()
    if not _polylas_tags:
        return None

    import re
    import unicodedata

    def strip(s):
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", s, flags=re.UNICODE)
        nfd = unicodedata.normalize("NFD", cleaned.lower())
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    # Collect all tag tokens for lines in this passage, in order.
    # Tags are from the ORIGINAL text (pre-redistribution), so we
    # concatenate them across all lines to match the flattened word stream.
    tags_by_line = {}
    for book_tags in _polylas_tags.values():
        for entry in book_tags:
            if entry["line"] in set(ag_lines):
                tags_by_line[entry["line"]] = entry["tokens"]
        if tags_by_line:
            break

    if not tags_by_line:
        return None

    # Build flat tag stream (all tokens from all lines in order)
    flat_tags = []
    for ln in ag_lines:
        for t in tags_by_line.get(ln, []):
            flat_tags.append(t)

    # Match words to flat_tags by stripped form using a greedy scan
    tag_idx = 0
    word_to_tag = {}  # word_flat_idx -> tag_flat_idx
    for w_idx, word in enumerate(words):
        w_stripped = strip(word)
        # Scan forward in tags to find a match
        for scan in range(tag_idx, min(tag_idx + 3, len(flat_tags))):
            if strip(flat_tags[scan].get("form", "")) == w_stripped:
                word_to_tag[w_idx] = scan
                tag_idx = scan + 1
                break

    # Build results: for each word, find its head in the flat word stream
    tag_to_word = {v: k for k, v in word_to_tag.items()}
    results = []
    for w_idx in range(len(words)):
        t_idx = word_to_tag.get(w_idx)
        if t_idx is None:
            results.append((w_idx, ""))  # no tag: self-reference
            continue

        t = flat_tags[t_idx]
        deprel = t.get("deprel", "")
        head_tok = t.get("head")

        if head_tok is not None and isinstance(head_tok, int) and head_tok > 0:
            # head is 1-indexed within the LINE's token list.
            # Find which line this tag belongs to, compute line-local offset,
            # then map back to flat tag index.
            line_start = 0
            for ln in ag_lines:
                line_tokens = tags_by_line.get(ln, [])
                if t_idx < line_start + len(line_tokens):
                    # This tag is in this line
                    head_tag_idx = line_start + head_tok - 1
                    head_word = tag_to_word.get(head_tag_idx, w_idx)
                    results.append((head_word, deprel))
                    break
                line_start += len(line_tokens)
            else:
                results.append((w_idx, deprel))
        else:
            results.append((w_idx, deprel))  # root or missing head

    return results if results else None


def redistribute_passage(
    ag_lines: list[int],
    translation_by_line: dict[int, str],
    alignments: list[dict],
    src_line_key: str = "ag_line",
    tgt_line_key: str = "en_line",
    tgt_idx_key: str = "en_word_idx",
) -> dict[int, str]:
    """Redistribute translation text across AG lines using DP.

    Concatenates all translation words, builds alignment mapping,
    runs DP to find optimal cuts, returns new text per line.
    """
    if len(ag_lines) <= 1:
        if ag_lines:
            return {ag_lines[0]: translation_by_line.get(ag_lines[0], "")}
        return {}

    # Concatenate words in AG line order
    words: list[str] = []
    word_offsets: dict[int, int] = {}
    for ln in ag_lines:
        word_offsets[ln] = len(words)
        text = translation_by_line.get(ln, "")
        if text:
            words.extend(text.split())

    if not words:
        return {ln: "" for ln in ag_lines}

    # Build alignment mapping
    word_to_ag = build_word_to_ag(
        ag_lines, words, alignments, translation_by_line, word_offsets,
        src_line_key, tgt_line_key, tgt_idx_key,
    )

    # Build syntactic bonds from dependency parsing if available
    dp_results = _get_dp_results(ag_lines, words, translation_by_line, word_offsets)
    bonds = build_bonds(words, dp_results=dp_results)

    # Run alignment-driven DP
    cuts = redistribute(ag_lines, words, word_to_ag, bonds=bonds)

    # Build result
    return {
        ag_lines[i]: " ".join(words[cuts[i]:cuts[i + 1]])
        for i in range(len(ag_lines))
    }
