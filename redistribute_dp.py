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


def redistribute(
    ag_lines: list[int],
    words: list[str],
    word_to_ag: dict[int, list[int]],
) -> list[int]:
    """Find optimal cut points for distributing words across AG lines.

    Args:
        ag_lines: Sorted list of AG line numbers.
        words: All translation words in reading order.
        word_to_ag: Maps word index -> list of AG line numbers it aligns to.

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

    def span_score(line_idx: int, start: int, end: int) -> float:
        return prefix[line_idx][end] - prefix[line_idx][start]

    # DP
    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (m + 1) for _ in range(n)]
    parent = [[0] * (m + 1) for _ in range(n)]

    # First line gets words[0:j]
    for j in range(m + 1):
        dp[0][j] = span_score(0, 0, j)

    for i in range(1, n):
        for j in range(m + 1):
            for k in range(j + 1):
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

    # Run DP
    cuts = redistribute(ag_lines, words, word_to_ag)

    # Build result
    return {
        ag_lines[i]: " ".join(words[cuts[i]:cuts[i + 1]])
        for i in range(len(ag_lines))
    }
