#!/usr/bin/env python3
"""
Example: redistribute Iliad 1.158-160 using the alignment-driven DP.

Demonstrates the difference between naive proportional splitting and
DP-optimized splitting driven by word-level alignment scores.
"""

from redistribute_dp import redistribute

# Ancient Greek verse lines (fixed)
ag_lines = [158, 159, 160]
ag_text = {
    158: "ἀλλὰ σοὶ ὦ μέγʼ ἀναιδὲς ἅμʼ ἑσπόμεθʼ ὄφρα σὺ χαίρῃς,",
    159: "τιμὴν ἀρνύμενοι Μενελάῳ σοί τε κυνῶπα",
    160: "πρὸς Τρώων· τῶν οὔ τι μετατρέπῃ οὐδʼ ἀλεγίζεις·",
}

# English prose translation (continuous)
en_words = (
    "But you, shameless one, we followed, so that you might "
    "rejoice, seeking to win recompense for Menelaus and for "
    "yourself, dog-face, from the Trojans. This you disregard, "
    "and take no heed of."
).split()

# Word-level alignments from the UGARIT/grc-alignment model.
# Maps word index -> list of AG line numbers the word aligns to.
word_to_ag = {
    0: [158],   # But -> ἀλλὰ
    1: [158],   # you, -> σοὶ
    2: [158],   # shameless -> ἀναιδὲς
    3: [158],   # one, -> μέγʼ
    4: [158],   # we -> ἑσπόμεθʼ
    5: [158],   # followed, -> ἑσπόμεθʼ
    6: [158],   # so -> ὄφρα
    7: [158],   # that -> ὄφρα
    8: [158],   # you -> σὺ
    9: [158],   # might -> χαίρῃς
    # 10: rejoice, -> (unaligned - cross-boundary)
    11: [159],  # seeking -> ἀρνύμενοι
    12: [159],  # to -> ἀρνύμενοι
    13: [159],  # win -> ἀρνύμενοι
    14: [159],  # recompense -> τιμὴν
    15: [159],  # for -> Μενελάῳ
    16: [159],  # Menelaus -> Μενελάῳ
    17: [159],  # and -> τε
    18: [159],  # for -> σοί
    19: [159],  # yourself, -> σοί
    20: [159],  # dog-face, -> κυνῶπα
    21: [160],  # from -> πρὸς
    # 22: the -> (unaligned article)
    23: [160],  # Trojans. -> Τρώων
    24: [160],  # This -> τῶν
    25: [160],  # you -> μετατρέπῃ
    26: [160],  # disregard, -> μετατρέπῃ
    27: [160],  # and -> ἀλεγίζεις
    28: [160],  # take -> ἀλεγίζεις
    29: [160],  # no -> οὔ
    30: [160],  # heed -> ἀλεγίζεις
    # 31: of. -> (unaligned)
}

# --- Naive proportional split ---
n = len(ag_lines)
m = len(en_words)
prop_size = m // n
naive_cuts = [0, prop_size, 2 * prop_size, m]

print("=== Naive proportional split ===")
for i, ln in enumerate(ag_lines):
    s, e = naive_cuts[i], naive_cuts[i + 1]
    print(f"  AG {ln}: {' '.join(en_words[s:e])}")

# --- DP-optimized split ---
cuts = redistribute(ag_lines, en_words, word_to_ag)

print("\n=== Alignment-driven DP split ===")
for i, ln in enumerate(ag_lines):
    s, e = cuts[i], cuts[i + 1]
    print(f"  AG {ln}: {' '.join(en_words[s:e])}")

print("\n=== Ancient Greek (reference) ===")
for ln in ag_lines:
    print(f"  AG {ln}: {ag_text[ln]}")
