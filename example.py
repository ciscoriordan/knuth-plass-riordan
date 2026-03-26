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


# --- Modern Greek (Polylas) example ---
# Polylas's 1875 verse translation, treated as continuous prose.
# Original Polylas has line breaks matching AG, but a prose MG translation
# (or one reformatted as continuous text) would need redistribution.
print("\n\n--- Modern Greek (Polylas, 1875) ---\n")

mg_words = (
    "ἀλλὰ γιὰ τὸν Μενέλαο καί, ἀναίσχυντε, γιὰ σένα "
    "ἡλθομεν ὅλοι ἐκδίκησιν νὰ πάρωμε τῶν Τρώων, "
    "καὶ σύ, ὦ σκυλοπρόσωπε, λησμονημένα τά ᾽χεις."
).split()

# AG-MG word alignments
mg_word_to_ag = {
    0: [158],   # ἀλλὰ -> ἀλλὰ
    1: [159],   # γιὰ -> σοί
    2: [159],   # τὸν -> Μενελάῳ
    3: [159],   # Μενέλαο -> Μενελάῳ
    # 4: καί, -> (unaligned)
    5: [158],   # ἀναίσχυντε, -> ἀναιδὲς
    6: [158],   # γιὰ -> σοὶ
    7: [158],   # σένα -> σοὶ
    8: [158],   # ἡλθομεν -> ἑσπόμεθʼ
    9: [158],   # ὅλοι -> ἅμʼ
    10: [159],  # ἐκδίκησιν -> τιμὴν
    11: [159],  # νὰ -> ἀρνύμενοι
    12: [159],  # πάρωμε -> ἀρνύμενοι
    13: [160],  # τῶν -> Τρώων
    14: [160],  # Τρώων, -> Τρώων
    # 15: καὶ -> (unaligned)
    16: [158],  # σύ, -> σὺ
    # 17: ὦ -> (unaligned)
    18: [159],  # σκυλοπρόσωπε, -> κυνῶπα
    19: [160],  # λησμονημένα -> μετατρέπῃ
    20: [160],  # τά -> οὔ τι
    21: [160],  # ᾽χεις. -> ἀλεγίζεις
}

# --- Naive proportional split ---
mg_m = len(mg_words)
mg_prop = mg_m // n
mg_naive_cuts = [0, mg_prop, 2 * mg_prop, mg_m]

print("=== Naive proportional split (MG) ===")
for i, ln in enumerate(ag_lines):
    s, e = mg_naive_cuts[i], mg_naive_cuts[i + 1]
    print(f"  AG {ln}: {' '.join(mg_words[s:e])}")

# --- DP-optimized split ---
mg_cuts = redistribute(ag_lines, mg_words, mg_word_to_ag)

print("\n=== Alignment-driven DP split (MG) ===")
for i, ln in enumerate(ag_lines):
    s, e = mg_cuts[i], mg_cuts[i + 1]
    print(f"  AG {ln}: {' '.join(mg_words[s:e])}")

print("\n=== Original Polylas line breaks (reference) ===")
mg_ref = {
    158: "ἀλλὰ γιὰ τὸν Μενέλαο καί, ἀναίσχυντε, γιὰ σένα",
    159: "ἡλθομεν ὅλοι ἐκδίκησιν νὰ πάρωμε τῶν Τρώων,",
    160: "καὶ σύ, ὦ σκυλοπρόσωπε, λησμονημένα τά ᾽χεις.",
}
for ln in ag_lines:
    print(f"  AG {ln}: {mg_ref[ln]}")
