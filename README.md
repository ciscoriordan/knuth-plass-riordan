# Alignment-Driven Line Breaking for Parallel Texts

> A novel adaptation of the Knuth-Plass line-breaking algorithm for
> redistributing prose translations to match the line structure of verse
> originals, using word-level alignment scores as the optimization objective.

## Table of Contents

- [The Problem](#the-problem)
- [Prior Art](#prior-art)
- [This Work: Alignment-Driven Line Breaking](#this-work-alignment-driven-line-breaking)
  - [Formulation](#formulation)
  - [Worked Example: Murray EN](#worked-example-iliad-1158-160)
  - [Worked Example: Polylas MG](#modern-greek-example-polylas-1875)
- [Extended Version: Two-Pass with Visual Balance](#extended-version-two-pass-with-visual-balance)
- [Usage](#usage)
- [References](#references)
- [Related Repos](#related-repos)
- [How to Cite](#how-to-cite)
- [License](#license)

---

## The Problem

Parallel text displays of verse originals with prose translations face a fundamental layout problem: the original has fixed line breaks (verse structure), but the translation is continuous prose. Naive approaches split the prose proportionally by word count, which often places translation words on the wrong line relative to the words they translate.

For example, Homer's Iliad in Ancient Greek has fixed verse lines. Murray's English prose translation and Polylas's Modern Greek prose translation must be split across those lines for a three-column parallel display. When the prose says *"dog-face, from the Trojans. This you disregard,"* the phrase *"from the Trojans"* may end up on the wrong line if the split point is chosen by word count rather than by alignment.

---

## Prior Art

**Crane sentence alignment.** The standard approach in digital classics uses hand-curated sentence-level alignment data. Gregory Crane's [Beyond Translation](https://github.com/scaife-viewer/beyond-translation-site) project provides gold-standard sentence boundaries mapping Murray's English sentences to Ancient Greek line ranges. This tells you *which lines form a sentence block*, not *where to split translation words within that block*.

<details>
<summary><strong>Example: Iliad 1.158-160</strong></summary>

Crane correctly identifies AG lines 158-160 as one sentence:

```
AG 158: ἀλλὰ σοὶ ὦ μέγʼ ἀναιδὲς ἅμʼ ἑσπόμεθʼ ὄφρα σὺ χαίρῃς,
AG 159: τιμὴν ἀρνύμενοι Μενελάῳ σοί τε κυνῶπα
AG 160: πρὸς Τρώων· τῶν οὔ τι μετατρέπῃ οὐδʼ ἀλεγίζεις·
```

The EN prose for this block is:

> *"But you, shameless one, we followed, so that you might rejoice, seeking to win recompense for Menelaus and for yourself, dog-face, from the Trojans. This you disregard, and take no heed of."*

Crane says these ~30 English words belong to AG lines 158-160. It does not say how to distribute them across the three lines. That distribution problem - assigning *"from the Trojans"* to line 160 (where `πρὸς Τρώων` lives) rather than leaving it on line 159 - is the gap this work fills.

</details>

**Knuth-Plass line breaking.** Knuth and Plass (1981) formulated typographic line breaking as a dynamic programming problem: given a paragraph of words and a target line width, find break points that minimize a global "badness" function. The key insight is that locally greedy line breaking produces suboptimal results, while the DP finds globally optimal break points by considering all possible splits simultaneously.

**Word-level alignment models.** Neural word alignment models like [SimAlign](https://github.com/cisnlp/simalign) (Jalili Sabet et al., 2020) and the [UGARIT](https://huggingface.co/UGARIT/grc-alignment) project (Yousef et al., 2022) use multilingual transformer embeddings to produce word-level correspondences between parallel texts.

---

## This Work: Alignment-Driven Line Breaking

We adapt the Knuth-Plass formulation by replacing the visual spacing objective with a cross-lingual alignment objective. Instead of minimizing badness based on line width, we maximize the total alignment score of words assigned to their correct source lines.

Sentence-level alignment (Crane) and word-level alignment (SimAlign/UGARIT) solve different problems. Crane answers *"which lines form a block?"* The alignment model answers *"which words correspond?"* Neither answers the question this algorithm addresses: ***"given a block of translation words and the source lines they align to, where should we cut?"***

### Formulation

Given:
- **N** source (verse) lines with fixed boundaries
- **M** translation words in reading order
- A word-to-line alignment function **a(w, l)** giving the alignment score of assigning translation word *w* to source line *l*

Find cut points $c_0 = 0 < c_1 < \cdots < c_N = M$ that solve:

$$\max_{c_0,\ldots,c_N}\;\sum_{i=0}^{N-1} \sum_{w=c_i}^{c_{i+1}-1} a(w, \text{line}_i) \quad c_0=0,\; c_N=M$$

subject to the monotonicity constraint (translation word order is preserved).

### Score Function

| Condition | Score |
|:----------|------:|
| Word is aligned to the assigned line | **+2** |
| Word is aligned to a *different* line | **-1** |
| Word is unaligned | **0** |

The asymmetric weighting (+2/-1) biases toward keeping aligned content together: correctly placed words are valued more than misplaced words are penalized.

### Algorithm

```python
# dp[i][j] = max total score for assigning words[0:j] to lines[0:i+1]

# Base case: first line gets words[0:j]
dp[0][j] = span_score(line_0, 0, j)

# Transition
dp[i][j] = max over k of: dp[i-1][k] + span_score(line_i, k, j)

# Answer
dp[N-1][M]
```

`span_score(l, a, b)` is the sum of `a(w, l)` for `w` in `[a, b)`, computed in **O(1)** using prefix sums.

> **Complexity:** O(N * M<sup>2</sup>) time, O(N * M) space.
> In practice, N ~ 7 and M ~ 50, so the DP completes in microseconds per passage.

### Comparison

| | Proportional / median heuristic | Alignment-Driven DP |
|:---|:---|:---|
| **Split method** | Divide words evenly, or by median position | Globally optimal via dynamic programming |
| **"from the Trojans" on 1.160?** | :x: stays on 1.159 | :white_check_mark: DP sees alignment to AG 160 |
| **Optimality** | Local heuristic, no guarantees | Globally optimal within block |
| **Edge cases** | Requires ad hoc post-passes | Handled naturally by the cost function |

### Worked Example: Iliad 1.158-160

**Ancient Greek** (fixed verse lines):
```
158: ἀλλὰ σοὶ ὦ μέγʼ ἀναιδὲς ἅμʼ ἑσπόμεθʼ ὄφρα σὺ χαίρῃς,
159: τιμὴν ἀρνύμενοι Μενελάῳ σοί τε κυνῶπα
160: πρὸς Τρώων· τῶν οὔ τι μετατρέπῃ οὐδʼ ἀλεγίζεις·
```

**Proportional split** (naive):
```
159: ...for yourself, dog-face, from the Trojans.
                                ^^^^^^^^^^^^^^^^^
160: This you disregard, and take no heed of.
```
*"from the Trojans"* is on line 159, but `πρὸς Τρώων` is on AG line 160.

**DP redistribution:**
```
159: ...for yourself, dog-face,
160: from the Trojans. This you disregard, and take no heed
     ^^^^^^^^^^^^^^^^^^
```
*"from the Trojans"* moves to line 160 where `πρὸς Τρώων` lives. The DP sees that *"from"* and *"Trojans"* both align to AG 160 and cuts accordingly.

### Modern Greek Example: Polylas (1875)

The algorithm works across any language pair. Here it redistributes Polylas's Modern Greek verse translation (concatenated into prose) across the same AG lines:

**Proportional split** (naive):
```
158: ἀλλὰ γιὰ τὸν Μενέλαο καί, ἀναίσχυντε, γιὰ
159: σένα ἡλθομεν ὅλοι ἐκδίκησιν νὰ πάρωμε τῶν
160: Τρώων, καὶ σύ, ὦ σκυλοπρόσωπε, λησμονημένα τά ᾽χεις.
```
*"τῶν Τρώων"* is split across lines 159-160.

**DP redistribution:**
```
158: ἀλλὰ γιὰ τὸν Μενέλαο καί, ἀναίσχυντε, γιὰ σένα ἡλθομεν ὅλοι
159: ἐκδίκησιν νὰ πάρωμε
160: τῶν Τρώων, καὶ σύ, ὦ σκυλοπρόσωπε, λησμονημένα τά ᾽χεις.
     ^^^^^^^^^^
```
*"τῶν Τρώων"* moves to line 160 where `πρὸς Τρώων` lives. The trade-off is uneven line lengths - line 159 is short. The [extended version](#extended-version-two-pass-with-visual-balance) addresses this.

---

## Extended Version: Two-Pass with Visual Balance

The core algorithm optimizes purely for alignment score, which can produce very uneven line lengths (as in the MG example above). The extended version adds a Knuth-Plass visual balance pass and several penalty terms.

### Pass 1: Knuth-Plass visual balance

A classic Knuth-Plass pass distributes words as evenly as possible across the available lines, minimizing the sum of squared deviations from the ideal line length (total characters / number of lines). This produces a visually balanced baseline.

### Pass 2: Alignment-driven DP with balance penalty

The alignment DP's `span_score` is augmented with a balance term that penalizes deviating from the Pass 1 baseline:

```python
def span_score(line_idx, start, end):
    align   = prefix[line_idx][end] - prefix[line_idx][start]
    num     = end - start
    dev     = (num - ideal) / max(ideal, 1)
    balance = -0.3 * dev * dev * ideal
    short   = SHORT_PENALTY * max(0, MIN_WORDS - num) if num < MIN_WORDS else 0
    bond    = BOND_PENALTY if start > 0 and (start - 1) in bond_set else 0
    return align + balance + short + bond
```

This balances alignment accuracy against visual evenness, preventing the DP from creating 1-word orphan lines just to score a single alignment.

### Short-line penalties

Lines with fewer than `MIN_WORDS` (default 2) words receive an additional penalty. This discourages orphan lines like:

```
159: πάρωμε
160: τῶν Τρώων, καὶ σύ...
```

### Syntactic bonding

When dependency parse data is available (e.g., from [Opla](https://github.com/ciscoriordan/opla)), the algorithm adds a penalty for cutting between syntactically bonded words. Bond types by dependency relation:

| Relation | Example | Bond strength |
|----------|---------|:---:|
| `det` (determiner) | τὸν Μενέλαο | high |
| `case` (preposition) | πρὸς Τρώων | high |
| `mark` (subordinator) | νὰ πάρωμε | high |
| `amod` (adjective) | θεῖος Ἀχιλλέας | medium |
| `cc` (conjunction) | καὶ σύ | medium |

This prevents splits like `νὰ / πάρωμε` where the subjunctive particle and verb should stay together. Bonds are weighted so strong syntactic dependencies (det, case, mark) are almost never broken, while weaker ones (cc, amod) yield to alignment when needed.

The extended version with all features is implemented in [`redistribute_dp.py`](redistribute_dp.py).

---

## Usage

```python
from redistribute_dp import redistribute, redistribute_passage

# Low-level: find optimal cuts for a word sequence
cuts = redistribute(
    ag_lines=[158, 159, 160],
    words=["But", "you", ..., "Trojans.", "This", ...],
    word_to_ag={0: [158], 1: [158], ..., 23: [160], ...},
)

# With syntactic bonds (MG example)
from redistribute_dp import build_bonds
bonds = build_bonds(mg_words, dp_results=dep_parse_output)
cuts = redistribute(ag_lines, mg_words, mg_word_to_ag, bonds=bonds)

# High-level: redistribute a full passage
result = redistribute_passage(
    ag_lines=[158, 159, 160],
    translation_by_line={158: "But you, ...", 159: "to win ...", 160: "This ..."},
    alignments=[{"ag_line": 158, "en_line": 158, "en_word_idx": 0}, ...],
)
# result = {158: "But you, ...", 159: "win recompense ...", 160: "Trojans. This ..."}
```

Run the included example:

```bash
python example.py
```

---

## References

1. D.E. Knuth and M.F. Plass, "Breaking paragraphs into lines," *Software: Practice and Experience*, 11(11):1119-1184, 1981.
2. M. Jalili Sabet, P. Dufter, F. Yvon, and H. Schutze, "[SimAlign: High quality word alignments without parallel training data using static and contextualized embeddings](https://aclanthology.org/2020.findings-emnlp.147/)," *Findings of EMNLP*, 2020.
3. T. Yousef, C. Palladino, F. Shamsian, A. Meeus, and G.C. Crane, "[Translation alignment with Ugarit](https://www.mdpi.com/2078-2489/13/2/65)," *Information*, 13(2):65, 2022.
4. G.R. Crane et al., [Beyond Translation](https://github.com/scaife-viewer/beyond-translation-site), Scaife Viewer project.

## Related Repos

- **[Opla](https://github.com/ciscoriordan/opla)** - GPU-optimized Greek POS tagger + dependency parser (provides syntactic bonds)
- **[Dilemma](https://github.com/ciscoriordan/dilemma)** - Greek lemmatizer (used by the alignment pipeline that produces word-to-line mappings)

## How to Cite

```
Francisco Riordan, "Alignment-Driven Line Breaking for Parallel Texts" (2026).
https://github.com/ciscoriordan/knuth-plass-riordan
```

## License

MIT License. See [LICENSE](LICENSE).
