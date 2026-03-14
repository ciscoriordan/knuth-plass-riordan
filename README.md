# Alignment-Driven Line Breaking for Parallel Texts

> A novel adaptation of the Knuth-Plass line-breaking algorithm for
> redistributing prose translations to match the line structure of verse
> originals, using word-level alignment scores as the optimization objective.

---

## The Problem

Parallel text displays of verse originals with prose translations face a fundamental layout problem: the original has fixed line breaks (verse structure), but the translation is continuous prose. Naive approaches split the prose proportionally by word count, which often places translation words on the wrong line relative to the words they translate.

For example, Homer's Iliad in Ancient Greek has fixed verse lines. Murray's English prose translation and Polylas's Modern Greek prose translation must be split across those lines for a three-column parallel display. When the prose says *"dog-face, from the Trojans. This you disregard,"* the phrase *"from the Trojans"* may end up on the wrong line if the split point is chosen by word count rather than by alignment.

---

## Prior Art

### Crane Sentence Alignment

The standard approach in digital classics uses hand-curated sentence-level alignment data. Gregory Crane's [Beyond Translation](https://github.com/scaife-viewer/beyond-translation-site) project provides gold-standard sentence boundaries mapping Murray's English sentences to Ancient Greek line ranges.

Crane's data is accurate for what it provides, but it operates at the sentence level only. It tells you *which lines form a sentence block*, not *where to split translation words within that block*.

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

### Knuth-Plass Line Breaking

Knuth and Plass (1981) formulated typographic line breaking as a dynamic programming problem: given a paragraph of words and a target line width, find break points that minimize a global "badness" function penalizing overfull and underfull lines.

The key insight is that locally greedy line breaking produces suboptimal results, while the DP finds globally optimal break points by considering all possible splits simultaneously.

> D.E. Knuth and M.F. Plass, "Breaking paragraphs into lines," *Software: Practice and Experience*, 11(11):1119-1184, 1981.

### Word-Level Alignment Models

Neural word alignment models like [SimAlign](https://github.com/cisnlp/simalign) (Jalili Sabet et al., 2020) and AccAlign (Wang et al., 2022) use multilingual transformer embeddings to produce word-level correspondences between parallel texts. The [UGARIT](https://huggingface.co/UGARIT/grc-alignment) project (Yousef et al., 2022) fine-tuned XLM-RoBERTa specifically for Ancient Greek alignment.

---

## This Work: Alignment-Driven Line Breaking

We adapt the Knuth-Plass formulation by replacing the visual spacing objective with a cross-lingual alignment objective. Instead of minimizing badness based on line width, we maximize the total alignment score of words assigned to their correct source lines.

Sentence-level alignment (Crane) and word-level alignment (SimAlign/UGARIT) solve different problems. Crane answers *"which lines form a block?"* The alignment model answers *"which words correspond?"* Neither answers the question this algorithm addresses: ***"given a block of translation words and the source lines they align to, where should we cut?"***

### Formulation

Given:
- **N** source (verse) lines with fixed boundaries
- **M** translation words in reading order
- A word-to-line alignment function **a(w, l)** giving the alignment score of assigning translation word *w* to source line *l*

Find cut points $c_0 = 0 < c_1 < \cdots < c_N = M$ that maximize:

$$\sum_{i=0}^{N-1} \sum_{w=c_i}^{c_{i+1}-1} a(w, \text{line}_i)$$

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

## License

MIT License. See [LICENSE](LICENSE).
