"""
semantic_directory.py

The upgraded Module 2: semantic (TF-IDF + keyword) matching, ported from
the teammate's Memory.ipynb TransactiveSharedMemory class (its second,
final version -- not the earlier dead draft in the same notebook), adapted
to this project's SharedMemoryDirectory public shape (lookup/record/stats)
so qa_pipeline.py's calling convention stays familiar.

Differs from shared_memory.py's SharedMemoryDirectory in one necessary way:
that class is keyed on a SHA256 hash of (stage, input) -- fine for exact
match, useless for semantic similarity, which needs the actual words. This
class stores the raw text instead, and keeps ONE INDEPENDENT SIMILARITY
INDEX PER STAGE -- the notebook's original has a single global memory
(single-agent chat cache); our pipeline has four stages with different
prompt content built from different upstream context, so an Extractor's
stored answer must never be allowed to match against a Writer's query.

Also reports whether each hit was EXACT (normalized text identical to the
stored query) or SEMANTIC-ONLY (similar text, not identical) -- the
notebook doesn't distinguish these since it never combined a semantic
matcher with an exact-hash matcher across two different runs. This split
is what turns "semantic matching found paraphrases" from a claim into a
measured number.

FALSE-POSITIVE GUARD (added after Experiment 17's pool_size=1000 audit) --
-----------------------------------------------------------------------
At pool_size=1000 scale, manually auditing every semantic-only hit found
that ALL of them were false positives: two genuinely different questions
(different logic-puzzle instances, different coding prompts, different
word problems) sharing a long boilerplate template, which inflates
TF-IDF/keyword overlap on the template text while the actual
distinguishing details (different clue sets, different function specs,
different numbers) get drowned out. A tuned threshold alone can't fix
this: false positives measured HIGHER combined scores (up to 0.591) than
some genuine paraphrases (as low as 0.364) -- the two populations overlap
on `combined`/`keyword_similarity` alone.

What DOES separate them empirically (see Experiment 17's audit, all 5
genuine paraphrases vs. all 17 unique false-positive pairs found):
  1. NUMBERS must match exactly. Every genuine paraphrase pair had
     identical number sets (same digits, just reworded around them);
     4/17 false positives differed only in their numbers (e.g. a
     3-house puzzle matching a 4-house puzzle, or two word problems with
     different quantities).
  2. Character-level similarity (not word-bag -- the ACTUAL text, via
     difflib.SequenceMatcher) must clear a floor. Genuine paraphrases
     scored 0.595-0.835 here; 13/17 false positives (every long,
     boilerplate-heavy one) scored 0.065-0.365 -- nearly all of a long
     puzzle's TEXT differs even though many of its WORDS repeat.

Together these two guards eliminate 14 of 17 false-positive patterns with
ZERO loss of genuine paraphrase recall (verified against every hit
Experiment 16/17 actually produced). The remaining 3 are SHORT sentences
that differ from a different question by only a few words (e.g. "Write a
python function to count the number of lists..." vs "...count the number
of unique lists...") -- structurally indistinguishable from how a genuine
short paraphrase differs from its original using surface text alone. This
is a known, documented residual risk, not something either guard (or any
cheap text-similarity heuristic) can close -- closing it would need either
semantic/entity-level understanding or an LLM verification step.
"""

import difflib
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Identical stopword set to Memory.ipynb's final TransactiveSharedMemory --
# kept as-is rather than swapping in sklearn's "english" list for THIS step
# (extract_keywords), because tfidf_similarity() below separately already
# uses sklearn's own stop_words="english", exactly mirroring how the
# notebook uses two different stopword treatments for its two signals.
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to",
    "and", "or", "why", "how", "can", "could", "would", "should", "do",
    "does", "did", "explain", "describe", "define", "tell", "me", "about",
    "please", "give", "details", "information", "meaning", "mean",
}


class SemanticSharedMemoryDirectory:
    """"Who already knows something semantically close to this?" -- one
    independent TF-IDF + keyword index per stage name, matching this
    project's SharedMemoryDirectory (shared_memory.py) public shape:
    lookup()/record()/stats(), so qa_pipeline.py's orchestration loop reads
    like pipeline.py's."""

    def __init__(self, similarity_threshold=0.35, keyword_threshold=0.40,
                 char_ratio_floor=0.5):
        self.similarity_threshold = similarity_threshold
        self.keyword_threshold = keyword_threshold
        # Minimum difflib.SequenceMatcher ratio on the ACTUAL normalized text
        # (not word-bag overlap) a candidate must clear to be accepted as a
        # semantic-only hit -- see the false-positive guard note in this
        # module's docstring. 0.5 is set with margin below the lowest
        # genuine-paraphrase ratio measured (0.595) and above the highest
        # long-boilerplate false-positive ratio measured (0.365).
        self.char_ratio_floor = char_ratio_floor
        # stage -> list of {"text":.., "item_id":.., "source_label":.., "timestamp":..}
        self._by_stage = {}
        self.lookups = 0
        self.hits = 0
        self.exact_hits = 0
        self.semantic_only_hits = 0
        self.guard_rejections = 0  # candidates that cleared the score
                                   # threshold but failed the divergence guard

    # -- text processing, identical logic to the notebook -------------------

    def _normalize(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_keywords(self, text):
        words = self._normalize(text).split()
        return {w for w in words if w not in _STOPWORDS}

    def _keyword_similarity(self, query, stored_query):
        q_words = self._extract_keywords(query)
        s_words = self._extract_keywords(stored_query)
        if not q_words or not s_words:
            return 0.0
        return len(q_words & s_words) / len(q_words | s_words)

    def _tfidf_similarities(self, query, stored_texts):
        documents = stored_texts + [query]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(documents)
        return cosine_similarity(matrix[-1], matrix[:-1])[0]

    def _numbers(self, text):
        return set(re.findall(r"\d+", text))

    def _char_ratio(self, query, stored_query):
        return difflib.SequenceMatcher(
            None, self._normalize(query), self._normalize(stored_query)
        ).ratio()

    def _passes_divergence_guard(self, query, stored_query):
        """A candidate that clears the score threshold is still rejected if
        it's a different-numbers or mostly-different-TEXT match -- see the
        false-positive guard note in this module's docstring. Both checks
        are necessary: numbers alone misses same-number-different-content
        collisions (e.g. two different 3-house logic puzzles), and
        char-ratio alone misses different-number collisions that happen to
        share most of their surrounding text."""
        if self._numbers(query) != self._numbers(stored_query):
            return False
        return self._char_ratio(query, stored_query) >= self.char_ratio_floor

    # -- public interface -----------------------------------------------------

    def lookup(self, stage, query):
        """Returns a dict with at least "hit" (bool). On a hit, also
        "item_id", "matched_query", "similarity" (combined score),
        "keyword_similarity", and "exact" (bool -- normalized text was
        byte-identical to the stored query, vs. merely similar)."""
        self.lookups += 1
        entries = self._by_stage.get(stage, [])

        if not entries:
            return dict(hit=False, item_id=None, matched_query=None,
                       similarity=0.0, keyword_similarity=0.0, exact=False)

        normalized_query = self._normalize(query)

        # -- exact match first (cheap, and avoids TF-IDF noise on an
        #    already-certain case) --------------------------------------
        for entry in entries:
            if self._normalize(entry["text"]) == normalized_query:
                self.hits += 1
                self.exact_hits += 1
                return dict(hit=True, item_id=entry["item_id"], matched_query=entry["text"],
                           similarity=1.0, keyword_similarity=1.0, exact=True)

        # -- semantic (TF-IDF + keyword) ----------------------------------
        stored_texts = [e["text"] for e in entries]
        tfidf_scores = self._tfidf_similarities(query, stored_texts)
        keyword_scores = [self._keyword_similarity(query, t) for t in stored_texts]
        combined = [0.40 * t + 0.60 * k for t, k in zip(tfidf_scores, keyword_scores)]

        best_idx = int(np.argmax(combined))
        best_combined = combined[best_idx]
        best_keyword = keyword_scores[best_idx]

        # Candidates in descending combined-score order -- the top-scoring
        # one is usually also the one that passes the divergence guard, but
        # checking every candidate that clears the score threshold (not
        # just the single best one) means a close runner-up doesn't get
        # missed just because an even-higher-scoring false-positive-shaped
        # candidate happened to rank first.
        order = sorted(range(len(entries)), key=lambda i: combined[i], reverse=True)
        for idx in order:
            if combined[idx] < self.similarity_threshold and keyword_scores[idx] < self.keyword_threshold:
                break  # remaining candidates score even lower; none can qualify
            entry = entries[idx]
            if not self._passes_divergence_guard(query, entry["text"]):
                self.guard_rejections += 1
                continue
            self.hits += 1
            self.semantic_only_hits += 1
            return dict(hit=True, item_id=entry["item_id"], matched_query=entry["text"],
                       similarity=float(combined[idx]), keyword_similarity=float(keyword_scores[idx]),
                       exact=False)

        return dict(hit=False, item_id=None, matched_query=None,
                   similarity=float(best_combined), keyword_similarity=float(best_keyword),
                   exact=False)

    def record(self, stage, query, item_id, source_label):
        self._by_stage.setdefault(stage, []).append(dict(
            text=query, item_id=item_id, source_label=source_label, timestamp=time.time(),
        ))

    def stats(self):
        hit_rate = self.hits / self.lookups if self.lookups else 0.0
        return dict(
            lookups=self.lookups, hits=self.hits, hit_rate=hit_rate,
            exact_hits=self.exact_hits, semantic_only_hits=self.semantic_only_hits,
            guard_rejections=self.guard_rejections,
        )
