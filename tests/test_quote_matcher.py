"""Pure helper tests over authored strings, with no dataset or PyArrow imports.

AST extraction executes only the actual PrefixMatcher class and PREFIX_LENGTH
constant from the VM-oriented script. It isolates that pure algorithm so its
correctness can be tested locally without importing storage libraries or main().
The expected result is an independent naive full-substring oracle.
"""

import ast
from collections import deque
from pathlib import Path
import random
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/discover_cross_agent_references.py"
_tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
_selected = [node for node in _tree.body
             if (isinstance(node, ast.ClassDef) and node.name == "PrefixMatcher")
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "PREFIX_LENGTH"
                         for target in node.targets))]
if len(_selected) != 2:
    raise AssertionError("Expected the real PrefixMatcher class and its length constant")
_namespace = {"deque": deque}
exec(compile(ast.Module(body=_selected, type_ignores=[]), str(SCRIPT), "exec"), _namespace)
PrefixMatcher = _namespace["PrefixMatcher"]
PREFIX_LENGTH = _namespace["PREFIX_LENGTH"]


def matcher_for(quotes, guard=lambda: None):
    """Provide the documented fixed-length, unique-prefix input contract."""
    grouped = {}
    for index, quote in enumerate(quotes):
        if len(quote) < PREFIX_LENGTH:
            raise ValueError("Test fixture quote must be at least one full prefix")
        grouped.setdefault(quote[:PREFIX_LENGTH], []).append(index)
    return PrefixMatcher(list(grouped), guard), list(grouped.values())


def expected_matches(text, quotes):
    return {index for index, quote in enumerate(quotes) if quote in text}


class QuoteMatcherTests(unittest.TestCase):
    def assert_matches_oracle(self, text, quotes):
        matcher, groups = matcher_for(quotes)
        self.assertEqual(matcher.matches(text, groups, quotes), expected_matches(text, quotes))

    def test_fixed_length_contract_is_32_characters(self):
        self.assertEqual(PREFIX_LENGTH, 32)

    def test_overlapping_patterns_and_failure_links(self):
        quotes = ["a" * 32, "a" * 31 + "b", "a" * 30 + "ba",
                  "a" * 29 + "baa", "ba" * 16, "ab" * 16]
        self.assert_matches_oracle("a" * 70 + "b" + "a" * 40 + "ab" * 20, quotes)

    def test_shifted_distinct_32_character_patterns(self):
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        quotes = [text[start:start + 32] for start in range(20)]
        matcher, groups = matcher_for(quotes)
        self.assertEqual(matcher.matches(text, groups, quotes), set(range(20)))

    def test_shared_prefix_requires_full_quote_verification(self):
        prefix = "SYNTHETIC-REFERENCE-PREFIX".ljust(32, "-")
        self.assertEqual(len(prefix), 32)
        quotes = [prefix + "alpha", prefix + "alphabet", prefix + "alpine", prefix]
        matcher, groups = matcher_for(quotes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(matcher.matches("before " + prefix + "alphabet after", groups, quotes), {0, 1, 3})

    def test_repeated_and_overlapping_occurrences_count_once_per_quote(self):
        quotes = ["r" * 32, "r" * 33, "r" * 40]
        matcher, groups = matcher_for(quotes)
        self.assertEqual(matcher.matches("r" * 400, groups, quotes), {0, 1, 2})

    def test_all_quote_ids_sharing_an_identical_quote_are_returned(self):
        quote = "A completely synthetic repeated quote."
        quotes = [quote, quote]
        matcher, groups = matcher_for(quotes)
        self.assertEqual(matcher.matches(quote, groups, quotes), {0, 1})

    def test_unicode_uses_characters_not_utf8_bytes(self):
        prefix = "λ🙂é漢" * 8
        quotes = [prefix, prefix + " finished", "🙂" * 32 + " done"]
        self.assert_matches_oracle("preface " + prefix + " finished; " + "🙂" * 32 + " done", quotes)

    def test_unicode_normalization_and_case_are_not_silently_changed(self):
        quotes = ["é" * 32, "e\u0301" * 16, "A" * 32]
        matcher, groups = matcher_for(quotes)
        self.assertEqual(matcher.matches("é" * 32 + "a" * 32, groups, quotes), {0})

    def test_negatives_empty_text_short_prefix_and_wrong_suffix(self):
        quotes = ["z" * 32 + " approved"]
        for text in ("", "z" * 31, "z" * 32 + " rejected", "unrelated synthetic text"):
            with self.subTest(text=text):
                self.assert_matches_oracle(text, quotes)

    def test_empty_pattern_collection(self):
        self.assert_matches_oracle("some synthetic text", [])

    def test_match_at_start_end_and_across_punctuation(self):
        quotes = ["one synthetic quotation with punctuation: yes!",
                  "another synthetic statement with spaces."]
        for text in (quotes[0], "prefix " + quotes[1], quotes[0] + "\n" + quotes[1]):
            with self.subTest(text=text):
                self.assert_matches_oracle(text, quotes)

    def test_construction_calls_resource_guard(self):
        calls = []
        quotes = [format(index, "032b") for index in range(600)]
        matcher_for(quotes, lambda: calls.append(True))
        self.assertGreaterEqual(len(calls), 3)

    def test_deterministic_random_cases_match_naive_full_string_search(self):
        randomizer = random.Random(72418)
        alphabet = "abλ🙂"
        for case in range(100):
            quotes = ["".join(randomizer.choice(alphabet) for _ in range(randomizer.randint(32, 65)))
                      for _ in range(20)]
            # Deliberately add multiple full strings with an identical prefix.
            quotes += [quotes[0][:32] + "a", quotes[0][:32] + "b", quotes[0][:32]]
            pieces = ["".join(randomizer.choice(alphabet) for _ in range(150))]
            pieces += randomizer.sample(quotes, 5)
            text = "|".join(pieces)
            with self.subTest(case=case):
                self.assert_matches_oracle(text, quotes)


if __name__ == "__main__":
    unittest.main()
