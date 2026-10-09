from __future__ import annotations

import unittest

from real_time_translation.translation.opus_mt import OpusTranslator


class TranslationRoutingTests(unittest.TestCase):
    def test_english_source_uses_multilingual_output_model(self) -> None:
        self.assertEqual(
            OpusTranslator._pairs_for_languages("en", "fr"),
            (("en", "mul"),),
        )
        self.assertEqual(
            OpusTranslator._pairs_for_languages("en", "de"),
            (("en", "mul"),),
        )

    def test_english_target_uses_multilingual_input_model(self) -> None:
        self.assertEqual(
            OpusTranslator._pairs_for_languages("fr", "en"),
            (("mul", "en"),),
        )
        self.assertEqual(
            OpusTranslator._pairs_for_languages("tr", "en"),
            (("mul", "en"),),
        )

    def test_non_english_pairs_share_the_same_english_pivot(self) -> None:
        expected = (("mul", "en"), ("en", "mul"))
        for source, target in (("fr", "de"), ("tr", "zh"), ("it", "es")):
            with self.subTest(source=source, target=target):
                self.assertEqual(OpusTranslator._pairs_for_languages(source, target), expected)

    def test_identical_languages_do_not_load_translation_models(self) -> None:
        self.assertEqual(OpusTranslator._pairs_for_languages("en", "en"), ())


if __name__ == "__main__":
    unittest.main()
