from __future__ import annotations

import json
import unittest

import joblib
from PIL import Image

from backend.model_service import get_style, get_top_styles, refresh_cache
from data_science.config import CONFIG


class ArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        refresh_cache()
        cls.top_payload = json.loads(CONFIG.top_styles_path.read_text(encoding="utf-8"))
        cls.metrics = json.loads(CONFIG.metrics_path.read_text(encoding="utf-8"))
        cls.concepts = json.loads(CONFIG.concept_brief_path.read_text(encoding="utf-8"))

    def test_top_three_are_ranked_and_have_images(self) -> None:
        styles = self.top_payload["styles"]
        self.assertEqual([style["rank"] for style in styles], [1, 2, 3])
        self.assertEqual(len({style["style_id"] for style in styles}), 3)
        self.assertTrue(all(style["image_url"] for style in styles))
        self.assertTrue(all(style["predicted_units"] >= 0 for style in styles))

    def test_referenced_images_are_packaged_for_deployment(self) -> None:
        payloads = [self.top_payload, json.loads(CONFIG.seasonal_styles_path.read_text(encoding="utf-8"))]
        image_urls = {
            value
            for payload in payloads
            for value in self._find_values(payload, "image_url")
            if value
        }
        self.assertGreater(len(image_urls), 3)
        for image_url in image_urls:
            relative_path = image_url.removeprefix("/images/")
            self.assertTrue(
                (CONFIG.deployment_images_dir / relative_path).is_file(),
                f"Missing deployment image: {relative_path}",
            )

    def test_model_selection_uses_multiple_folds_and_reports_holdout(self) -> None:
        rolling = self.metrics["rolling_backtest"]
        self.assertGreaterEqual(rolling["fold_count"], 3)
        self.assertEqual(len(rolling["folds"]), rolling["fold_count"])
        self.assertIn(self.metrics["selected_model_key"], rolling["candidate_summary"])
        self.assertIn("precision_at_3", self.metrics["final_holdout"]["candidate_model"])
        self.assertIn("precision_at_10", self.metrics["final_holdout"]["candidate_model"])

    def test_saved_model_is_the_selected_deployable_pipeline(self) -> None:
        artifact = joblib.load(CONFIG.model_path)
        self.assertEqual(artifact["selected_model"], self.metrics["selected_model_key"])
        self.assertIn("model", artifact)
        self.assertGreater(len(artifact["features"]), 0)
        self.assertTrue(hasattr(artifact["model"], "predict"))

    def test_concepts_match_locked_rank_order(self) -> None:
        style_ids = [style["style_id"] for style in self.top_payload["styles"]]
        concept_ids = [concept["style_id"] for concept in self.concepts["concepts"]]
        concept_ranks = [concept["rank"] for concept in self.concepts["concepts"]]
        self.assertEqual(concept_ids, style_ids)
        self.assertEqual(concept_ranks, [1, 2, 3])

    def test_final_concept_board_is_wide_and_nonblank(self) -> None:
        with Image.open(CONFIG.concept_image_path) as image:
            self.assertGreater(image.width, image.height)
            extrema = image.convert("RGB").getextrema()
            self.assertTrue(any(low != high for low, high in extrema))

    def test_model_service_contract(self) -> None:
        payload = get_top_styles()
        self.assertEqual(len(payload["styles"]), 3)
        first = payload["styles"][0]
        self.assertEqual(get_style(first["style_id"]), first)
        self.assertIsNone(get_style("not-a-real-style"))

    @staticmethod
    def _find_values(value: object, key: str) -> list[str]:
        if isinstance(value, dict):
            matches = [value[key]] if key in value else []
            for nested in value.values():
                matches.extend(ArtifactContractTests._find_values(nested, key))
            return matches
        if isinstance(value, list):
            matches = []
            for nested in value:
                matches.extend(ArtifactContractTests._find_values(nested, key))
            return matches
        return []


if __name__ == "__main__":
    unittest.main()
