import unittest

from evaluation.temporal_blend import select_candidate


class TemporalCandidateSelectionTests(unittest.TestCase):
    def test_false_darkening_limit_filters_temporally_better_unsafe_candidate(self):
        candidates = [("rgb", 0.0), ("residual", 0.5), ("residual", 0.75)]
        scores = {
            "rgb:0.0": {"mae": 0.01, "temporal_error": 0.006,
                         "unchanged_darken_rate_gt_5_255": 0.0002},
            "residual:0.5": {"mae": 0.0101, "temporal_error": 0.0058,
                             "unchanged_darken_rate_gt_5_255": 0.0006},
            "residual:0.75": {"mae": 0.0102, "temporal_error": 0.0055,
                              "unchanged_darken_rate_gt_5_255": 0.0012},
        }
        self.assertEqual(select_candidate(candidates, scores, 0.001), ("residual", 0.5))

    def test_unconstrained_selection_uses_validation_sum(self):
        candidates = [("rgb", 0.0), ("residual", 0.5)]
        scores = {
            "rgb:0.0": {"mae": 0.01, "temporal_error": 0.006,
                         "unchanged_darken_rate_gt_5_255": 0.0002},
            "residual:0.5": {"mae": 0.0101, "temporal_error": 0.0058,
                             "unchanged_darken_rate_gt_5_255": 0.0012},
        }
        self.assertEqual(select_candidate(candidates, scores), ("residual", 0.5))


if __name__ == "__main__":
    unittest.main()
