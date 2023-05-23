import numpy as np
from analysis.analysis_utils import corr_frame_stim
from unittest import TestCase


class TestAnalysisUtils(TestCase):
    def test_corr_frame_stim(self):
        f = np.array(
            [[0.0, 1], [1.0, 2], [2.0, 3], [3.0, 5], [4.0, 6], [50.0, 7]]  # Time, frame
        )
        s = np.array([[0.5, 99], [1, 96], [3, 91], [7, 42]])  # Time, stim
        expected = np.array(
            [
                [1.0, np.nan],  # t = 0  No stim info.
                [2.0, 96.0],  # t = 1  Stim newly updated.
                [3.0, 96.0],  # t = 2  Use data at t = 1
                [5.0, 91.0],  # t = 3  Stim newly updated.
                [6.0, 91.0],  # t = 4  Use data at t = 6
                [7.0, 42.0],
            ]
        )  # t = 50 Use data at t = 7
        self.assert_(np.allclose(corr_frame_stim(f, s), expected, equal_nan=True))
