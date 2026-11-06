"""Percentile-scale maximum on a small dataset with tied performance."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from discussion_4_analyze_parameter_effects import analyze_frame


class MainEffectsWorkflowTests(unittest.TestCase):
    def test_only_maximum_percentile_effect(self):
        data = pd.DataFrame({'A':[0,0,1,1], 'B':[0,1,0,1], 'y':[10,10,20,30]})
        result = analyze_frame(data, 'example', True)
        self.assertEqual(set(result), {'system','strongest_parameter','max_percentile_eta_squared'})
        self.assertEqual(result['strongest_parameter'], 'A')
        self.assertAlmostEqual(result['max_percentile_eta_squared'], 8/9)
        self.assertEqual(result, analyze_frame(data, 'example', False))

if __name__ == '__main__':
    unittest.main()
