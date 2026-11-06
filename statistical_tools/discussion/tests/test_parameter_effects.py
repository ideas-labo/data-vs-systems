"""Single-option eta-squared definitions and config-only grouping."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from parameter_effects import encode_parameters, effect_profile


class EffectTests(unittest.TestCase):
    def test_marginal_association(self):
        a = np.tile([0,0,1,1], 20)
        y = 2*a + np.tile([0,1,0,1], 20)
        self.assertAlmostEqual(effect_profile(a,y)[0], .8)

    def test_constant_and_affine_invariance(self):
        a = np.tile([0,1],20)
        y = a*2.
        self.assertTrue(np.isnan(effect_profile(a,np.ones(len(a)))[0]))
        self.assertAlmostEqual(effect_profile(a,y)[0], effect_profile(a,y*7+5)[0])

    def test_numeric_binning_and_categorical_levels(self):
        codes, metadata = encode_parameters(pd.DataFrame({
            'numeric':np.arange(120), 'flag':['On','Off']*60}),6)
        self.assertEqual(len(np.unique(codes['numeric'])),6)
        self.assertEqual(len(np.unique(codes['flag'])),2)
        self.assertEqual(metadata[0]['binning'],'quantile')

if __name__ == '__main__':
    unittest.main()
