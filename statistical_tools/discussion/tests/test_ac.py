"""Definition tests: wrong anchors, signs, correlation and graph weighting must fail."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
class CliTests(unittest.TestCase):
    def test_cli_exports_only_ac_without_mds_points(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data'/'gcc').mkdir(parents=True)
            (root/'cache'/'summary').mkdir(parents=True)
            pd.DataFrame({'radius': [.34]}).to_csv(root/'cache'/'summary'/'gcc_large_summary.csv', index=False)
            pd.DataFrame({'x': [0, 1, 2, 3], 'run_time': [0, 1, 2, 3]}).to_csv(root/'data'/'gcc'/'gcc_large.csv', index=False)
            script = Path(__file__).resolve().parents[1]/'scripts'/'discussion_3_analyze_ac.py'
            result = subprocess.run([sys.executable, str(script), '--systems', 'gcc', '--dataset-root', str(root/'data'),
                                     '--cache-dir', str(root/'cache'), '--output-dir', str(root/'out')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = pd.read_csv(root/'out'/'large'/'ac_summary.csv')
            self.assertTrue(np.isfinite(summary.AC.iloc[0]))
            self.assertEqual(summary.columns.tolist(), ['System', 'AC'])
            self.assertEqual(sorted(p.name for p in (root/'out'/'large').iterdir()),
                             ['ac_summary.csv'])

if __name__ == '__main__':
    unittest.main()
