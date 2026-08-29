import tempfile
import unittest
from pathlib import Path

from calibration_tool.gui.pages import discover_result_artifacts, load_residual_csv


class GuiResultTests(unittest.TestCase):
    def test_discovers_and_loads_residual_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            csv_path = output / "reprojection_residuals.csv"
            csv_path.write_text("image,residual_px\na,0.1\nb,-0.2\n", encoding="utf-8")
            unrelated = output / "frames.csv"; unrelated.write_text("x\n1\n", encoding="utf-8")
            result = {"stages": [{"output_dir": str(output)}]}
            self.assertEqual(discover_result_artifacts(result), [csv_path.resolve()])
            headers, rows, values = load_residual_csv(csv_path)
            self.assertEqual(headers, ["image", "residual_px"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(values, [0.1, -0.2])


if __name__ == "__main__":
    unittest.main()
