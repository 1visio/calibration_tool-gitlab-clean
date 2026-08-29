import tempfile
import unittest
from pathlib import Path

from calibration_tool.camera import load_capture_plan
from calibration_tool.camera.models import CameraConfig
from calibration_tool.camera.plan_builder import (
    CaptureRecipe,
    CaptureRecipeItem,
    build_capture_plan_from_recipe,
    capture_plan_summary,
    save_generated_capture_plan,
)


class CapturePlanBuilderTests(unittest.TestCase):
    def _items(self):
        return (
            CaptureRecipeItem(
                role="chess",
                filename_prefix="chess",
                exposure_us=800,
                laser_state="off",
                quality_mode="chessboard",
                instruction_template="{split} pose {pose_id}: chess ({laser_state})",
            ),
            CaptureRecipeItem(
                role="laser",
                filename_prefix="laser",
                exposure_us=1600,
                laser_state="on",
                quality_mode="laser",
            ),
            CaptureRecipeItem(
                role="nolaser",
                filename_prefix="nolaser",
                exposure_us=1200,
                laser_state="off",
                quality_mode="generic",
            ),
        )

    def _recipe(self, root: Path, **overrides):
        values = {
            "dataset_id": "visual_batch",
            "output_dir": root / "dataset",
            "plan_output_path": root / "plans" / "visual_batch.yaml",
            "fit_group_count": 18,
            "include_validation": True,
            "validation_group_count": 6,
            "camera": CameraConfig(pixel_format="Mono8", width=64, height=48),
            "serial_number": "SIM-001",
            "backend": "synthetic",
            "backend_options": {"target_fps": 1000},
            "items": self._items(),
        }
        values.update(overrides)
        return CaptureRecipe(**values)

    def test_18_fit_6_validation_triple_generates_72_tasks_and_contiguous_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_capture_plan_from_recipe(self._recipe(Path(temporary)))
            self.assertEqual(len(plan.tasks), 72)
            self.assertEqual(sum(task.frames for task in plan.tasks), 72)
            self.assertEqual(plan.tasks[0].pose_id, "001")
            self.assertEqual(plan.tasks[53].pose_id, "018")
            self.assertEqual(plan.tasks[54].pose_id, "019")
            self.assertEqual(plan.tasks[-1].pose_id, "024")
            self.assertTrue(all(task.relative_path(1).parts[0] == "fit" for task in plan.tasks[:54]))
            self.assertTrue(all(task.relative_path(1).parts[0] == "validation" for task in plan.tasks[54:]))
            self.assertEqual(
                [task.relative_path(1).as_posix() for task in plan.tasks[:3]],
                ["fit/chess 001.tif", "fit/laser 001.tif", "fit/nolaser 001.tif"],
            )
            self.assertEqual(plan.tasks[54].relative_path(1).as_posix(), "validation/chess 019.tif")

    def test_exposures_and_laser_state_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_capture_plan_from_recipe(self._recipe(Path(temporary)))
            first_group = plan.tasks[:3]
            self.assertEqual([task.config.exposure_us for task in first_group], [800.0, 1600.0, 1200.0])
            self.assertEqual([task.tags["laser_state"] for task in first_group], ["off", "on", "off"])

    def test_disabled_item_reduces_task_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            items = (
                self._items()[0],
                self._items()[2],
                CaptureRecipeItem(
                    enabled=False,
                    role="laser",
                    filename_prefix="laser",
                    exposure_us=1600,
                    laser_state="on",
                    quality_mode="laser",
                ),
            )
            plan = build_capture_plan_from_recipe(self._recipe(Path(temporary), items=items))
            self.assertEqual(len(plan.tasks), 48)
            self.assertTrue(all(task.role != "laser" for task in plan.tasks))

    def test_task_ids_and_output_paths_are_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_capture_plan_from_recipe(self._recipe(Path(temporary)))
            task_ids = [task.task_id for task in plan.tasks]
            paths = [task.relative_path(index) for task in plan.tasks for index in range(1, task.frames + 1)]
            self.assertEqual(len(task_ids), len(set(task_ids)))
            self.assertEqual(len(paths), len(set(str(path).casefold() for path in paths)))

    def test_mono8_tif_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            recipe = self._recipe(
                Path(temporary),
                items=(
                    CaptureRecipeItem(
                        role="chess",
                        filename_prefix="chess",
                        exposure_us=1000,
                        quality_mode="chessboard",
                        image_format="tif",
                    ),
                ),
                include_validation=False,
            )
            plan = build_capture_plan_from_recipe(recipe)
            self.assertEqual(plan.base_config.pixel_format, "Mono8")
            self.assertEqual(plan.tasks[0].image_format, "tif")

    def test_summary_contains_gui_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = capture_plan_summary(self._recipe(Path(temporary)))
            self.assertEqual(summary["fit_group_count"], 18)
            self.assertEqual(summary["validation_group_count"], 6)
            self.assertEqual(summary["task_count"], 72)
            record = summary["tasks"][0]
            self.assertEqual(
                set(record),
                {
                    "split",
                    "pose_id",
                    "task_id",
                    "role",
                    "exposure_us",
                    "laser_state",
                    "quality_mode",
                    "relative_output_path",
                    "frames",
                },
            )
            self.assertTrue(record["relative_output_path"].startswith("fit/"))

    def test_save_round_trip_and_does_not_create_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self._recipe(root)
            saved = save_generated_capture_plan(recipe)
            self.assertEqual(saved, recipe.plan_output_path.resolve())
            self.assertTrue(saved.is_file())
            self.assertFalse(recipe.output_dir.exists())
            loaded = load_capture_plan(saved)
            self.assertEqual(len(loaded.tasks), 72)
            self.assertEqual(loaded.tasks[0].tags["laser_state"], "off")
            with self.assertRaises(FileExistsError):
                save_generated_capture_plan(recipe)

    def test_invalid_counts_exposure_and_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                CaptureRecipe(
                    dataset_id="bad",
                    output_dir=root / "dataset",
                    plan_output_path=root / "bad.yaml",
                    fit_group_count=0,
                    items=self._items(),
                )
            with self.assertRaises(ValueError):
                CaptureRecipeItem(exposure_us=0)
            duplicate_items = (
                CaptureRecipeItem(role="same", filename_prefix="same", exposure_us=1000),
                CaptureRecipeItem(role="same", filename_prefix="same", exposure_us=1000),
            )
            with self.assertRaises(ValueError):
                build_capture_plan_from_recipe(self._recipe(root, items=duplicate_items))

    def test_plan_path_must_be_outside_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                build_capture_plan_from_recipe(
                    self._recipe(root, plan_output_path=root / "dataset" / "plan.yaml")
                )


if __name__ == "__main__":
    unittest.main()
