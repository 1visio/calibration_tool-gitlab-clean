from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .exporters import write_csv, write_ply
from .interfaces import FrameSource, ProfileReconstructor, StripeExtractor
from .metrics import summarize_profile_quality


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    csv_path: Path
    ply_path: Path
    summary_path: Path


class StaticProfilePipeline:
    def __init__(
        self,
        source: FrameSource,
        extractor: StripeExtractor,
        reconstructor: ProfileReconstructor,
    ) -> None:
        self.source = source
        self.extractor = extractor
        self.reconstructor = reconstructor

    def run_once(self, output_dir: str | Path, context: dict[str, Any]) -> RunArtifacts:
        frame = self.source.capture()
        profile = self.extractor.extract(frame)
        cloud = self.reconstructor.reconstruct(profile)

        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = Path(output_dir) / run_id
        csv_path = write_csv(run_dir / "profile.csv", cloud)
        ply_path = write_ply(run_dir / "profile.ply", cloud)
        summary_path = run_dir / "run_summary.json"

        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": now.isoformat(),
            "context": context,
            "frame": asdict(frame.metadata),
            "profile": summarize_profile_quality(profile),
            "point_cloud": {
                "point_count": cloud.size,
                "valid_count": int(cloud.valid.sum()),
                "valid_ratio": float(cloud.valid.mean()) if cloud.valid.size else 0.0,
                "unit": "mm",
                "fields": [
                    "x_mm",
                    "y_mm",
                    "z_mm",
                    "intensity",
                    "confidence",
                    "valid",
                ],
            },
            "artifacts": {"csv": csv_path.name, "ply": ply_path.name},
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return RunArtifacts(run_dir, csv_path, ply_path, summary_path)
