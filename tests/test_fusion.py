import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from configs.config_elephant import FUSION_WEIGHTS
from models.fusion import Recommendation


def test_fusion_weights_sum_to_one():
    total = sum(FUSION_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"FUSION_WEIGHTS sum {total} != 1.0"


def test_recommendation_dataclass():
    rec = Recommendation(
        individual_id="eleph_kunene",
        image_id="img-0001",
        crop_path="/data/crops/img-0001.jpg",
        viewpoint="left",
        fused_sim=0.87,
        global_sims={"megadescriptor": 0.82, "miewid": 0.79},
        local_inliers=42,
        local_sim=0.77,
        viz_payload={"matches": []},
    )

    assert rec.individual_id == "eleph_kunene"
    assert rec.image_id == "img-0001"
    assert rec.crop_path == "/data/crops/img-0001.jpg"
    assert rec.viewpoint == "left"
    assert rec.fused_sim == pytest.approx(0.87)
    assert rec.global_sims["megadescriptor"] == pytest.approx(0.82)
    assert rec.global_sims["miewid"] == pytest.approx(0.79)
    assert rec.local_inliers == 42
    assert rec.local_sim == pytest.approx(0.77)
    assert isinstance(rec.viz_payload, dict)
