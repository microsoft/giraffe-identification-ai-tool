import pandas as pd

from pipeline.elephant_splits import generate_splits, validate_splits


def test_elpephants_manifest_uses_bteh_compatible_split_contract():
    rows = []
    for identity_index in range(6):
        individual_id = f"elpephants_{identity_index}"
        for session_index in range(3):
            rows.append(
                {
                    "image_id": f"image_{identity_index}_{session_index}",
                    "individual_id": individual_id,
                    "session_id": (
                        f"{individual_id}_201{session_index}-01"
                    ),
                    "content_hash": (
                        f"hash_{identity_index}_{session_index}"
                    ),
                    "include_status": "included",
                    "source_split": (
                        "train" if session_index < 2 else "val"
                    ),
                }
            )
    manifest = pd.DataFrame(rows)

    splits = generate_splits(manifest, n_unseen_folds=3, seed=7)

    assert validate_splits(splits) == []
    assert set(splits["split_protocol"]) == {"temporal", "unseen_identity"}
    assert splits["split"].isin(
        {"gallery", "probe", "held_out_gallery", "held_out_probe"}
    ).all()
