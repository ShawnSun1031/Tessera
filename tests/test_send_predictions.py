import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from label_studio_sdk.converter.brush import decode_rle
from PIL import Image
from sqlmodel import Session

from app.db import get_engine
from app.models import Dataset, Job, TrainedModel
from app.services.ls_predictions import labeling_tools, prediction_results
from app.storage import job_dir
from conftest import FakeLabelStudio

W, H = 200, 100
POLY_CONFIG = (
    '<View><Image name="img" value="$image"/>'
    '<BrushLabels name="brush" toName="img"><Label value="Crack"/><Label value="Hole"/></BrushLabels>'
    '<PolygonLabels name="poly" toName="img"><Label value="Crack"/><Label value="Hole"/></PolygonLabels>'
    "</View>"
)
BRUSH_ONLY = (
    '<View><Image name="image" value="$image"/><BrushLabels name="tag" toName="image">'
    '<Label value="Crack"/><Label value="Hole"/></BrushLabels></View>'
)
RECT_CONFIG = (
    '<View><Image name="image" value="$image"/><RectangleLabels name="box" toName="image">'
    '<Label value="Hole"/></RectangleLabels></View>'
)

SEG_OBJECT = {
    "class": "Crack", "confidence": 0.9, "box_xyxy": [20, 10, 100, 50],
    "polygon_xyn": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
}  # fmt: skip
BOX_OBJECT = {"class": "Hole", "confidence": 0.6, "box_xyxy": [20, 10, 100, 50]}


def record(*objects: dict[str, Any]) -> dict[str, Any]:
    return {"image": "a.png", "width": W, "height": H, "objects": list(objects)}


def test_labeling_tools_parsing() -> None:
    tools = labeling_tools(POLY_CONFIG)
    assert tools.image_name == "img"
    assert set(tools.controls) == {"brushlabels", "polygonlabels"}
    with pytest.raises(ValueError, match="no <Image>"):
        labeling_tools("<View><Text name='t' value='$t'/></View>")
    with pytest.raises(ValueError, match="no rectangle, polygon or brush"):
        labeling_tools('<View><Image name="i" value="$i"/></View>')


def test_segmentation_prefers_polygons() -> None:
    [result], skipped = prediction_results(record(SEG_OBJECT), labeling_tools(POLY_CONFIG))
    assert skipped == 0
    assert (result["type"], result["from_name"], result["to_name"]) == ("polygonlabels", "poly", "img")
    assert result["value"]["points"] == [[10, 10], [50, 10], [50, 50], [10, 50]]
    assert result["value"]["polygonlabels"] == ["Crack"]
    assert result["score"] == 0.9


def test_brush_only_project_gets_masks() -> None:
    [result], _ = prediction_results(record(SEG_OBJECT), labeling_tools(BRUSH_ONLY))
    assert result["type"] == "brushlabels" and result["value"]["format"] == "rle"
    mask = decode_rle(result["value"]["rle"]).reshape(H, W, 4)[:, :, 3] > 127
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 100, 10, 50)


def test_boxes_and_unknown_labels() -> None:
    tools = labeling_tools(RECT_CONFIG)
    results, skipped = prediction_results(record(SEG_OBJECT, BOX_OBJECT), tools)
    assert skipped == 1  # "Crack" isn't a label in this Label Studio project
    value = results[0]["value"]
    assert results[0]["type"] == "rectanglelabels"
    assert (value["x"], value["y"], value["width"], value["height"]) == (10, 10, 40, 40)
    # a detection model's boxes still work in an outline-only project
    [poly], _ = prediction_results(record(BOX_OBJECT), labeling_tools(POLY_CONFIG))
    assert poly["type"] == "polygonlabels" and len(poly["value"]["points"]) == 4


def finished_predict_job(client: TestClient, admin: dict[str, str], ls_id: int) -> int:
    body = {"ls_project_id": ls_id}
    pid = client.post("/api/projects/from-labelstudio", json=body, headers=admin).json()["id"]
    with Session(get_engine()) as session:
        train = Job(project_id=pid, user_id=1, kind="train", status="succeeded")
        dataset = Dataset(project_id=pid, name="d", source="upload", classes=["Crack"])
        session.add_all([train, dataset])
        session.commit()
        assert train.id is not None and dataset.id is not None
        model = TrainedModel(
            project_id=pid, job_id=train.id, dataset_id=dataset.id, name="v1",
            task_type="segment", architecture="yolo26n-seg.pt", classes=["Crack", "Hole"],
        )  # fmt: skip
        session.add(model)
        session.commit()
        job = Job(project_id=pid, user_id=1, kind="predict", status="succeeded",
                  params={"model_id": model.id})  # fmt: skip
        session.add(job)
        session.commit()
        job_id = job.id
    assert job_id is not None
    out = job_dir(job_id)
    (out / "inputs").mkdir(parents=True)
    (out / "output").mkdir()
    records = []
    for name in ("a.png", "b.png"):
        Image.new("RGB", (W, H)).save(out / "inputs" / name)
        records.append({**record(SEG_OBJECT, BOX_OBJECT), "image": name})
    records[1]["objects"] = []  # nothing found: still worth labeling
    (out / "output" / "predictions.json").write_text(json.dumps(records))
    return job_id


def test_send_predictions_to_label_studio(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    ls_id = fake_ls.add_existing(BRUSH_ONLY, "Cracks")
    job_id = finished_predict_job(client, admin, ls_id)
    url = f"/api/labelstudio/send-predictions/{job_id}"

    resp = client.post(url, headers=admin)
    assert resp.status_code == 200, resp.text
    sent = resp.json()
    assert (sent["images"], sent["objects"], sent["skipped_objects"]) == (2, 2, 0)
    assert sent["ls_url"].endswith(f"/projects/{ls_id}")
    assert fake_ls.uploaded == ["a.png", "b.png"]
    [pred] = fake_ls.predictions[1001]  # a.png's task; b.png had nothing to pre-label
    assert pred["model_version"].startswith("v1 (model #")
    assert {r["type"] for r in pred["result"]} == {"brushlabels"}
    assert pred["score"] == pytest.approx(0.75)
    assert 1002 not in fake_ls.predictions
    # Label Studio only pre-fills with the project's selected model version
    assert fake_ls.projects[ls_id]["model_version"] == pred["model_version"]

    job = client.get(f"/api/jobs/{job_id}", headers=admin).json()
    assert job["result"]["label_studio"]["images"] == 2
    images = client.get(f"/api/projects/{job['project_id']}/images", headers=admin).json()
    assert [i["in_label_studio"] for i in images] == [True, True]

    assert client.post(url, headers=admin).status_code == 409  # no accidental duplicates
    assert client.post(f"{url}?force=true", headers=admin).status_code == 200
    assert len(fake_ls.uploaded) == 4


def test_only_finished_predictions_can_be_sent(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    job_id = finished_predict_job(client, admin, fake_ls.add_existing(BRUSH_ONLY))
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.status = "running"
        session.add(job)
        session.commit()
    resp = client.post(f"/api/labelstudio/send-predictions/{job_id}", headers=admin)
    assert resp.status_code == 409
