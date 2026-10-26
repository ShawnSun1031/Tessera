import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from label_studio_sdk.converter.brush import mask2rle

from app import worker
from app.db import get_engine
from app.services.export_yolo import (
    annotation_to_lines,
    image_key,
    image_name,
    infer_task_type,
    write_dataset,
)
from conftest import FakeLabelStudio, png_bytes

BRUSH_CONFIG = (
    '<View><Image name="img" value="$photo"/><BrushLabels name="t" toName="img">'
    '<Label value="Crack"/><Label value="Hole"/></BrushLabels></View>'
)
RECT_CONFIG = (
    '<View><Image name="image" value="$image"/><RectangleLabels name="r" '
    'toName="image"><Label value="Car"/></RectangleLabels></View>'
)


def brush_result(mask: np.ndarray, label: str = "Crack") -> dict[str, Any]:
    h, w = mask.shape
    value = {"format": "rle", "rle": mask2rle(mask), "brushlabels": [label]}
    return {"type": "brushlabels", "value": value,
            "original_width": w, "original_height": h}  # fmt: skip


def two_blob_mask() -> np.ndarray:
    mask = np.zeros((64, 64), np.uint8)
    mask[5:15, 5:20] = 255
    mask[40:60, 30:50] = 255
    return mask


def test_brush_with_separate_regions_gives_one_polygon_each() -> None:
    lines = annotation_to_lines([brush_result(two_blob_mask())], {"Crack": 0}, "segment")
    assert len(lines) == 2
    for line in lines:
        cls, *coords = line.split()
        assert cls == "0" and len(coords) >= 6 and len(coords) % 2 == 0
        assert all(0 <= float(c) <= 1 for c in coords)


def test_brush_specks_below_min_area_are_dropped() -> None:
    mask = two_blob_mask()
    mask[30, 2] = 255  # one stray pixel from a sloppy brush stroke
    lines = annotation_to_lines([brush_result(mask)], {"Crack": 0}, "segment")
    assert len(lines) == 2


def test_label_config_inference() -> None:
    assert image_key(BRUSH_CONFIG) == "photo"
    assert image_key("<View/>") == "image"
    assert infer_task_type(BRUSH_CONFIG) == "segment"
    assert infer_task_type(RECT_CONFIG) == "detect"
    assert infer_task_type("<View><Choices/></View>") is None


def test_image_name_handles_storage_urls() -> None:
    png = png_bytes()
    assert image_name("/data/upload/3/ab12-x.png", png) == "ab12-x.png"
    assert image_name("/data/local-files/?d=scans/batch1/a.tif", png) == "a.tif"
    assert image_name("https://bucket.s3.amazonaws.com/img?X-Amz=1", png) == "img.png"


def test_write_dataset_with_custom_image_field(tmp_path: Path) -> None:
    task = {"id": 1, "data": {"photo": "/data/local-files/?d=a/scan.png"},
            "annotations": [{"result": [brush_result(two_blob_mask())]}]}  # fmt: skip
    write_dataset([task], ["Crack"], tmp_path / "ds", lambda u: png_bytes(),
                  log=lambda *_: None, data_key="photo")  # fmt: skip
    label = (tmp_path / "ds/labels/train/scan.txt").read_text().splitlines()
    assert len(label) == 2


def test_list_and_create_from_label_studio(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    brush = fake_ls.add_existing(BRUSH_CONFIG, "Weld cracks")
    fake_ls.add_existing("<View><Text name='t' value='$t'/></View>", "Text project")
    listed = client.get("/api/labelstudio/projects", headers=admin).json()
    assert listed[0]["task_type"] == "segment"
    assert listed[0]["classes"] == ["Crack", "Hole"]
    assert listed[0]["problem"] is None
    assert listed[1]["problem"]

    bad = {"ls_project_id": listed[1]["id"]}
    resp = client.post("/api/projects/from-labelstudio", json=bad, headers=admin)
    assert resp.status_code == 422
    missing = {"ls_project_id": 99}
    resp = client.post("/api/projects/from-labelstudio", json=missing, headers=admin)
    assert resp.status_code == 404

    body = {"ls_project_id": brush}
    project = client.post("/api/projects/from-labelstudio", json=body, headers=admin)
    assert project.status_code == 201
    p = project.json()
    assert (p["name"], p["task_type"], p["classes"]) == ("Weld cracks", "segment", ["Crack", "Hole"])
    assert p["ls_project_id"] == brush


def test_link_validation(
    client: TestClient, admin: dict[str, str], fake_ls: FakeLabelStudio
) -> None:
    rect = fake_ls.add_existing(RECT_CONFIG)
    brush = fake_ls.add_existing(BRUSH_CONFIG)

    def make(task_type: str, classes: list[str]) -> int:
        body = {"name": "p", "task_type": task_type, "classes": classes}
        return client.post("/api/projects", json=body, headers=admin).json()["id"]

    def link(pid: int, ls_id: int) -> Any:  # noqa: ANN401
        url = f"/api/projects/{pid}/labelstudio/link"
        return client.post(url, json={"ls_project_id": ls_id}, headers=admin)

    seg = make("segment", ["Crack", "Hole", "Dent"])
    assert link(seg, rect).status_code == 422  # rectangles can't train masks
    small = make("segment", ["Crack"])
    resp = link(small, brush)
    assert resp.status_code == 422 and "Hole" in resp.json()["detail"]
    assert link(seg, brush).status_code == 200
    assert link(seg, brush).status_code == 409  # already linked
    det = make("detect", ["Crack", "Hole"])
    assert link(det, brush).status_code == 200  # brush masks become boxes


def test_export_linked_brush_project_and_download(
    client: TestClient,
    admin: dict[str, str],
    fake_ls: FakeLabelStudio,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ls_id = fake_ls.add_existing(BRUSH_CONFIG)
    fake_ls.tasks = [
        {"id": i, "data": {"photo": f"/data/upload/{ls_id}/abc-scan{i}.png"},
         "annotations": [{"result": [brush_result(two_blob_mask())]}]}
        for i in range(5)
    ]  # fmt: skip
    body = {"ls_project_id": ls_id}
    pid = client.post("/api/projects/from-labelstudio", json=body, headers=admin).json()["id"]
    monkeypatch.setattr(worker.LabelStudio, "from_key", lambda *a: fake_ls)
    job = client.post(f"/api/projects/{pid}/datasets/export", json={}, headers=admin).json()
    engine = get_engine()
    assert worker.claim_next(engine, "test") == job["id"]
    worker.run_job(engine, job["id"])
    done = client.get(f"/api/jobs/{job['id']}", headers=admin).json()
    assert done["status"] == "succeeded", done["error"]
    assert done["result"]["stats"]["instances"] == {"Crack": 10, "Hole": 0}

    ds = done["result"]["dataset_id"]
    resp = client.get(f"/api/datasets/{ds}/download", headers=admin)
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    yaml = zf.read("data.yaml").decode()
    assert "path:" not in yaml and "  0: Crack" in yaml
    labels = [n for n in names if n.startswith("labels/") and n.endswith(".txt")]
    assert len(labels) == 5
    assert all(len(zf.read(n).decode().splitlines()) == 2 for n in labels)
