import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.services.datasets import label_objects, list_items, write_data_yaml
from app.storage import dataset_dir


def make_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    wide = np.linspace(0, 5000, 300 * 1200, dtype=np.uint16).reshape(300, 1200)
    Image.fromarray(wide).save(root / "images/train/scan16.tif")
    for i in range(3):
        Image.new("RGB", (64, 64)).save(root / f"images/train/img{i}.png")
    Image.new("RGB", (64, 64)).save(root / "images/val/v0.png")
    (root / "labels/train/scan16.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
    (root / "labels/train/img0.txt").write_text("1 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    (root / "labels/train/img1.txt").write_text("")
    write_data_yaml(root, ["Crack", "Hole"])


def test_label_objects_boxes_become_corners(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    box, poly = label_objects(f)
    assert box["kind"] == "box" and box["class_id"] == 1
    assert box["points"] == [[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]]
    assert poly["kind"] == "polygon" and len(poly["points"]) == 3
    assert label_objects(tmp_path / "missing.txt") == []


def test_list_items_filters(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    names = [i["name"] for i in list_items(tmp_path, "train")]
    assert names == ["img0.png", "img1.png", "img2.png", "scan16.tif"]
    assert [i["name"] for i in list_items(tmp_path, "train", class_id=1)] == ["img0.png"]
    assert [i["name"] for i in list_items(tmp_path, "train", class_id=0)] == ["img0.png", "scan16.tif"]
    # img1 has an empty label file, img2 has none at all
    assert [i["name"] for i in list_items(tmp_path, "train", unlabeled=True)] == ["img1.png", "img2.png"]
    assert len(list_items(tmp_path, "val")) == 1


def test_preview_endpoints(client: TestClient, admin: dict[str, str]) -> None:
    body = {"name": "p", "task_type": "segment", "classes": ["Crack", "Hole"]}
    pid = client.post("/api/projects", json=body, headers=admin).json()["id"]
    # register a dataset row, then lay its files out on disk
    from sqlmodel import Session

    from app.db import get_engine
    from app.models import Dataset

    with Session(get_engine()) as session:
        ds = Dataset(project_id=pid, name="d", source="upload", classes=["Crack", "Hole"],
                     stats={"splits": {}, "instances": {}})  # fmt: skip
        session.add(ds)
        session.commit()
        ds_id = ds.id
    assert ds_id is not None
    make_dataset(dataset_dir(pid, ds_id))
    url = f"/api/datasets/{ds_id}"

    page = client.get(f"{url}/items?split=train&limit=2&offset=2", headers=admin).json()
    assert page["total"] == 4
    assert [i["name"] for i in page["items"]] == ["img2.png", "scan16.tif"]
    filtered = client.get(f"{url}/items?class_id=1", headers=admin).json()
    assert filtered["total"] == 1 and filtered["items"][0]["objects"][0]["kind"] == "box"

    resp = client.get(f"{url}/preview/train/scan16.tif?size=500", headers=admin)
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (512, 128)  # 16-bit TIF -> 8-bit PNG, longest side 512
    assert im.mode == "L" and np.array(im).max() == 255

    assert client.get(f"{url}/preview/train/..%2F..%2Fdata.yaml", headers=admin).status_code == 404
    assert client.get(f"{url}/preview/test/img0.png", headers=admin).status_code == 422
    assert client.get(f"{url}/items", headers={}).status_code == 401
