import io

import numpy as np
import pytest
from label_studio_sdk.converter.brush import mask2rle
from PIL import Image

from app.services.export_yolo import annotation_to_lines, class_names, write_dataset
from app.services.ingest import natural_key, sortable_name
from app.services.tif import convert_tif

W, H = 100, 50
CLASSES = {"Crack": 0, "Hole": 1}


def rect(label="Hole", x=10, y=20, w=30, h=40, rotation=0):
    v = {"x": x, "y": y, "width": w, "height": h, "rotation": rotation, "rectanglelabels": [label]}
    return {"type": "rectanglelabels", "value": v, "original_width": W, "original_height": H}


def polygon(label="Crack"):
    v = {"points": [[10, 10], [50, 10], [50, 60]], "polygonlabels": [label]}
    return {"type": "polygonlabels", "value": v, "original_width": W, "original_height": H}


def brush(label="Crack"):
    mask = np.zeros((H, W), np.uint8)
    mask[10:20, 20:40] = 255
    v = {"format": "rle", "rle": mask2rle(mask), "brushlabels": [label]}
    return {"type": "brushlabels", "value": v, "original_width": W, "original_height": H}


def parse(line):
    cls, *nums = line.split()
    return int(cls), [float(n) for n in nums]


def test_class_names():
    cfg = '<View><BrushLabels name="t" toName="image"><Label value="Crack"/><Label value="Hole" background="red"/></BrushLabels></View>'
    assert class_names(cfg) == ["Crack", "Hole"]


def test_rectangle_detect():
    [line] = annotation_to_lines([rect()], CLASSES, "detect")
    cls, xywh = parse(line)
    assert cls == 1
    assert xywh == pytest.approx([0.25, 0.4, 0.3, 0.4])


def test_rectangle_segment_is_four_corners():
    [line] = annotation_to_lines([rect()], CLASSES, "segment")
    _, pts = parse(line)
    assert pts == pytest.approx([0.1, 0.2, 0.4, 0.2, 0.4, 0.6, 0.1, 0.6])


def test_rotated_rectangle_detect_covers_rotation():
    [line] = annotation_to_lines([rect(rotation=90)], CLASSES, "detect")
    _, (cx, cy, w, h) = parse(line)
    # 30x20 px box rotated 90deg about (10,10) spans x in [-10,10] -> clipped to [0,10]
    assert cx - w / 2 == pytest.approx(0.0)
    assert cx + w / 2 == pytest.approx(0.1)


def test_polygon_segment_and_detect():
    [seg] = annotation_to_lines([polygon()], CLASSES, "segment")
    assert parse(seg) == (0, pytest.approx([0.1, 0.1, 0.5, 0.1, 0.5, 0.6]))
    [det] = annotation_to_lines([polygon()], CLASSES, "detect")
    assert parse(det)[1] == pytest.approx([0.3, 0.35, 0.4, 0.5])


def test_brush_segment_and_detect():
    [seg] = annotation_to_lines([brush()], CLASSES, "segment")
    cls, pts = parse(seg)
    xs, ys = pts[0::2], pts[1::2]
    assert cls == 0
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((0.20, 0.39, 0.20, 0.38))
    [det] = annotation_to_lines([brush()], CLASSES, "detect")
    assert parse(det)[1] == pytest.approx([0.295, 0.29, 0.19, 0.18])


def test_unknown_label_and_type_ignored():
    results = [rect(label="Other"), {"type": "choices", "value": {}}]
    assert annotation_to_lines(results, CLASSES, "detect") == []


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (W, H)).save(buf, "PNG")
    return buf.getvalue()


def test_write_dataset(tmp_path):
    def task(i, results, cancelled=False):
        return {"id": i, "data": {"image": f"/data/upload/4/abc-img{i}.png"},
                "annotations": [{"was_cancelled": cancelled, "result": results}]}

    tasks = [task(i, [rect()]) for i in range(5)] + [task(9, [rect()], cancelled=True), task(10, [])]
    data_yaml = write_dataset(tasks, ["Crack", "Hole"], tmp_path / "ds", lambda url: png_bytes(),
                              task_type="detect", log=lambda *_: None)
    assert "  1: Hole" in data_yaml.read_text()
    train = list((tmp_path / "ds/images/train").glob("*.png"))
    val = list((tmp_path / "ds/images/val").glob("*.png"))
    assert (len(train), len(val)) == (4, 1)
    label = (tmp_path / "ds/labels/val" / f"{val[0].stem}.txt").read_text()
    assert label.startswith("1 ")


def test_convert_16bit_tif(tmp_path):
    arr = np.linspace(0, 4000, W * H, dtype=np.uint16).reshape(H, W)
    src = tmp_path / "a.tif"
    Image.fromarray(arr).save(src)
    [out] = convert_tif(src, tmp_path / "png")
    im = Image.open(out)
    assert (im.mode, im.size) == ("L", (W, H))
    px = np.array(im)
    assert px.min() == 0 and px.max() == 255


def test_convert_multipage_tif(tmp_path):
    src = tmp_path / "stack.tif"
    frames = [Image.new("RGB", (W, H), c) for c in ("red", "green", "blue")]
    frames[0].save(src, save_all=True, append_images=frames[1:])
    outs = convert_tif(src, tmp_path / "png")
    assert [o.name for o in outs] == ["stack_p000.png", "stack_p001.png", "stack_p002.png"]


def test_natural_and_sortable_names():
    names = ["10.tif", "2.tif", "1_p010.png", "1_p002.png", "1.tif"]
    assert sorted(names, key=natural_key) == ["1.tif", "1_p002.png", "1_p010.png", "2.tif", "10.tif"]
    padded = [sortable_name(n) for n in names]
    assert sorted(padded) == [sortable_name(n) for n in sorted(names, key=natural_key)]
    assert sortable_name("scan_7.png") == "scan_000007.png"
