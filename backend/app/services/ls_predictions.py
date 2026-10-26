"""Turn platform predictions into Label Studio pre-annotations ("predictions").

Annotators then correct them in Label Studio instead of labeling from scratch.
Shapes are written with whichever tools the project's labeling config offers:
polygons when there is a PolygonLabels tool, brush masks for brush-only configs,
rectangles for RectangleLabels.
"""

import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from label_studio_sdk.converter.brush import mask2rle

CONTROL_TAGS = {
    "RectangleLabels": "rectanglelabels",
    "PolygonLabels": "polygonlabels",
    "BrushLabels": "brushlabels",
}


@dataclass
class LabelingTools:
    image_name: str  # <Image name=...> the shapes are drawn on
    # result type -> (control tag name, labels it offers)
    controls: dict[str, tuple[str, set[str]]] = field(default_factory=dict)


def labeling_tools(label_config: str) -> LabelingTools:
    root = ET.fromstring(label_config)
    image = next((el for el in root.iter() if el.tag == "Image"), None)
    if image is None or not image.get("name"):
        raise ValueError("The Label Studio project has no <Image> to label")
    tools = LabelingTools(image_name=image.get("name", ""))
    for el in root.iter():
        kind = CONTROL_TAGS.get(el.tag)
        if kind and el.get("name") and el.get("toName") == tools.image_name:
            labels = {lb.get("value", "") for lb in el.iter("Label")}
            tools.controls.setdefault(kind, (el.get("name", ""), labels))
    if not tools.controls:
        raise ValueError(
            "The Label Studio project has no rectangle, polygon or brush tool"
        )
    return tools


def box_polygon(box_xyxy: list[float], w: int, h: int) -> list[list[float]]:
    x0, y0, x1, y1 = box_xyxy
    return [[x0 / w, y0 / h], [x1 / w, y0 / h], [x1 / w, y1 / h], [x0 / w, y1 / h]]


def result_for(
    obj: dict[str, Any], tools: LabelingTools, w: int, h: int
) -> dict[str, Any] | None:
    """One LS result for a predicted object, or None if no tool has its label."""
    label = obj["class"]
    outline = obj.get("polygon_xyn") or box_polygon(obj["box_xyxy"], w, h)
    if len(outline) < 3:
        outline = box_polygon(obj["box_xyxy"], w, h)
    has_outline = "polygon_xyn" in obj

    # order of preference: keep the shape the model produced
    order = ["polygonlabels", "brushlabels", "rectanglelabels"]
    if not has_outline:
        order = ["rectanglelabels", "polygonlabels", "brushlabels"]
    for kind in order:
        if kind in tools.controls and label in tools.controls[kind][1]:
            break
    else:
        return None
    from_name = tools.controls[kind][0]

    if kind == "rectanglelabels":
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        value: dict[str, Any] = {
            "x": x0 * 100, "y": y0 * 100,
            "width": (min(1.0, max(xs)) - x0) * 100,
            "height": (min(1.0, max(ys)) - y0) * 100,
            "rotation": 0,
        }  # fmt: skip
    elif kind == "polygonlabels":
        points = [[min(100.0, max(0.0, x * 100)), min(100.0, max(0.0, y * 100))]
                  for x, y in outline]  # fmt: skip
        value = {"points": points, "closed": True}
    else:
        mask = np.zeros((h, w), np.uint8)
        pts = np.round(np.array(outline) * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        value = {"format": "rle", "rle": mask2rle(mask)}
    value[kind] = [label]
    return {
        "id": uuid.uuid4().hex[:10],
        "from_name": from_name,
        "to_name": tools.image_name,
        "type": kind,
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "value": value,
        "score": obj.get("confidence"),
    }


def prediction_results(
    record: dict[str, Any], tools: LabelingTools
) -> tuple[list[dict[str, Any]], int]:
    """LS results for one image's predictions, and how many objects were skipped."""
    w, h = record["width"], record["height"]
    results, skipped = [], 0
    for obj in record["objects"]:
        result = result_for(obj, tools, w, h)
        if result is None:
            skipped += 1
        else:
            results.append(result)
    return results, skipped
