
from __future__ import annotations

import math
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PseudoImageRecord:
    image_name: str
    label_name: str


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_labelled_images(images: Sequence[Path], val_ratio: float = 0.2) -> Tuple[List[Path], List[Path]]:
    items = list(images)
    if not items:
        raise ValueError("At least one human-labelled image is required.")
    if len(items) == 1:
        return items, items
    ratio = min(max(float(val_ratio), 0.0), 0.5)
    val_count = max(1, int(round(len(items) * ratio)))
    val = items[-val_count:]
    train = items[:-val_count]
    if not train:
        train = items[:1]
    return train, val


def read_yolo_lines(label_path: Path, require_boxes: bool = False) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        if require_boxes:
            raise FileNotFoundError(f"Label file does not exist: {label_path}")
        return boxes
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}: {line}")
        try:
            cls = int(float(parts[0]))
            xc, yc, width, height = (float(x) for x in parts[1:5])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric YOLO label at {label_path}:{line_no}: {line}") from exc
        values = (xc, yc, width, height)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite YOLO label at {label_path}:{line_no}: {line}")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"YOLO coordinates must be normalized to [0,1] at {label_path}:{line_no}: {line}")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"YOLO width/height must be positive at {label_path}:{line_no}: {line}")
        boxes.append((cls, xc, yc, width, height))
    if require_boxes and not boxes:
        raise ValueError(f"Human-labelled image has no boxes: {label_path}")
    return boxes


def write_yolo_boxes(
    label_path: Path,
    boxes: Iterable[Tuple[int, float, float, float, float]],
    confidences: Optional[Iterable[float]] = None,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    box_list = list(boxes)
    conf_list = list(confidences) if confidences is not None else None
    if conf_list is not None and len(conf_list) != len(box_list):
        raise ValueError("Number of confidences does not match number of boxes.")
    lines: List[str] = []
    for idx, (cls, xc, yc, width, height) in enumerate(box_list):
        base = f"{int(cls)} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"
        if conf_list is not None:
            base += f" {float(conf_list[idx]):.8f}"
        lines.append(base)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_pair(image: Path, label: Path, image_dst: Path, label_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(image), str(image_dst))
    shutil.copy2(str(label), str(label_dst))


def _write_dataset_yaml(dataset_dir: Path, class_names: Sequence[str]) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_warmup_dataset(
    labelled_images: Sequence[Path],
    human_label_dir: Path,
    dataset_dir: Path,
    class_names: Sequence[str],
    val_ratio: float,
) -> Tuple[Path, List[Path], List[Path]]:
    if dataset_dir.exists():
        shutil.rmtree(str(dataset_dir))
    train_images, val_images = split_labelled_images(labelled_images, val_ratio)
    total_human_boxes = sum(
        len(read_yolo_lines(human_label_dir / f"{image.stem}.txt")) for image in labelled_images
    )
    if total_human_boxes == 0:
        raise ValueError("The selected human-labelled set contains no target box.")
    for split, items in (("train", train_images), ("val", val_images)):
        for image in items:
            label = human_label_dir / f"{image.stem}.txt"
            read_yolo_lines(label)
            _copy_pair(
                image,
                label,
                dataset_dir / "images" / split / image.name,
                dataset_dir / "labels" / split / f"{image.stem}.txt",
            )
    return _write_dataset_yaml(dataset_dir, class_names), train_images, val_images


def parse_class_thresholds(spec: str, default_threshold: float, num_classes: int) -> Dict[int, float]:
    thresholds = {i: float(default_threshold) for i in range(num_classes)}
    text = (spec or "").strip()
    if not text:
        return thresholds
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid class threshold '{item}', expected class_id:threshold")
        cls_text, threshold_text = item.split(":", 1)
        cls = int(cls_text)
        threshold = float(threshold_text)
        if cls < 0 or cls >= num_classes:
            raise ValueError(f"Class threshold id {cls} is outside [0, {num_classes - 1}]")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Class threshold must be in [0,1], got {threshold}")
        thresholds[cls] = threshold
    return thresholds


def linear_threshold(start: float, end: float, round_index: int, total_rounds: int) -> float:
    if total_rounds <= 1:
        return float(end)
    alpha = (round_index - 1) / float(total_rounds - 1)
    return float(start + alpha * (end - start))


def generate_pseudo_labels(
    teacher_weight: Path,
    unlabeled_images: Sequence[Path],
    pseudo_dir: Path,
    imgsz: int,
    device: str,
    iou: float,
    default_threshold: float,
    class_thresholds: Mapping[int, float],
    min_area_ratio: float,
    max_area_ratio: float,
    min_boxes_per_image: int,
) -> List[PseudoImageRecord]:
    from ultralytics import YOLO

    if pseudo_dir.exists():
        shutil.rmtree(str(pseudo_dir))
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    records: List[PseudoImageRecord] = []
    if not unlabeled_images:
        return records

    min_predict_conf = min([float(default_threshold)] + [float(v) for v in class_thresholds.values()])
    model = YOLO(str(teacher_weight))
    results = model.predict(
        source=[str(p) for p in unlabeled_images],
        imgsz=imgsz,
        conf=max(0.001, min_predict_conf),
        iou=iou,
        device=device,
        stream=True,
        save=False,
        save_txt=False,
        augment=False,
        verbose=False,
    )

    seen = 0
    for result in results:
        seen += 1
        image_path = Path(result.path)
        kept_boxes: List[Tuple[int, float, float, float, float]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            coords_list = boxes.xywhn.detach().cpu().tolist()
            cls_list = boxes.cls.detach().cpu().tolist()
            conf_list = boxes.conf.detach().cpu().tolist()
            for cls_value, coords, conf_value in zip(cls_list, coords_list, conf_list):
                cls = int(cls_value)
                if cls not in class_thresholds:
                    continue
                conf = float(conf_value)
                threshold = float(class_thresholds.get(cls, default_threshold))
                xc, yc, width, height = (float(x) for x in coords)
                area = width * height
                if conf < threshold:
                    continue
                if area < min_area_ratio or area > max_area_ratio:
                    continue
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    continue
                kept_boxes.append((cls, xc, yc, width, height))

        if len(kept_boxes) >= min_boxes_per_image:
            label_path = pseudo_dir / f"{image_path.stem}.txt"

            write_yolo_boxes(label_path, kept_boxes)
            record = PseudoImageRecord(
                image_name=image_path.name,
                label_name=label_path.name,
            )
            records.append(record)

    if seen != len(unlabeled_images):
        raise RuntimeError(f"Teacher returned {seen} results for {len(unlabeled_images)} unlabelled images.")

    return records


def prepare_combined_dataset(
    labelled_train: Sequence[Path],
    labelled_val: Sequence[Path],
    human_label_dir: Path,
    unlabeled_by_stem: Mapping[str, Path],
    pseudo_label_dir: Path,
    pseudo_records: Sequence[PseudoImageRecord],
    dataset_dir: Path,
    class_names: Sequence[str],
    labelled_repeat: int,
    pseudo_repeat: int,
) -> Path:
    if dataset_dir.exists():
        shutil.rmtree(str(dataset_dir))
    labelled_repeat = max(1, int(labelled_repeat))
    pseudo_repeat = max(1, int(pseudo_repeat))

    labelled_copies = 0
    for image in labelled_train:
        source_label = human_label_dir / f"{image.stem}.txt"
        read_yolo_lines(source_label)
        for repeat_index in range(labelled_repeat):
            stem = f"human_r{repeat_index:02d}_{image.stem}"
            image_name = stem + image.suffix.lower()
            _copy_pair(
                image,
                source_label,
                dataset_dir / "images" / "train" / image_name,
                dataset_dir / "labels" / "train" / f"{stem}.txt",
            )
            labelled_copies += 1

    for record in pseudo_records:
        source_image = unlabeled_by_stem.get(Path(record.image_name).stem)
        source_label = pseudo_label_dir / record.label_name
        if source_image is None or not source_label.exists():
            continue
        boxes = read_yolo_lines(source_label)
        if not boxes:
            continue
        for repeat_index in range(pseudo_repeat):
            stem = f"pseudo_r{repeat_index:02d}_{source_image.stem}"
            image_name = stem + source_image.suffix.lower()
            _copy_pair(
                source_image,
                source_label,
                dataset_dir / "images" / "train" / image_name,
                dataset_dir / "labels" / "train" / f"{stem}.txt",
            )

    for image in labelled_val:
        source_label = human_label_dir / f"{image.stem}.txt"
        read_yolo_lines(source_label)
        _copy_pair(
            image,
            source_label,
            dataset_dir / "images" / "val" / image.name,
            dataset_dir / "labels" / "val" / f"{image.stem}.txt",
        )

    if labelled_copies == 0:
        raise RuntimeError("Combined semi-supervised dataset contains no human-labelled training samples.")

    return _write_dataset_yaml(dataset_dir, class_names)


def train_yolo_model(
    initial_weight: Path,
    dataset_yaml: Path,
    project: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    strong_augmentation: bool,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(initial_weight))
    train_args = dict(
        data=str(dataset_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        workers=int(workers),
        patience=int(patience),
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    if strong_augmentation:
        train_args.update(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0002,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.1,
        )
    model.train(**train_args)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"YOLO training finished but best.pt was not found: {best}")
    return best


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _checkpoint_module(checkpoint):
    import torch.nn as nn

    if isinstance(checkpoint, nn.Module):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("ema", "model"):
            value = checkpoint.get(key)
            if isinstance(value, nn.Module):
                return value
    raise TypeError("Checkpoint does not contain an Ultralytics model/ema module.")


def ema_merge_checkpoints(
    teacher_weight: Path,
    student_weight: Path,
    output_weight: Path,
    decay: float,
) -> Path:
    import torch

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0,1), got {decay}")
    teacher_ckpt = _torch_load(teacher_weight)
    student_ckpt = _torch_load(student_weight)
    teacher_module = _checkpoint_module(teacher_ckpt).float()
    student_module = _checkpoint_module(student_ckpt).float()

    teacher_state = teacher_module.state_dict()
    student_state = student_module.state_dict()
    if teacher_state.keys() != student_state.keys():
        missing = sorted(set(teacher_state) ^ set(student_state))[:20]
        raise RuntimeError(f"Teacher and Student architectures differ; mismatched keys: {missing}")

    averaged_module = deepcopy(student_module).float()
    averaged_state = {}
    with torch.no_grad():
        for key, student_tensor in student_state.items():
            teacher_tensor = teacher_state[key]
            if teacher_tensor.shape != student_tensor.shape:
                raise RuntimeError(
                    f"Teacher/Student tensor shape mismatch for {key}: "
                    f"{tuple(teacher_tensor.shape)} vs {tuple(student_tensor.shape)}"
                )
            if torch.is_floating_point(student_tensor):
                averaged_state[key] = teacher_tensor.to(torch.float32).mul(decay).add(
                    student_tensor.to(torch.float32), alpha=1.0 - decay
                ).to(student_tensor.dtype)
            else:
                averaged_state[key] = student_tensor
    averaged_module.load_state_dict(averaged_state, strict=True)

    if isinstance(student_ckpt, dict):
        output_ckpt = deepcopy(student_ckpt)
        output_ckpt["model"] = deepcopy(averaged_module).half()
        output_ckpt["ema"] = deepcopy(averaged_module).half()
        output_ckpt["optimizer"] = None
        output_ckpt["epoch"] = -1
        output_ckpt["best_fitness"] = None
    else:
        output_ckpt = averaged_module.half()

    output_weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, str(output_weight))
    return output_weight


def write_final_detection_labels(
    final_teacher: Path,
    all_images: Sequence[Path],
    labelled_stems: Sequence[str],
    human_label_dir: Path,
    output_dir: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> None:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.txt"):
        old.unlink()

    labelled = set(labelled_stems)
    unlabelled_images = [image for image in all_images if image.stem not in labelled]

    for image in all_images:
        if image.stem in labelled:
            boxes = read_yolo_lines(human_label_dir / f"{image.stem}.txt")
            write_yolo_boxes(output_dir / f"{image.stem}.txt", boxes, [1.0] * len(boxes))

    if unlabelled_images:
        model = YOLO(str(final_teacher))
        results = model.predict(
            source=[str(p) for p in unlabelled_images],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            stream=True,
            save=False,
            save_txt=False,
            augment=False,
            verbose=False,
        )
        seen = 0
        for result in results:
            seen += 1
            image_path = Path(result.path)
            predicted_boxes: List[Tuple[int, float, float, float, float]] = []
            confidences: List[float] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                coords_list = boxes.xywhn.detach().cpu().tolist()
                cls_list = boxes.cls.detach().cpu().tolist()
                conf_list = boxes.conf.detach().cpu().tolist()
                for cls, coords, confidence in zip(cls_list, coords_list, conf_list):
                    xc, yc, width, height = (float(x) for x in coords)
                    predicted_boxes.append((int(cls), xc, yc, width, height))
                    confidences.append(float(confidence))
            write_yolo_boxes(
                output_dir / f"{image_path.stem}.txt", predicted_boxes, confidences
            )
        if seen != len(unlabelled_images):
            raise RuntimeError(
                f"Final Teacher returned {seen} results for {len(unlabelled_images)} unlabelled images."
            )

    missing = [image.name for image in all_images if not (output_dir / f"{image.stem}.txt").exists()]
    if missing:
        raise RuntimeError(f"Final label generation missed {len(missing)} images: {missing[:10]}")
