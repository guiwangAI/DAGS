
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from tools.interactive_annotator import BoxAnnotator
from tools.semi_supervised_detector import (
    ema_merge_checkpoints,
    generate_pseudo_labels,
    linear_threshold,
    list_images,
    parse_class_thresholds,
    prepare_combined_dataset,
    prepare_warmup_dataset,
    train_yolo_model,
    write_final_detection_labels,
)


def resolve_selected_images(image_dir: Path, requested: Sequence[str], available: Sequence[Path]) -> List[Path]:
    by_name = {p.name: p for p in available}
    by_stem = {p.stem: p for p in available}
    selected: List[Path] = []
    for item in requested:
        candidate = Path(item)
        if candidate.is_file():
            selected.append(candidate.resolve())
        elif item in by_name:
            selected.append(by_name[item])
        elif candidate.stem in by_stem:
            selected.append(by_stem[candidate.stem])
        else:
            raise FileNotFoundError(f"Selected annotation image was not found: {item}")
    unique: List[Path] = []
    seen = set()
    for path in selected:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def prompt_for_images(image_dir: Path, available: Sequence[Path]) -> List[Path]:
    print(f"Found {len(available)} original images in: {image_dir}")
    print("Enter one or more image names separated by spaces or commas.")
    raw = input("Images to human-label: ").strip().replace(",", " ")
    if not raw:
        raise ValueError("No annotation images were specified.")
    return resolve_selected_images(image_dir, raw.split(), available)


def copy_weight(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return destination


def save_state(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_reconstruction(args, det_label_dir: Path, extra_args: Sequence[str]) -> None:
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(root / "train.py"),
        "-s", str(args.scene.resolve()),
        "-m", str(args.output.resolve()),
        "--dags_enable",
        "--det_label_dirs", str(det_label_dir.resolve()),
        "--dags_num_classes", str(len(args.class_names)),
        "--dags_class_densify_weights", str(args.class_densify_weights),
        "--dags_evidence_views_per_update", str(args.evidence_views_per_update),
        "--dags_evidence_update_interval", str(args.evidence_update_interval),
        "--dags_densify_max_soft_boost", str(args.densify_max_soft_boost),
        "--dags_densify_maturity_observations", str(args.densify_maturity_observations),
        "--dags_densify_ramp_iters", str(args.densify_ramp_iters),
        "--dags_densify_hold_end_iter", str(args.densify_hold_end_iter),
        "--dags_densify_anneal_end_iter", str(args.densify_anneal_end_iter),
        "--dags_gate_sharpen", str(args.gate_sharpen),
        "--dags_roi_loss_boost", str(args.roi_loss_boost),
        "--dags_mv_start_iter", str(args.mv_start_iter),
        "--dags_mv_interval", str(args.mv_interval),
        "--dags_mv_global_weight", str(args.mv_global_weight),
    ]
    if args.eval:
        command.append("--eval")
    command.extend(extra_args)
    print("[DAGS] reconstruction command:")
    print(" ".join(f'"{item}"' if " " in item else item for item in command))
    subprocess.run(command, check=True, cwd=str(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DAGS: Detection-Guided Adaptive Gaussian Splatting semi-supervised pipeline"
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-dir-name", default="images")
    parser.add_argument("--annotate-images", nargs="*", default=None)
    parser.add_argument("--class-names", nargs="+", default=["target"])
    parser.add_argument("--class-densify-weights", default="1.5",
                        help="Comma-separated per-class maximum densification multipliers")
    parser.add_argument("--evidence-views-per-update", type=int, default=48,
                        help="Rotating labelled views used per 3D evidence update; 0 uses all views")
    parser.add_argument("--evidence-update-interval", type=int, default=2000)
    parser.add_argument("--densify-max-soft-boost", type=float, default=0.45,
                        help="Positive detection bonus; the standard densification gradient stays unchanged")
    parser.add_argument("--densify-maturity-observations", type=float, default=6.0)
    parser.add_argument("--densify-ramp-iters", type=int, default=1000)
    parser.add_argument("--densify-hold-end-iter", type=int, default=17000)
    parser.add_argument("--densify-anneal-end-iter", type=int, default=19500)
    parser.add_argument("--gate-sharpen", type=float, default=1.5)
    parser.add_argument("--roi-loss-boost", type=float, default=0.10)
    parser.add_argument("--mv-start-iter", type=int, default=6000)
    parser.add_argument("--mv-interval", type=int, default=100)
    parser.add_argument("--mv-global-weight", type=float, default=0.002)
    parser.add_argument("--yolo-pretrained", default="yolo26s.pt",
                        help="Pretrained detector weights (default: YOLO26s)")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-batch", type=int, default=2)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-workers", type=int, default=0)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-patience", type=int, default=30)

    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--semi-rounds", type=int, default=3)
    parser.add_argument("--semi-epochs", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.8,
                        help="Round-wise EMA weight assigned to the previous Teacher")
    parser.add_argument("--labelled-val-ratio", type=float, default=0.2)
    parser.add_argument("--labelled-repeat", type=int, default=4,
                        help="Repeat human-labelled train images in each semi-supervised dataset")
    parser.add_argument("--pseudo-repeat", type=int, default=1)

    parser.add_argument("--pseudo-conf-start", type=float, default=0.80)
    parser.add_argument("--pseudo-conf-end", type=float, default=0.60)
    parser.add_argument("--pseudo-class-thresholds", default="",
                        help="Optional fixed thresholds, e.g. 0:0.75,1:0.80")
    parser.add_argument("--pseudo-min-area", type=float, default=0.0001)
    parser.add_argument("--pseudo-max-area", type=float, default=0.95)
    parser.add_argument("--pseudo-min-boxes", type=int, default=1)
    parser.add_argument("--final-conf", type=float, default=0.25)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip-annotation", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-semi", action="store_true")
    parser.add_argument("--skip-final-predict", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args, extra_train_args = parser.parse_known_args()
    args.scene = args.scene.resolve()
    args.output = args.output.resolve()

    image_dir = args.scene / args.images_dir_name
    all_images = list_images(image_dir)
    if not all_images:
        raise FileNotFoundError(f"No original images were found in: {image_dir}")

    work_dir = args.scene / "detector_workspace"
    human_label_dir = work_dir / "human_labels"

    if args.annotate_images:
        selected = resolve_selected_images(image_dir, args.annotate_images, all_images)
    elif not args.skip_annotation:
        selected = prompt_for_images(image_dir, all_images)
    else:
        labelled_stems = {path.stem for path in human_label_dir.glob("*.txt")}
        selected = [path for path in all_images if path.stem in labelled_stems]
        if not selected and not (args.skip_warmup and args.skip_semi and args.skip_final_predict):
            raise FileNotFoundError(
                "--skip-annotation requires existing human labels in "
                f"{human_label_dir}, or skip the detector stages and use existing det_labels."
            )

    selected_stems = {path.stem for path in selected}
    unlabeled_images = [path for path in all_images if path.stem not in selected_stems]
    unlabeled_by_stem = {path.stem: path for path in unlabeled_images}
    warmup_dataset_dir = work_dir / "warmup_dataset"
    weights_dir = work_dir / "weights"
    rounds_dir = work_dir / "rounds"
    state_path = work_dir / "state.json"
    det_label_dir = args.scene / "det_labels"
    warmup_teacher = weights_dir / "teacher_warmup.pt"
    final_teacher = weights_dir / "best_teacher.pt"

    detector_bypassed = args.skip_warmup and args.skip_semi and args.skip_final_predict
    if detector_bypassed:
        if not det_label_dir.is_dir():
            raise FileNotFoundError(
                "Detector stages were skipped, but no final label directory exists: "
                f"{det_label_dir}"
            )
        if not args.skip_reconstruction:
            print("[DAGS] Using existing detection labels and starting reconstruction")
            run_reconstruction(args, det_label_dir, extra_train_args)
        return

    if not args.skip_annotation:
        BoxAnnotator(selected, human_label_dir, args.class_names).run()

    needs_detector_dataset = not (args.skip_warmup and args.skip_semi and args.skip_final_predict)
    if needs_detector_dataset:
        warmup_yaml, labelled_train, labelled_val = prepare_warmup_dataset(
            selected,
            human_label_dir,
            warmup_dataset_dir,
            args.class_names,
            args.labelled_val_ratio,
        )
    else:
        warmup_yaml, labelled_train, labelled_val = None, [], []

    if not args.skip_warmup:
        print("[DAGS] Stage 1/4: supervised Teacher warm-up")
        warmup_best = train_yolo_model(
            initial_weight=Path(args.yolo_pretrained),
            dataset_yaml=warmup_yaml,
            project=work_dir / "runs",
            run_name="teacher_warmup",
            epochs=args.warmup_epochs,
            imgsz=args.yolo_imgsz,
            batch=args.yolo_batch,
            device=args.yolo_device,
            workers=args.yolo_workers,
            patience=args.yolo_patience,
            strong_augmentation=False,
        )
        copy_weight(warmup_best, warmup_teacher)
    elif not warmup_teacher.exists():
        raise FileNotFoundError(f"--skip-warmup was used but Teacher weight is missing: {warmup_teacher}")

    current_teacher = warmup_teacher
    completed_rounds = 0

    if not args.skip_semi:
        print("[DAGS] Stage 2/4: semi-supervised Teacher-Student rounds")
        for round_index in range(1, args.semi_rounds + 1):
            round_dir = rounds_dir / f"round_{round_index:02d}"
            pseudo_dir = round_dir / "pseudo_labels"
            dataset_dir = round_dir / "dataset"
            threshold = linear_threshold(
                args.pseudo_conf_start,
                args.pseudo_conf_end,
                round_index,
                args.semi_rounds,
            )
            class_thresholds = parse_class_thresholds(
                args.pseudo_class_thresholds,
                threshold,
                len(args.class_names),
            )
            print(
                f"[DAGS][Round {round_index}] Teacher pseudo-label threshold={threshold:.3f}; "
                f"class thresholds={class_thresholds}"
            )
            pseudo_records = generate_pseudo_labels(
                teacher_weight=current_teacher,
                unlabeled_images=unlabeled_images,
                pseudo_dir=pseudo_dir,
                imgsz=args.yolo_imgsz,
                device=args.yolo_device,
                iou=args.yolo_iou,
                default_threshold=threshold,
                class_thresholds=class_thresholds,
                min_area_ratio=args.pseudo_min_area,
                max_area_ratio=args.pseudo_max_area,
                min_boxes_per_image=args.pseudo_min_boxes,
            )
            if not pseudo_records:
                print(
                    f"[DAGS][Round {round_index}][Warning] no reliable pseudo-labelled image; "
                    "stopping semi-supervised rounds and keeping the current Teacher."
                )
                break

            dataset_yaml = prepare_combined_dataset(
                labelled_train=labelled_train,
                labelled_val=labelled_val,
                human_label_dir=human_label_dir,
                unlabeled_by_stem=unlabeled_by_stem,
                pseudo_label_dir=pseudo_dir,
                pseudo_records=pseudo_records,
                dataset_dir=dataset_dir,
                class_names=args.class_names,
                labelled_repeat=args.labelled_repeat,
                pseudo_repeat=args.pseudo_repeat,
            )
            student_best = train_yolo_model(
                initial_weight=current_teacher,
                dataset_yaml=dataset_yaml,
                project=work_dir / "runs",
                run_name=f"student_round_{round_index:02d}",
                epochs=args.semi_epochs,
                imgsz=args.yolo_imgsz,
                batch=args.yolo_batch,
                device=args.yolo_device,
                workers=args.yolo_workers,
                patience=args.yolo_patience,
                strong_augmentation=True,
            )
            teacher_saved = ema_merge_checkpoints(
                teacher_weight=current_teacher,
                student_weight=student_best,
                output_weight=round_dir / "teacher_ema.pt",
                decay=args.ema_decay,
            )
            current_teacher = teacher_saved
            completed_rounds = round_index
            save_state(
                state_path,
                {
                    "completed_rounds": completed_rounds,
                    "current_teacher": str(current_teacher),
                },
            )
    else:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            teacher_candidate = Path(state.get("current_teacher", ""))
            if teacher_candidate.exists():
                current_teacher = teacher_candidate
            completed_rounds = int(state.get("completed_rounds", 0))
        elif final_teacher.exists():
            current_teacher = final_teacher
        else:
            print("[DAGS][Warning] --skip-semi used without prior round state; using warm-up Teacher.")

    copy_weight(current_teacher, final_teacher)
    print(f"[DAGS] final Teacher: {final_teacher}")
    print(f"[DAGS] completed semi-supervised rounds: {completed_rounds}")

    if not args.skip_final_predict:
        print("[DAGS] Stage 3/4: final Teacher creates txt files for every original image")
        write_final_detection_labels(
            final_teacher=final_teacher,
            all_images=all_images,
            labelled_stems=sorted(selected_stems),
            human_label_dir=human_label_dir,
            output_dir=det_label_dir,
            imgsz=args.yolo_imgsz,
            device=args.yolo_device,
            conf=args.final_conf,
            iou=args.yolo_iou,
        )
        print(f"[DAGS] final labels: {det_label_dir}")
    elif not det_label_dir.is_dir():
        raise FileNotFoundError(
            f"--skip-final-predict was used but final label directory is missing: {det_label_dir}"
        )

    if not args.skip_reconstruction:
        print("[DAGS] Stage 4/4: DAGS reconstruction")
        run_reconstruction(args, det_label_dir, extra_train_args)


if __name__ == "__main__":
    main()
