
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2


Box = Tuple[int, float, float, float, float]


class BoxAnnotator:
    def __init__(self, image_paths: Sequence[Path], label_dir: Path, class_names: Sequence[str]):
        if not image_paths:
            raise ValueError("No images were selected for annotation.")
        if not class_names:
            raise ValueError("At least one class name is required.")
        self.image_paths = list(image_paths)
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names)
        self.index = 0
        self.current_class = 0
        self.boxes: List[Box] = []
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.original = None
        self.display_scale = 1.0
        self.window_name = "DAGS Interactive YOLO Annotator"

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / (image_path.stem + ".txt")

    def _load_labels(self, image_path: Path) -> List[Box]:
        result: List[Box] = []
        path = self._label_path(image_path)
        if not path.exists():
            return result
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            if 0 <= cls < len(self.class_names):
                result.append((cls, xc, yc, w, h))
        return result

    def _save_labels(self) -> None:
        image_path = self.image_paths[self.index]
        lines = [f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}" for cls, xc, yc, w, h in self.boxes]
        self._label_path(image_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[Annotator] saved {len(lines)} boxes: {self._label_path(image_path)}")

    def _load_current(self) -> None:
        image_path = self.image_paths[self.index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        self.original = image
        self.boxes = self._load_labels(image_path)
        self.drag_start = None
        self.drag_end = None
        h, w = image.shape[:2]
        self.display_scale = min(1.0, 1500.0 / max(w, 1), 850.0 / max(h, 1))

    def _mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        del flags, param
        ox = int(round(x / self.display_scale))
        oy = int(round(y / self.display_scale))
        h, w = self.original.shape[:2]
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (ox, oy)
            self.drag_end = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = (ox, oy)
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                self.boxes.append((self.current_class, xc, yc, bw, bh))
            self.drag_start = None
            self.drag_end = None

    def _draw(self):
        canvas = self.original.copy()
        h, w = canvas.shape[:2]
        for cls, xc, yc, bw, bh in self.boxes:
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f"{cls}:{self.class_names[cls]}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 200, 255), 2)
        title = (
            f"[{self.index + 1}/{len(self.image_paths)}] {self.image_paths[self.index].name} | "
            f"class={self.current_class}:{self.class_names[self.current_class]} | "
            "drag:box  0-9:class  u:undo  s:save  n:next  p:prev  q:finish"
        )
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if self.display_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.display_scale, fy=self.display_scale,
                                interpolation=cv2.INTER_AREA)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse)
        self._load_current()
        while True:
            cv2.imshow(self.window_name, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if ord("0") <= key <= ord("9"):
                class_id = key - ord("0")
                if class_id < len(self.class_names):
                    self.current_class = class_id
            elif key == ord("u"):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord("s"):
                self._save_labels()
            elif key == ord("n"):
                self._save_labels()
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                    self._load_current()
                else:
                    print("[Annotator] already at the last selected image.")
            elif key == ord("p"):
                self._save_labels()
                if self.index > 0:
                    self.index -= 1
                    self._load_current()
            elif key in (ord("q"), 27):
                self._save_labels()
                break
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    BoxAnnotator(args.images, args.label_dir, args.classes).run()


if __name__ == "__main__":
    main()
