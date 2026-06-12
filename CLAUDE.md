# CLAUDE.md - solar-defect-detector Project Guidelines

## Core Command References

- Activate Environment: `conda activate fasternet`
- Test Config Grammar: `python -c "from ultralytics import YOLO; YOLO('faf_yolov11n.yaml')"`
- Run Light Test: `yolo task=detect mode=train data=coco8.yaml model=faf_yolov11n.yaml epochs=1 imgsz=64 batch=1 device=cpu`
- Lint Codebase: `flake8 ultralytics/nn/tasks.py`

## Development & Custom Module Registration Rules

- Implement all PyTorch neural network modules inside `ultralytics/nn/modules/block.py`. Do NOT create disconnected external scripts unless modularized.
- Expose classes in `ultralytics/nn/modules/__init__.py` inside the `__all__` list.
- Modify model parser logic in `parse_model` within `ultralytics/nn/tasks.py` to route custom modules, ensuring correct dimension/channel tracking via the `ch` list.
- Prioritize memory-efficient operations (avoid dense 2D convolutions in attention paths).
- Code style: Strictly enforce PEP 8 guidelines. Maintain high comments density in math-heavy transformations.
