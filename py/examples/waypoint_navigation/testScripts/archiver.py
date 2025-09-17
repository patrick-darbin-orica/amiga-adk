from luxonis_ml.nn_archive.archive_generator import ArchiveGenerator
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from pathlib import Path
import json

blob_path = Path("/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/detection/yolov8nCones_openvino_2022.1_5shave.blob")
cfg_path  = Path("/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/testScripts/config.json")
out_dir   = Path("/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/testScripts/nnarchives")
out_dir.mkdir(parents=True, exist_ok=True)

cfg = json.loads(cfg_path.read_text())
cfg["config_version"] = CONFIG_VERSION
cfg["model"]["metadata"]["name"] = "yolov8nCones"
cfg["model"]["metadata"]["path"] = blob_path.name   # filename INSIDE the archive (not absolute)

# IMPORTANT: do NOT overwrite cfg["model"]["heads"] here

gen = ArchiveGenerator(
    archive_name="yolov8nCones",
    save_path=str(out_dir),                  # directory; the tool writes yolov8nCones.tar.xz here
    cfg_dict=cfg,
    executables_paths=[str(blob_path)]       # files to include at archive root
)
gen.make_archive()
print(out_dir / "yolov8nCones.tar.xz")
