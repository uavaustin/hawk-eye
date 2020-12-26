""" Contains various configuration settings for data generation. """

import pathlib
import re
import yaml

ARCHIVE_FILENAME = re.compile("^[^.]*")

config = yaml.safe_load((pathlib.Path(__file__).parent / "config.yaml").read_text())
generate_config = config["generate"]

# This is where we are going to store all the assets.
ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

BACKGROUNDS = generate_config.get("backgrounds_archives", [])
BACKGROUNDS_URLS = [f"assets/{bkg}" for bkg in BACKGROUNDS]
BACKGROUNDS_DIRS = [
    ASSETS_DIR / ARCHIVE_FILENAME.match(bkg).group() for bkg in BACKGROUNDS
]

BASE_SHAPES_VERSION = generate_config.get("base_shapes_version", "v1")
BASE_SHAPES_URL = [f"assets/base-shapes-{v}.tar.gz" for v in BASE_SHAPES_VERSION]
FONTS_URL = "assets/fonts.tar.gz"


BASE_SHAPES_DIRS = [ASSETS_DIR / f"base-shapes-{v}" for v in BASE_SHAPES_VERSION]

DATA_DIR = pathlib.Path(__file__).parent / "data"

# Max images to generate per image
MAX_PER_SHAPE = config["generate"]["max_shapes_per_image"]

# Specify number of threads to use for shape generation. Default lets
# the multiprocessing library determine.
NUM_THREADS = config["generate"]["threads"]

# [Shape Specs]
SHAPE_TYPES = config["classes"]["shapes"]

CLF_TYPES = ["background", "target"]

TARGET_COLORS = [
    "white",
    "black",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
]

ALPHA_COLORS = [
    "white",
    "black",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
]

COLORS = {
    "white": [(240, 240, 235)],
    "black": [(5, 5, 5)],
    "gray": [(128, 128, 128)],
    "red": [(188, 60, 60), (255, 80, 80), (255, 0, 0), (154, 0, 0)],
    "blue": [(0, 0, 255), (0, 0, 135)],
    "green": [(64, 115, 64), (148, 255, 148), (0, 255, 0), (0, 128, 4)],
    "yellow": [(225, 221, 104), (255, 252, 122), (255, 247, 0), (210, 203, 0)],
    "purple": [(127, 127, 255), (128, 0, 128)],
    "orange": [
        (153, 76, 0),
        (216, 172, 83),
        (255, 204, 101),
        (255, 165, 0),
        (210, 140, 0),
    ],
}

ALPHAS = config["classes"]["alphas"]

ALPHA_FONT_DIR = ASSETS_DIR / "fonts"
ALPHA_FONTS = [
    ALPHA_FONT_DIR / "Rajdhani" / "Rajdhani-Bold.ttf",
    ALPHA_FONT_DIR / "Gudea" / "Gudea-Bold.ttf",
    ALPHA_FONT_DIR / "Inconsolata" / "Inconsolata-Bold.ttf",
    ALPHA_FONT_DIR / "Open_Sans" / "OpenSans-Bold.ttf",
    ALPHA_FONT_DIR / "Open_Sans" / "OpenSans-SemiBold.ttf",
    ALPHA_FONT_DIR / "News_Cycle" / "NewsCycle-Bold.ttf",
]

OD_CLASSES = SHAPE_TYPES  # + ALPHAS

# [Model Dimensions]
FULL_SIZE = (
    config["inputs"]["full_image"]["width"],
    config["inputs"]["full_image"]["height"],
)
CROP_SIZE = (
    config["inputs"]["cropping"]["width"],
    config["inputs"]["cropping"]["height"],
)
CROP_OVERLAP = config["inputs"]["cropping"]["overlap"]
DETECTOR_SIZE = (
    config["inputs"]["detector"]["width"],
    config["inputs"]["detector"]["height"],
)
PRECLF_SIZE = (
    config["inputs"]["preclf"]["width"],
    config["inputs"]["preclf"]["height"],
)

# Whether to delete full image data when they are converted
DELETE_ON_CONVERT = generate_config.get("delete_on_convert", False)
IMAGE_EXT = generate_config.get("img_ext", ".png")
IMAGE_EXT = f".{IMAGE_EXT.replace('.', '')}"

TARGET_COMBINATIONS = [
    SHAPE_TYPES,
    TARGET_COLORS,
    ALPHAS,
    ALPHA_COLORS,
    [angle for angle in range(0, 360, 45)],
]

# Dataset images and splits
detection = generate_config.get("detection", {})
EMPTY_TILE_PROB = detection.get("empty_slice_probability", 0.0)
DET_TRAIN_OFFSET = detection.get("train_offset", 0)
DET_TRAIN_IMAGES = detection.get("train_images", 10)
DET_VAL_OFFSET = detection.get("val_offset", 0)
DET_VAL_IMAGES = detection.get("val_images", 10)

clf = generate_config.get("classification", {})
CLF_OFFSET = clf.get("offset", 0)
CLF_IMAGES = clf.get("images", 10)
CLF_VAL_FRACTION = clf.get("val_fraction", 0.2)
