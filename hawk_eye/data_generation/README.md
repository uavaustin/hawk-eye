#### Contents
* `config.yaml`: The file containing many of the tweakable parameters for data
generation.
* `create_clf_data.py`: Creates classification data. This is binary data, in other words,
only two classes: target or background.
* `create_detection_data.py`: Create detection data for the object detection models.
The data will be saved as a COCO formatted archive.
* `create_shape_combinations.py`: **Experimental** script for generating all possible combinations of targets/shapes/colors.
* `generate_config.py`: The config file which references `config.yaml`.
