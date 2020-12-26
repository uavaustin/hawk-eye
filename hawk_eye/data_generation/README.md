# Data Generation

We create large batches of fake data to train our models.


## Real Data

Ocasionally, we acquire real-life data from the drone. This is when we have test flights
or competition. There are a few things we can do with this data, but since right now
(in 2020) we don't have a lot of real data, we can't use all of it to train. We need to
keep some set aside so we known how well the models are doing during training.


### Processing Real Data

There are a few steps to generating a COCO dataset of real data.

1. Call `hawk_eye/data_generation/slice_image.py` to create tiles to label.

2. Upload the folder of tiles to a labeling tool. The best one in the browser right now
is https://www.makesense.ai/.

3. After labeling the data, download the by clicking
"Actions" -> "Export Annotations" -> "Single CSV file."

4. Call `hawk_eye/data_generation/process_labels.py` with the labels and tiles.
