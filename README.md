# Retinal Image Quality Sorter

A fast, rule-based tool that sorts retinal (fundus) photos into **Gradeable** and **Ungradeable** folders before they reach a doctor.

In diabetic retinopathy screening, a large share of photos can't be used: they're underexposed, blurry or low-resolution. Catching those automatically means a clinician only reviews images they can actually diagnose from, and patients with a bad photo can be re-imaged on the spot.

I built this during a data-science training program at **SDAIA** (Saudi Data & AI Authority) in 2024, working with retinal screening images from Saudi clinics.

![Example results](docs/examples.png)

## How it works

Every image goes through two checks:

| Check | How | Default limit | Why |
|---|---|---|---|
| **Brightness** | Convert to HSV and average the V (value) channel | below **35 / 255** → ungradeable | Underexposed photos hide the vessels and lesions a doctor needs to see |
| **File size** | Size of the image file on disk | below **109 KB** → ungradeable | Very small files are low-resolution or heavily compressed |

An image is **gradeable** only if it passes both. The brightness check runs first, so the output says *why* each image was rejected.

Images are processed in parallel with a thread pool, so the tool handles a folder of full-resolution photos in about a second.

## Quick start

```bash
pip install -r requirements.txt
python sort_images.py examples
```

Output:

```
gradeable    gradeable.jpg  (passed both checks; brightness=91.9, size=4495.1 KB)
ungradeable  ungradeable_small_file.jpg  (file too small; brightness=44.0, size=102.5 KB)
ungradeable  ungradeable_too_dark.jpg  (too dark; brightness=20.0, size=3567.7 KB)

3 images: 1 gradeable, 2 ungradeable, 0 skipped in 0.11s
```

This creates `sorted/Gradeable/` and `sorted/Ungradeable/` with copies of the images.

### Options

```
python sort_images.py INPUT_DIR
    --output DIR            where to create the Gradeable/Ungradeable folders (default: ./sorted)
    --brightness-limit N    minimum mean brightness, 0-255 (default: 35)
    --size-threshold-kb N   minimum file size in KB (default: 109)
    --move                  move files instead of copying them
    --report results.csv    save each image's brightness, size and decision to a CSV
```

## Results

On the 51-image sample set from the program, the sorter put **29 images in Gradeable and 22 in Ungradeable**, in about one second.

- 12 were rejected for being **too dark**, including a full-resolution 3.6 MB photo that a size-only rule would have accepted.
- 10 were rejected for being **too small**.

The cleaned-up script in this repo reproduces the original notebook's decisions exactly, image for image.

## Limitations and what I'd do next

I'm listing these because they're the most interesting part of the project.

- **File size partly measures the camera, not the photo.** The images came from two cameras: one saves ~4.5 MB files and the other ~100 KB files. The 109 KB threshold therefore acts largely as a camera detector, and a sharp, well-lit photo from the smaller-file camera can be rejected. A better signal would be image resolution, or a sharpness measure such as the variance of the Laplacian.
- **Brightness includes the black border.** Every fundus photo has a black area around the circular retina, so the average is pulled down. Measuring brightness only inside the retina (a circular mask) would be more accurate.
- **The thresholds were tuned by eye** on a small sample. The next step is to measure agreement against doctors' gradability labels and choose thresholds from that.
- **The sample set contained duplicates:** 51 files but only 35 unique images (one photo appears 13 times). The results above count every file.
- **Beyond rules:** the natural next step is a small CNN (e.g. a fine-tuned ResNet18) trained on clinician gradeability labels.

## Project history

1. **v1:** file size only. Anything under 1 MB was considered ungradeable.
2. **v2 (this repo):** added the HSV brightness check, lowered the size threshold to 109 KB, copied files instead of moving them, and parallelised the processing.

## Data

The full dataset is **not** included. It contains clinical screening images and patient information that can't be shared publicly. The `examples/` folder contains three sample images, with camera metadata removed, so the tool can be run out of the box.
