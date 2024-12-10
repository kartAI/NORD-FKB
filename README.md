# NORD-FKB
NORD: Norwegian geospatial Object Reconstruction Dataset - FKB edition




# Miro-board

The Miro-board (pass: kartai1234) is here: https://miro.com/app/board/uXjVL5KvO8Q=/?share_link_id=695142389210

# Goals
1. Develop a public / open State-of-the-art benchmark dataset for Norwegian aerial images and precise map data (FKB)
* All polygon-FKB object categories
* Both for object detection _and_ object segmentation
* Balanced / representative
* As correct as possible

2. Make and publish a developer framework for evaluation
3. Publish a peer-reviewed academic paper with the benchmark dataset
4. Conference / seminar on geospatial benchmark datasets (autumn 2025) (geoai:hub, GI Norden ++)

## Step 1: New Norwegian map data benchmark
* Orthophotos with (not adjusted) FKB-data
* True ortophotos with FKB-data (more precise)

## Step 2: State-of-the-art FKB "reconstruction" benchmark for Norway
* Single aerial images and 3D FKB-data
* 3D object detection
* 3D Instance segmentation

# Task list
**Milestone: January 30th: Data-access**

* [Alex] Github repo
* [Marianne]: Share some samples of: true orthophotos, regular orthophotos, FKB-objects (surface/polygons)
* [Aditya] Research state-of-the-art for benchmarks on object detection and segmentation of aerial imagery
* [Ivar] Get access to the open data
   * 7 regions in Norway
   * Ortophotos, FKB-data as geojson/gml,
* [Ivar] Generate true orthophotos
* [Ben] Create segmentation labels from the FKB-objects in image coordinate system (masks)
* [Marianne -> Ben / Sander] Create object detection labels (bbox) from the FKB-objects
   * Oriented bounding boxes - yolo11??
   * Coco-format / Yolo text format
* [Aditya] Statistics and analysis of the dataset

 **Milestone: February 27th: Evaluation framework**

 **Milestone: March 27th: Developer tools and developer testing**

 **Milestone: April 17th: Publish benchmark and paper**

 **Milestone: Mai 5th: Publish developer competition (host hybrid hackathon?)**

 **Milestone: september: GeoAI seminar: Segmentation and Object detection from aerial images**

**Step 2: Benchmark of 3D map data reconstruction from overlapping aerial images**
