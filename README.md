# NORD-FKB
NORD: Norwegian geospatial Object Reconstruction Dataset - FKB edition




# Miro-board

The Miro-board (pass: kartai1234) is here: https://miro.com/app/board/uXjVL5KvO8Q=/?share_link_id=695142389210

# Project Google Drive folder

https://drive.google.com/drive/folders/12oujXtB8r6a6xlx8Ronij_WT-fdcRiz0?usp=drive_link

# Goals
1. Develop a public / open State-of-the-art benchmark dataset for Norwegian aerial images and precise map data (FKB)
* Based on the open GeoVekst research data (7 areas à ~3km^2)
* All polygon-FKB object categories
* Both for object detection _and_ object segmentation
* Balanced / representative
* As correct as possible

2. Make and publish a developer framework for evaluation
3. Host a competition?
4. Publish a peer-reviewed academic paper with the benchmark dataset
5. Conference / seminar on geospatial benchmark datasets (autumn 2025) (geoai:hub, GI Norden ++)

## Step 1: New Norwegian map data benchmark
* Orthophotos with (not adjusted) FKB-data
* True ortophotos with FKB-data (more precise)
  * Make it from an image matched dataset. 

## Step 2: Hidden validation API-endpoint
* Make an endpoint for users to validate their results on unseen data
* Unknown geographic area

## Step 3: State-of-the-art FKB "reconstruction" benchmark for Norway
* Sander Jyhne is working on this as part of his PhD
* Single aerial images and 3D FKB-data
* 3D object detection
* 3D Instance segmentation

# Task list

**Notes: 13th februar**
* [Marianne]: The vector data is last updated in one single project. And does not match with the newest imagery. 
   * Possible to split the vector data to the matching image?
   * Goal: To find the best matching image for the (subset) vector data
   * Ben: would need to change the production routine (FYSAK) to a postgis/vector data from "Marianne"
* [Aditya]: Test the COCO labels and image from the FYSAK production (drive folder)
* [Aditya]: Read several papers, one very good with github repo. Will sync with Sander on further work
* [Ivar]: Image matching from point clouds is probably not good enough.
* [Alex]: Bergen agreed to release an area under MIT license. 


**Notes: Jan 24th: Data-access and preparation**
* [Marianne] works on QA on FKB-datasets
* Stavanger agrees to MIT license dataset
* [Ben] almost finished developing coco-support in FYSAK. Needs to be tested. 
* [Aditya] Surveyed some research articles. Works further on state-of-the-art
* [Ivar] works further on image matching technique for true ortophoto generation
* [Alex] push further on Bergen to release a dataset

**Notes: January 10th: Data-access**

* [Ben] Upload ortophotos with FKB polygons to the Github-repo
* [Alex] enable lfs on git repo
* [Alex] Can Marianne do a quality assessment on the FKB-datasets?
* [Alex] Contact Bergen and Stavanger to provide small geographic test areas
* [Ben] Develop code in FYSAK that outputs CoCo segmentation and CoCo object detection format to image tiles (https://cocodataset.org/#format-data and https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)
* [Ivar] Ben and Ivar produce true ortophotos from images using image matched point clouds. Need validation. Remember document steps!
* [Sander] Develop python-library that makes it easy to use the benchmark and a sample notebook.

**Milestone: January 30th: Data-access**

* ~~[Alex] Github repo~~
* ~~[Marianne]: Share some samples of: true orthophotos, regular orthophotos, FKB-objects (surface/polygons)~~
* [Aditya] Research state-of-the-art for benchmarks on object detection and segmentation of aerial imagery
* [Aditya] Statistics and analysis of the dataset
* ~~[Ivar] Get access to the open data~~
   * 7 regions in Norway
   * Ortophotos, FKB-data as geojson/gml,
* [Ivar] Generate true orthophotos using image matching
* [Ben] Create coco-format segmentation labels from the FKB-objects in image coordinate system (masks)
* [Marianne -> Ben / Sander] Create object detection labels (bbox) from the FKB-objects
   * Oriented bounding boxes - yolo11??
   * Coco-format / Yolo text format

 **Milestone: February 27th: Evaluation framework**

 **Milestone: March 27th: Developer tools and developer testing**

 **Milestone: April 17th: Publish benchmark and paper**

 **Milestone: Mai 5th: Publish developer competition (host hybrid hackathon?)**

 **Milestone: september: GeoAI seminar: Segmentation and Object detection from aerial images**

**Step 2: Benchmark of 3D map data reconstruction from overlapping aerial images**
