# NORD-FKB training dataset creation

## Data sources
**Norge i Bilder** (NiB)   
**FKB data** (SOSI-files or PostGIS)

## Software
**FYSAK ß_2025-05-05 PROFF**, using:  
GDAL 3.10.0  
CURL v8.10  
nlohmann-json 3.11.3  

## Programming language
C++

## Procedure
**Determine the extent**  
Retrieve NiB-metadata for the ortofoto project from NiB’s REST-API. The metadata includes a geometry representing the extent of the project.
The area of interest (AOI) is either the NiB project’s BBOX, or a user-defined area.

**Image input**  
NiB images covering the AOI are identified by creating a polygon from an image’s filename using FYSAK’s Hjkoor-library, and checking for overlap with the AOI’s extent. 
Relevant image files are downloaded using a third-party SDK. 

**Tiling**  
The bounding box of the AOI is then used to define a grid of tiles. The tile size in this case is 512x512 pixels. Real world extent depends on the project’s image resolution.
FKB-features from the SOSI files or PostGIS that lie within the area of interest are then intersected with the tile grid, transforming the data underway if required. Feature type, geometry and extent of overlap with each tile are collected and stored in a container.
The downloaded images are organized in a virtual dataset and image tiles are retrieved for each tile containing features from the FKB data using GDALRasterIO. These, along with the mask images are saved.

**Metadata**  
Metadata structured in accordance with the OGC TrainingDML-AI standard and a COCO-dataset are written using nlohmann-json.
