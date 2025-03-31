

# **Table of Contents**
- [**Table of Contents**](#table-of-contents)
- [GIRAFFE: AI-Powered Giraffe Re-Identification for Conservation](#giraffe-ai-powered-giraffe-re-identification-for-conservation)
  - [**Key Features**](#key-features)
  - [**Why It Matters**](#why-it-matters)
  - [**Partnership**](#partnership)
  - [**Overview**](#overview)
  - [**System Components**](#system-components)
    - [**Image Preprocessing and Representations**](#image-preprocessing-and-representations)
    - [**Vector Similarity Search**](#vector-similarity-search)
    - [**User Interface**](#user-interface)
      - [**1. Review Re-identified Known Individuals**](#1-review-re-identified-known-individuals)
      - [**2. Review Unknown Individuals**](#2-review-unknown-individuals)
      - [**3. Review Partitioning of Unknown Individuals**](#3-review-partitioning-of-unknown-individuals)
  - [**Performance Evaluation and Benchmarking**](#performance-evaluation-and-benchmarking)
    - [**Accuracy**](#accuracy)
    - [**Runtime**](#runtime)
  - [**Software Dependencies**](#software-dependencies)
  - [**Project Codes and Data Overview**](#project-codes-and-data-overview)
    - [**Codes**](#codes)
      - [**Main Directory | User Interface**](#main-directory--user-interface)
      - [**Subdirectory | Object Detection**](#subdirectory--object-detection)
      - [**Subdirectory | Identification Pipeline**](#subdirectory--identification-pipeline)
      - [**Subdirectory | configs**](#subdirectory--configs)
      - [**Subdirectory | utils**](#subdirectory--utils)
      - [**Subdirectory | st\_pages**](#subdirectory--st_pages)
      - [**Subdirectory | app**](#subdirectory--app)
    - [**Data**](#data)
      - [**Main Directory**](#main-directory)
      - [**Subdirectory | models**](#subdirectory--models)
      - [**Subdirectory | query\_images**](#subdirectory--query_images)
      - [**Subdirectory | processed\_images**](#subdirectory--processed_images)
      - [**Subdirectory | reference\_dir**](#subdirectory--reference_dir)
      - [**Subdirectory | query\_dir**](#subdirectory--query_dir)
      - [**Key Data Files**](#key-data-files)
  - [**How to Train and Use Giraffe Torso Detection Model?**](#how-to-train-and-use-giraffe-torso-detection-model)
  - [**How to Process an Annotated Batch of Images as Reference Catalog?**](#how-to-process-an-annotated-batch-of-images-as-reference-catalog)
  - [**How to Process a Batch of Images for Querying Against a Catalog?**](#how-to-process-a-batch-of-images-for-querying-against-a-catalog)
    - [**Step 1: Upload a New Batch of Images and Reference Catalog**](#step-1-upload-a-new-batch-of-images-and-reference-catalog)
    - [**Step 2: Set Up .env File**](#step-2-set-up-env-file)
    - [**Step 3: Create a Metadata Table**](#step-3-create-a-metadata-table)
    - [**Step 4: Execute the Codes**](#step-4-execute-the-codes)
      - [**Option 1: Using the User Interface**](#option-1-using-the-user-interface)
      - [**Option 2: Using Python Scripts**](#option-2-using-python-scripts)
    - [**Step 5: Monitor and Export Results**](#step-5-monitor-and-export-results)
  - [**Default Hyperparameters and Configuration Settings**](#default-hyperparameters-and-configuration-settings)
  - [**Updates in Query Metadata File as Pipeline Progresses**](#updates-in-query-metadata-file-as-pipeline-progresses)
    - [New columns | conducting image preprocessing](#new-columns--conducting-image-preprocessing)
    - [New columns | obtaining image representations via SIFT](#new-columns--obtaining-image-representations-via-sift)
    - [New columns | running re-identification matching algorithm](#new-columns--running-re-identification-matching-algorithm)
    - [New columns | partitioning unknown items](#new-columns--partitioning-unknown-items)
    - [New columns | evaluating accuracy metrics if ground truth available](#new-columns--evaluating-accuracy-metrics-if-ground-truth-available)
    - [New columns | expert review and refinement of AI results](#new-columns--expert-review-and-refinement-of-ai-results)
    - [New columns | updating reference catalog with processd query images](#new-columns--updating-reference-catalog-with-processd-query-images)
  - [**Licensing**](#licensing)


# GIRAFFE: AI-Powered Giraffe Re-Identification for Conservation  

Accurate and scalable wildlife re-identification is essential for biodiversity monitoring and conservation efforts. **GIRAFFE** (Generalized Image-based Re-Identification using AI for Fauna Feature Extraction) is an advanced AI-driven system designed to automate the identification of individual giraffes, with potential applications for other species.  

## **Key Features**  
- **AI-Powered Identification**: Leverages local feature matching for high-accuracy individual recognition.  
- **Scalability**: Efficiently processes large datasets containing thousands of images.  
- **User-Friendly Interface**: Accessible to both technical and non-technical users.  
- **Automated Catalog Updates**: Reduces manual effort required for validating matches.  
- **Support for Conservation**: Aids in tracking endangered species and studying population dynamics.  

## **Why It Matters**  
Traditional re-identification methods require extensive manual work, making large-scale biodiversity studies time-consuming and error-prone. **GIRAFFE** automates key steps, enabling conservationists and researchers to efficiently curate datasets, analyze migration patterns, and develop data-driven conservation strategies—all while maintaining accuracy and interpretability.  

By streamlining population tracking, **GIRAFFE** enhances conservation efforts and supports biodiversity research, contributing to the long-term protection of giraffe populations in the wild.


## **Partnership**
This project was developed at the Microsoft AI for Good Lab in collaboration with Derek E. Lee, Ph.D., a quantitative ecologist, population biologist, and the Principal Scientist at the Wild Nature Institute to support [Masai Giraffe Conservation Project](https://www.wildnatureinstitute.org/giraffe.html).

## **Overview**  

We present a unified AI-driven framework for accurate and efficient wildlife re-identification. The system integrates deep learning-based computer vision models for image preprocessing, vector indexing and search libraries for scalable retrieval, and advanced matching algorithms within an interactive user interface. This design enables large-scale visualization, expert-in-the-loop validation, and iterative refinement.  

The AI pipeline supports end-to-end management of a matching project and consists of the following components:  

1. **Computer Vision Models for Image Preprocessing**: Perform giraffe segmentation and torso detection to remove unnecessary background noise.  
2. **Image Descriptor Creation**: Generate key points and descriptors for each preprocessed giraffe image.  
3. **Re-identification Matching Algorithm**: Compare preprocessed image descriptors against a reference dataset of previously identified giraffes to recognize known individuals.  
4. **Unknown Items Partitioning Algorithm**: Cluster unidentified giraffe images by comparing them against each other, ensuring unique labeling of new individuals.  
5. **Human Expert Validation and Intervention**: Enable expert review and refinement of AI-generated results when needed.  
6. **Reference Dataset Update**: Automatically update the reference dataset with new matching results, improving future re-identifications.  

Key contributions of our system include leveraging mode statistics to optimize matching criteria, implementing a distributed indexing and sharding strategy for robust retrieval, and integrating vector search libraries for efficient nearest-neighbor queries. Through extensive parameter optimization, the system consistently achieves over **90% accuracy across seven standard machine learning metrics**, with **re-identification accuracy reaching 99%**. Each query is processed in under two seconds, even against a catalog containing thousands of images.  

Evaluated on the Masai giraffe dataset, our approach enhances reliability by combining automated processing with expert oversight, enabling **accurate individual tracking and long-term conservation efforts**.  


While technical users can run individual scripts separately, we provide a **comprehensive User Interface (UI)** to make the solution accessible for non-technical users. The UI allows users to interact with each module independently, visualize matching results, and validate and refine outputs before updating the database. It simplifies execution by running scripts via button clicks within a **tmux session**, ensuring seamless logging for both standard and error outputs. Non-technical users can operate the system **without needing to interact with the terminal**. This interface effectively functions as a **batch query management tool**, streamlining the processing of giraffe photos after each survey and data collection phase to support biodiversity research on **population trends and survival rates**.  

The **computer vision model** focuses on detecting the giraffe’s torso, minimizing background interference to enhance accuracy. Processed results are saved, creating a **reusable pool of image descriptors** for future identification tasks. Currently, we use **SIFT (Scale-Invariant Feature Transform)** to generate key points and descriptors for the matching algorithm, but the pipeline is designed to be adaptable to other feature extraction methods. To enable **efficient retrieval at scale**, we integrate **FAISS (Facebook AI Similarity Search)** allowing for fast nearest-neighbor searches in both re-identification and partitioning algorithms.

## **System Components**
### **Image Preprocessing and Representations**  

Our system employs advanced computer vision algorithms for data preprocessing, ensuring high-quality inputs for the matching pipeline. The preprocessing consists of three key stages:  

1. **Giraffe Segmentation**:  
   In the first stage, we use a pretrained [Detectron2 instance segmentation model](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) to segment giraffes from images. This model isolates the **central giraffe**, removing background distractions and improving feature extraction.  

2. **Giraffe Torso Detection**:  
   In the second stage, we use a **fine-tuned model** based on a pretrained [Detectron2 object detection model](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml) to detect giraffe torsos. We created a specialized torso annotation dataset for fine-tuning by aligning cropped giraffe torso images from previous projects with the Wild Nature Institute’s dataset. The intersection of the segmentation model and our customized torso detection model, included in this repository, ensures accurate and consistent giraffe torso detection and segmentation, enhancing the reliability of the re-identification process.   

3. **Reference Catalog Creation**:  
   To enable efficient matching, we construct a **reference dataset** containing annotated images of individual giraffes, focusing exclusively on **torso-only images** with backgrounds removed. Each new query image is compared against this dataset using SIFT, which extracts key points and descriptors for feature matching. If a match is found, the system assigns the corresponding ID; otherwise, a new ID can be generated, and the reference dataset is updated accordingly.



### **Vector Similarity Search**  
We use the FAISS library for efficient similarity search and image descriptor retrieval. Training the FAISS index using `IndexHNSWFlat` on a reference dataset of approximately **26,000 images** containing approximately **37 million** descriptor vectors takes around **11 minutes**. Keep this overhead in mind when planning your workflow. To enhance accuracy and robustness, we employ a **distributed indexing and merging approach**, ensuring efficient and reliable similarity search across large-scale datasets.  


### **User Interface**  

The user interface (UI) is designed as a **multi-functional tool** that simplifies batch image processing and large-scale visualization. To ensure seamless collaboration between AI automation and expert oversight, the UI provides **three key visualization checkpoints** following the initial image matching process:

#### **1. Review Re-identified Known Individuals**  
This checkpoint allows experts to verify AI-generated matches by comparing each query image with its corresponding reference images. The interface displays preprocessed query images alongside potentially multiple reference images of the same giraffe, ensuring contextual clarity. Key functionalities include:
- **Navigation through multiple reference images** to verify matches.
- **Reviewing the top three AI-recommended matches** with detailed results displayed in a table.
- **Accepting AI-generated matches** for individual queries or in bulk if no expert review is required.
- **Skipping low-quality images** to streamline validation.
- **Rejecting incorrect matches** to refine the database and improve accuracy.

#### **2. Review Unknown Individuals**  
For cases where no matches were found in the reference dataset, this checkpoint allows experts to review and assign new IDs. The UI presents both the preprocessed image alongside the rejected matched item side by side, providing multiple action options: 
- **Reversing a rejected match** by accepting an AI-suggested image (useful if the cutoff threshold was too strict).  
- **Assigning a new ID** to an individual when no match is found.  
- **Skipping low-quality images** to maintain dataset quality.  
- **Auto-labeling all unmatched individuals** for efficient batch processing if no expert review is required.  

#### **3. Review Partitioning of Unknown Individuals**  
After unknown individuals have been identified, the partitioning algorithm clusters similar individuals together. This checkpoint enables experts to:  
- **Compare clustered image groups side by side** to validate partitioning accuracy.  
- **Ensure distinct groups represent unique individuals** before assigning labels.


## **Performance Evaluation and Benchmarking**
### **Accuracy**
In the giraffe re-identification task, we classify existing giraffes in the query input batch as positive items and new giraffes as negative items. To evaluate re-identification accuracy, we compute standard binary classification metrics. Additionally, for the subset of query data with re-identified labels, we report accuracy in the table below. To assess the partitioning accuracy of new, unknown giraffes, we use the Adjusted Rand Index. The results for several data splits on Wild Nature Institute's Masai giraffes are shown below.


| Case                         | Data Split 1 | Data Split 2 | Data Split 3 |
|------------------------------|-------------|-------------|-------------|
| **Reference catalog**        | 20,687      | 15,965      | 7,000       |
| **Query set**                | 4,666       | 4,666       | 4,666       |
| **Unknown items in query set** | 1,470      | 1,505       | 1,990       |
| **Known items in query set**  | 3,196       | 3,161       | 2,676       |
| **Sharding**                 | No / Yes    | No / Yes    | No          |
| **Overall accuracy**         | 95% / 95%   | 94% / 95%   | 95%         |
| **Accuracy (re-identified items)** | 99% / 99% | 99% / 99% | 100%       |
| **Recall (known)**           | 0.94 / 0.95 | 0.93 / 0.95 | 0.93        |
| **Precision (known)**        | 0.98 / 0.97 | 0.98 / 0.97 | 0.98        |
| **F1 score (known)**         | 0.96 / 0.96 | 0.96 / 0.96 | 0.95        |
| **Recall (unknown)**         | 0.95 / 0.93 | 0.97 / 0.93 | 0.98        |
| **Precision (unknown)**      | 0.88 / 0.90 | 0.87 / 0.90 | 0.91        |
| **F1 score (unknown)**       | 0.92 / 0.92 | 0.92 / 0.92 | 0.94        |
| **Adjusted Rand Index (partitioning)** | 0.82 / 0.83 | 0.81 / 0.83 | 0.82  |


### **Runtime**
On a Standard NC6s v3 Azure Linux machine (6 vCPUs, 112 GiB RAM) powered by NVIDIA Tesla V100 GPU, the expected runtime for each step is shown below. Training FAISS index on reference dataset of 25,363 images takes ~11 mins.

| Process in Workflow                                              | Time per Query Image (Seconds) |
|-----------------------------------------------------------------|-------------------------------|
| **Preprocess and save image using segmentation and object detection models** | 1.7                           |
| **Generate image key points and descriptors using SIFT**   | 0.13                          |
| **Conduct query matching for re-identification using trained index** | 0.03                          |
| **Conduct partitioning for unknown items by training a new index** | 0.07                          |
| **Update database with existing items**                   | 0.0075                        |


Total compute time for processing and matching a sample dataset provided by Wild Nature Institute is:
- **Reference dataset (~26,000 images)**: 13 hours
- **Query dataset (~15,000 images)**: 8 hours


## **Software Dependencies**

Set up the environment by creating a new conda environment and run:
```
conda env create -f environment.yaml
conda activate giraffe
pip install -r requirements.txt
```

## **Project Codes and Data Overview**

### **Codes**  

```
code_directory/
|
|__ configs/
|   |__ __init__.py
|   |__ config_matching.py
|   |__ config_vision.py
|
|__ utils/
|   |__ __init__.py
|   |__ helpers_matching.py
|   |__ utils_matching.py
|   |__ utils_sharding.py
|   |__ utils_vision.py
|   |__ utils_files.py
|
|__ object_detection/
|   |__ __init__.py
|   |__ trainer.py
|   |__ evaluator.py
|   |__ predictor.py
|
|__ pipeline/
|   |__ __init__.py
|   |__ step_1_run_vision_to_crop_torso.py
|   |__ step_2_create_image_discriptors.py
|   |__ step_3_run_initial_matching.py
|   |__ step_4_partition_new_items.py
|   |__ step_5_evaluate_matching_results.py
|   |__ step_6_update_database.py
|
|__ app/
|   |__ start_giraffe_ui.sh
|   |__ streamlit.service
|
|__ st_pages/
|   |__ st_0_Home.py
|   |__ st_1_Create_Query_Table.py
|   |__ st_2_Preprocess_Images.py
|   |__ st_3_Run_Reidentification.py
|   |__ st_4_Verify_Reidentification.py
|   |__ st_5_Identify_Unknown_Individuals.py
|   |__ st_6_Verify_New_Identifications.py
|   |__ st_7_Update_Catalogue.py
|   |__ st_8_Validate_based_on_Ground_Truth.py
|   |__ st_9_Visualize_Single_Image.py
|
|__ app.py
|__ mount_blob_gen2.sh
|__ environment.yaml
|__ requirements.txt
|__ README.md
|__ setup_pipeline.sh
|__ .env
```

#### **Main Directory | User Interface**  
- **`/app.py`**  
  A Streamlit-based user interface to streamline interaction with the pipeline. Allows users to run AI models, visualize and manage data, review results, and provide feedback without needing terminal commands.  

- **`/mount_blob_gen2.sh`**  
  A shell script to mount Azure Blob Storage (Gen2) to your virtual machine, enabling seamless access to data stored in the cloud.  

- **`/environment.yaml`**  
  Create a new conda environment.

- **`/requirement.txt`**  
  Required libraries and framework to run the codes.

- **`/setup_pipeline.sh`**  
  A general-purpose script for automating the pipeline execution. Can be customized to run specific steps or the entire workflow.  

#### **Subdirectory | Object Detection**
- **`/object_detection/trainer.py`**  
  Used to fine tune torso detection model using Detectron2 library.  

- **`/object_detection/evaluator.py`**  
  Used to evaluate torso detection model on validation data when ground truth bounding boxes available.   

- **`/object_detection/predictor.py`**  
  Used for inference, to apply giraffe segmentation and torso detection model on any image data when ground truth bounding boxes not available.    

#### **Subdirectory | Identification Pipeline**
- **`/pipeline/step_1_run_vision_to_crop_torso.py`**  
  Processes images using the vision model. Includes functions to load images, preprocess them, and run the giraffe detection algorithms.  

- **`/pipeline/step_2_create_image_discriptors.py`**  
  Generates image descriptors using the processed data from Step 1 using SIFT. These descriptors are stored for later use in matching.  

- **`/pipeline/step_3_run_initial_matching.py`**  
  Matches the image descriptors generated in step 2 against a reference database using FAISS to build a similarity search index. Outputs the top matches for each input query image.  

- **`/pipeline/step_4_partition_new_items.py`**  
  Runs a partitioning algorithm to label the unmatched photos from step 3. This step takes into account that there may be multiple photos of a new giraffe item in the query batch, and it identifies clusters, labeling them based on the last available label in the reference data to avoid duplicates. Additionally, this step prepares the query records for updating the data in step 5.
  
- **`/pipeline/step_5_evaluate_matching_results.py`**  
  Evaluates the matching results by comparing them against ground truth labels when available. This step calculates various metrics, including accuracy, precision, recall, and others, to assess the performance of the matching process.

- **`/pipeline/step_6_update_database.py`**  
  Updates the reference database with new data and adjustments done in previous steps. Ensures that the reference data remains up to date for future surveys.

#### **Subdirectory | configs**

- **`/configs/config_matching.py`**  
  Configuration file containing parameters, paths, and settings related to the matching algorithms shared across different stages of the pipeline.  

- **`/configs/config_vision.py`**  
  Configuration file containing parameters, paths, and settings for training, inference, prediction related to object detection model shared across different stages of the pipeline.  

#### **Subdirectory | utils**
- **`/utils/helpers_matching.py`**  
  A utility module with helper functions, such as file handling, data processing, and logging, related to the matching algorithms, used throughout the pipeline.

- **`/utils/utils_matching.py`**  
  A utility module with functions needed to do the matching step, such as a giraffe class to apply vision models, train, read, write faiss index for search, run union-find algorithm for partitioning used throughout the pipeline.

- **`/utils/utils_sharding.py`**  
  A utility module with functions needed to do distributed indexing and merging retrived results to help with robustness in accuracy. 

- **`/utils/utils_vision.py`**  
  A utility module with helper functions, such as file input data processing for training object detection model used throughout the pipeline.

- **`/utils/utils_files.py`**  
  A utility module used in UI scripts for reading files.

#### **Subdirectory | st_pages**
- The `st_pages` folder contains Python scripts used in the user interface app for various stages of the giraffe photo re-identification process, including query table creation, preprocessing, re-identification, validation, catalog updates, and visualization.

#### **Subdirectory | app**
- **`/app/start_giraffe_ui.sh`**  
  Script to run the user interface code.

- **`/app/streamlit.service`**  
  This file needs to be placed in the `/etc/systemd/system` directory to allow the UI to run as an application in the background when the Linux virtual machine starts.
  
### **Data**

```
data_root_directory/
|
|__ models/
|   |__ Base-RCNN-FPN.yaml
|   |__ faster_rcnn_R_101_FPN_3x.yaml
|   |__ mask_rcnn_R_50_FPN_3x.yaml
|   |__ config.yaml
|   |__ model_final_f10217_segmentation.pkl
|   |__ model_final_torso_detection.pth
|   |__ model_initial_torso_detection.pkl
|
|__ query_images/
|   |__ image1.jpg
|   |__ image2.jpg
|   |__ ...
|
|__ query_dir/
|   |__ metadata_query.csv
|   |__ giraffes_query_descriptors.pkl
|   |__ accuracy_results.csv
|   |__ logs/
|       |__ log__err_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log
|       |__ log__std_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log
|       |__ ...
|
|__ reference_dir/
|   |__ metadata_reference.csv
|   |__ giraffes_reference_descriptors.pkl
|
|__ processed_images/
|   |__ original_size/
|       |__ image1_cropped_torso.jpg
|       |__ image2_cropped_torso.jpg
|       |__ ...
|   |__ zoomed_version/
|       |__ image1_cropped_torso_zoomed.jpg
|       |__ image2_cropped_torso_zoomed.jpg
|       |__ ...

```

#### **Main Directory**  
- **`data_root_directory/`**  
  This directory serves as the central hub for all project-related data, including models, processed images, metadata, and logs. By default, it is mapped to a blob storage account using the script `code_directory/mount_blob_gen2.sh`. However, it can also be configured to any local directory by specifying the path in the `.env` file.


#### **Subdirectory | models**

- **`/models/`**  
   Contains the pretrained giraffe torso detection models, configuration files, and model weights.  
   - **`/models/Base-RCNN-FPN.yaml`**: A [base configuration](https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml) for the RCNN model.  
   - **`/models/faster_rcnn_R_101_FPN_3x.yaml`**: Configuration used for [Faster R-CNN](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml) using ResNet-101 with an FPN.  
   - **`/models/mask_rcnn_R_50_FPN_3x.yaml`**: Configuration for [Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) with ResNet-50 and FPN.  
   - **`/models/model_final_torso_detection.pth`**: Final trained model weights for torso detection task.  
   - **`/models/model_initial_torso_detection.pkl`**: A [faster_rcnn_R_101_FPN_3x model](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl) used for initialization.
   - **`/models/model_final_f10217_segmentation.pkl`**: A [mask_rcnn_R_101_FPN_3x model](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl) used for giraffe segmentation.
   - **`/models/config.yaml`**: Final config file for trained model weights for torso detection task.  

#### **Subdirectory | query_images**

- **`/query_images/`**  
   Contains query image files to be used for running the pipeline. The directory name is flexible as the file paths are stored in `data_root_directory/query_dir/metadata_query.csv` relative to the root directory.  

#### **Subdirectory | processed_images**

- **`/processed_images/`**  
   Contains images processed by the vision model. Organized into two subdirectories:  
   - **`/processed_images/original_size/`**: Images cropped to the giraffe torso, retaining their original resolution.  
   - **`/processed_images/zoomed_version/`**: Images cropped and zoomed in on the giraffe torso for better visibility.  

#### **Subdirectory | reference_dir**

- **`/reference_dir/`**  
   Contains metadata and descriptors for the reference dataset used for matching against query images.  
   - **`/reference_dir/metadata_reference.csv`**: Metadata file with columns for image paths, giraffe IDs, and serial numbers.  
   - **`/reference_dir/giraffes_reference_descriptors.pkl`**: Contains descriptors for giraffe torso images in the reference dataset.  

#### **Subdirectory | query_dir**

- **`/query_dir/`**  
   Contains metadata and descriptors for the query dataset as well as logs and accuracy results.  
   - **`/query_dir/metadata_query.csv`**: Metadata file for query images, with paths relative to the root directory. Optionally includes a `"AID2021"` column for ground truth labels.  
   - **`/query_dir/giraffes_query_descriptors.pkl`**: Contains descriptors for cropped giraffe torso images in the query dataset.  
   - **`/query_dir/accuracy_results.csv`**: Contains accuracy metrics computed if ground truth data is available.  

- **`/query_dir/logs/`**  
   Stores log files for each workflow step.  
   - **`/query_dir/logs/log__err_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log`**: Captures errors encountered during script execution.  
   - **`/query_dir/logs/log__std_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log`**: Captures standard output, including runtime details and memory profiling.  


#### **Key Data Files**

- **`/reference_dir/metadata_reference.csv`**  
  Metadata for the reference dataset, with columns:  
  - `"path_relative_to_root"`: Relative path to the image.  
  - `"AID2021"`: Unique giraffe ID.  
  - `"#Serial"`: Serial number for the giraffe image.  

- **`/reference_dir/giraffes_reference_descriptors.pkl`**  
  Stores descriptors for the reference dataset. Format:  
  ```python
  dict[label_id] = ([serial_id_1, ..., serial_id_m], [descriptors_id_1, ..., descriptors_id_m])
  ```  

- **`/query_dir/metadata_query.csv`**  
  Metadata for the query dataset, with columns:  
  - `"path_relative_to_root"`: Relative path to the image.  
  - `"AID2021"` (optional): Ground truth labels for accuracy evaluation.  

- **`/query_dir/giraffes_query_descriptors.pkl`**  
  Stores descriptors for the query dataset. Format:  
  ```python
  dict[image_filename] = image_descriptors
  ```  

- **`/query_dir/accuracy_results.csv`**  
  Generated after running `code_directory/pipeline/step_5_evaluate_matching_results.py` or using the corresponding UI button. Reports accuracy metrics when ground truth data is available in `/query_dir/metadata_query.csv`.  

- **Log Files**:  
  - **`/query_dir/logs/log__err_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log`**: Logs errors for a specific workflow step.  
  - **`/query_dir/logs/log__std_output__<workflow_step>__yyyy-mm-dd_hh-mm-ss.log`**: Logs standard output for a specific workflow step.  


## **How to Train and Use Giraffe Torso Detection Model?**
To train a torso detection model, we used original and cropped images of giraffes to create annotations for an object detection task. The model was fine-tuned to detect giraffe torsos by splitting the data into training, validation, and test sets.

## **How to Process an Annotated Batch of Images as Reference Catalog?**

To create a reference dataset of annotated giraffe images for comparison with query images, follow these steps:

1. **Create a Metadata Query Table** 
    Provide a `metadata_reference.csv` file containing metadata such as label IDs and image paths. Each row corresponds to a giraffe image. The expected columns in the CSV file are:  
    `"path_relative_to_root", "AID2021", "#Serial"`.

2. **Preprocess Images to Crop Torso, Remove Background and Store**  
   Preprocess the images by removing the background and cropping torso to prepare for next steps.

3. **Generate Image Descriptors**  
   Extract key points and descriptors for each image. `partition_vision` should be set to `reference` in `code_directory/configs/config_matching.py`.


## **How to Process a Batch of Images for Querying Against a Catalog?**

### **Step 1: Upload a New Batch of Images and Reference Catalog**
Set up a local directory for storing datasets or use Azure Storage Explorer to upload new images to the mounted storage account. For local storage, define the directory path in the `.env` file; for mounted storage, use `\mnt\`. Existing data can be retained, as query images are processed based on their filenames and paths (relative to the root directory) as listed in the `metadata_query.csv` file.

### **Step 2: Set Up .env File**
```bash
# .env Configuration File  
# Define environment variables for storage and data management  

# Name of the container in the storage account  
container_name = "your-container-name"  

# Name of the storage account  
storage_account_name = "your-storage-account-name"  

# Storage mount type (e.g., 'adls' for Azure Data Lake Storage)  
mount_type = "your-mount-type"  

# Application ID for authentication (if required)  
app_id = "your-app-id"  

# Absolute path to the root data directory (mounted or local)  
data_root_abs_path = "/your/data/root/path/"  
```

### **Step 3: Create a Metadata Table**
Create a `metadata_query.csv` file with at least one mandatory column, `"path_relative_to_root"`, which specifies the paths of the images relative to the root directory, and one optional column, `"AID2021"`, which stores ground truth labels for testing and accuracy metrics. The `metadata_query.csv` file is updated at various stages to monitor results.  

> [!TIP]
> If you are using Azure Storage accounts and running multiple tests with different `metadata_query.csv` files, **do not overwrite existing files** directly using Azure Storage Explorer. Delete the old file before uploading a new one to avoid synchronization issues between the storage and the virtual machine (VM).


### **Step 4: Execute the Codes**
You can execute the AI pipeline either stage-by-stage or via the user interface. `partition_vision` should be set to  `query` in `code_directory/configs/config_matching.py` when running codes to create SIFT descriptors.

#### **Option 1: Using the User Interface**
Access the application by running:  
```bash
streamlit run app.py
```
Alternatively, turn on the VM and navigate to the following link (ensure your IP is added to the network security settings):  
[AI Tool User Interface](http://giraffetrack.northeurope.cloudapp.azure.com:8088/)  

> [!TIP]
> **Important Notes:**  
> - During the visualization and validation step, you can record intermediary actions (approve/reject results). Always save and back up the `metadata_query.csv` file before using the restart button, as it may erase human input columns for a fresh start.  
> - Matching results are visualized once they are generated.

#### **Option 2: Using Python Scripts**
Activate the pre-configured Conda environment named `giraffe`:  
```bash
conda activate giraffe
```
Run the required Python script:  
```bash
python path\to\your_script.py
```
Check the saved logs for error messages and standard outputs.


### **Step 5: Monitor and Export Results**
After running each stage, two log files are generated to provide insights into performance, outputs, and errors. The `metadata_query.csv` and `metadata_reference.csv` files are refreshed every 1,000 queries, enabling progress tracking.

> [!TIP]
> Tips for Monitoring Experiment Executions on Azure VM and Storage Accounts:
> - Avoid opening log files before a job finishes, as this may interrupt execution.
> - Monitor memory and CPU usage in the Azure portal for the VM to gain insights into the experiment's execution status.

## **Default Hyperparameters and Configuration Settings**
The following hyperparameters are set by default in the configuration files for matching algorithms. These can be adjusted to optimize performance based on your use case or desired results.

1. **faiss_distance_cutoff_re_id** and **faiss_mode_cutoff_re_id**: Control the re-identification algorithms to accept or reject a match.
   - faiss_distance_cutoff_re_id (`inf`): Sets the maximum distance for retrieving nearest neighbors. With `inf`, no distance-based filtering is applied. 
   - faiss_mode_cutoff_re_id (`5`): After distance filtering, only key points occurring at least 10 times are considered. The match with the highest number of such key points is selected as the best match.
  
2. **faiss_distance_cutoff** and **faiss_mode_cutoff**: Control the new items partitioning process.
   - faiss_distance_cutoff (`0.062`): Sets the maximum distance for retrieving nearest neighbors.
   - faiss_mode_cutoff (`4`): After distance filtering, only key points occurring 
   - At least 4 times are considered. The match with the highest number of such key points is selected as the best match.

3. **num_recommended_ids** (`3`): This parameter controls how many IDs are recommended by the system after the comparison.

4. **cropped_img_size** (`512`): This parameter defines the size of the images after cropping before they are fed into SIFT model.
  
5. **n_features** (`1500`): This parameter defines the maximum number of key points found for images when SIFT model applies.


## **Updates in Query Metadata File as Pipeline Progresses**
The `metadata_query.csv` file contains various columns that track the different stages of the pipeline. Each column represents the results from different step as follows:

### New columns | conducting image preprocessing
- **ai_found_torso**: Indicates whether the giraffe's torso was detected.
- **giraffes_count**: Represents the number of giraffes detected in the image.
- **detection_coverage**: The proportion of the image covered by detected giraffe(s).
- **segmentation_coverage**: The proportion of the giraffe(s) covered by the segmentation mask.
- **combined_coverage**: A combined metric of detection and segmentation coverage.

### New columns | obtaining image representations via SIFT
- **descriptors_size**: Indicates the size of the descriptors found using SIFT.

### New columns | running re-identification matching algorithm
- **matching_attempt**: Indicates whether a matching attempt was made.
- **matching_status**: Shows whether a giraffe has been found in the reference dataset. If no match is found, the entry is marked as a new giraffe.
- **matched_img_serial_1**: The serial number of the first matched image.
- **matched_label_1**: The label of the first matched image.
- **matching_mean_dist_1**: The mean distance for the first match.
- **matched_img_serial_2**: The serial number of the second matched image.
- **matched_label_2**: The label of the second matched image.
- **matching_mean_dist_2**: The mean distance for the second match.
- **matched_img_serial_3**: The serial number of the third matched image.
- **matched_label_3**: The label of the third matched image.
- **matching_mean_dist_3**: The mean distance for the third match.

### New columns | partitioning unknown items
- **database_update_status**: Indicates whether row has been processed based on `human_input` column for validation purposes, you can accept all model results and populate `human_input` column.
- **new_id_aligned_with_ref**: Shows all the new ids assigned that are aligned with reference database and can beused for updating.
  
### New columns | evaluating accuracy metrics if ground truth available
- **AID2021**: The ground truth or reference dataset used for comparison during accuracy computation. This is an optional column.
- **out_of_sample**: A computed column that becomes available when the `AID2021` column is present. It indicates whether the item's label in query has been availble in reference data or not.
  
### New columns | expert review and refinement of AI results
These columns record the human intervention stage. By setting `auto_accept_model_matching_results` equal to True in `code_directory/configs/config_matching.py`, you can accept all the model outputs automatically.
- **human_input**: Indicates whether the human reviewer has accepted or rejected the AI results.

### New columns | updating reference catalog with processd query images
- **final_update_status**: Indicates the update status of the reference database.
- **reference_pkl_file**: The updated reference `.pkl` file used for future matching.
- **metadata_reference.csv**: The updated `metadata_reference.csv` file, which includes new matching results and updates to existing entries.

## **Licensing**  
   
This project is licensed under the MIT License. See the LICENSE file for more details.  