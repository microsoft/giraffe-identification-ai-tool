# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import json
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes, BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def load_metadata(metadata_dir):
    
    # Load metadata and choose files for which cropped files exists
    metadata_df = pd.read_csv(metadata_dir)
    
    required_columns = ['path_relative_to_root', '#Serial', 'AID2021', 'path', 'cropped_file_exists']

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]

    if missing_columns:
        print(f"Error: The following required columns are missing from metadata_df: {missing_columns}")
        exit(1) 
    
    metadata_df = metadata_df[metadata_df['cropped_file_exists'] == True]
    
    # Print final size information
    print(f"----- filtered metadata dataframe where cropped files exists: {metadata_df.shape[0]} samples, {metadata_df.shape[1]} features")
    
    return metadata_df
    
def merge_metadata_count_giraffes(metadata_df, giraffe_count_coverage_df_dir):
    
    # Load giraffe count data this comes from segmentation model results
    giraffe_count_coverage_df = pd.read_csv(giraffe_count_coverage_df_dir)
    
    # Filter only needed columns from metadata
    merged_df = pd.merge(metadata_df, giraffe_count_coverage_df, on=['#Serial', 'AID2021', 'path'])
    
    # Print final size information
    print(f"----- merged metadata dataframe merged with giraffe count data: {merged_df.shape[0]} samples, {merged_df.shape[1]} features")
    
    return merged_df

def choose_metadata_subset(metadata, giraffe_count=None, data_random_sample=True, small_dataset_serials=[]):
    
    if giraffe_count:
        metadata = metadata.loc[metadata['giraffes_count'] == giraffe_count,:]
        print(f"----- only image with {giraffe_count} giraffe(s) in the image selected for training model.")
    
    if data_random_sample:
        metadata = metadata.sample(min(1000, len(metadata)))
        
    if len(small_dataset_serials) != 0:
        metadata = metadata[metadata['#Serial'].isin(small_dataset_serials)]

    print(f"----- filetered dataset size to be used for training: {metadata.shape[0]} samples, {metadata.shape[1]} features")
    
    return metadata

def build_data_splits(metadata):
    
    # Build data splits
    train_df = metadata[metadata['id-split'] == 'train']
    print(f"----- Training set: {train_df.shape[0]} samples, {train_df.shape[1]} features")

    val_df = metadata[metadata['id-split'] == 'val']
    print(f"----- Validation set: {val_df.shape[0]} samples, {val_df.shape[1]} features")
    
    test_df = metadata[metadata['id-split'] == 'test']
    print(f"----- Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features")
    
    return train_df, val_df, test_df

def convert_annotations_for_visualization(annotations, image_height, image_width):
    """
    Convert annotations to an Instances object formatted for visualization.

    Args:
    - annotations: An Instances object with fields 'gt_boxes' and 'gt_classes'.
    - image_height: Height of the image (extracted from annotations).
    - image_width: Width of the image (extracted from annotations).

    Returns:
    - instances: An Instances object with fields required for visualization.
    """

    gt_boxes = annotations.gt_boxes.tensor.numpy()  # Bounding boxes in [x1, y1, x2, y2] format
    gt_classes = annotations.gt_classes.numpy()  # Class labels

    # Create an Instances object with specified size
    instances = Instances((image_height, image_width))

    # Convert to detectron2's format
    instances.pred_boxes = Boxes(torch.tensor(gt_boxes, dtype=torch.float32))
    instances.pred_classes = torch.tensor(gt_classes, dtype=torch.int64)
    instances.scores = torch.ones((len(gt_boxes),), dtype=torch.float32)  # Dummy scores

    return instances

def save_image_grid_from_dataloader(dataloader, metadata, output_dir, rows=20, cols=5):
    
    output_file=os.path.join(output_dir, "dataloader_grid.pdf")
    json_output=os.path.join(output_dir, "image_mapping.json")
    
    # List to store images
    images_list = []
    image_mapping = {}  # Dictionary to keep track of filenames and their indices

    image_counter = 0  # Counter to number the images

    # Load a TrueType font and set the font size
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Example path on Linux
    font_size = 40  # Set the desired font size
    font = ImageFont.truetype(font_path, font_size)

    padding = 20  # Space between images

    # Get images from the dataloader
    for batch in dataloader:
        # Extract the first image and annotations from each batch
        img_tensor = batch[0]['image']  # Tensor format (C, H, W)
        annotations = batch[0]['instances']  # Annotations
        filename = batch[0].get("file_name", f"image_{image_counter}.png")  # Get filename or use a default name
        
        # Track the filename and its index
        image_mapping[image_counter] = filename
        image_counter += 1

        print(f"Processing {filename}")
        print(img_tensor.shape)
        print(annotations)
        
        image_height, image_width = img_tensor.shape[1], img_tensor.shape[2]

        annotations_reformatted = convert_annotations_for_visualization(annotations, image_height, image_width)
        print(annotations_reformatted)
        
        # Convert tensor to NumPy for visualization
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = img[:, :, ::-1]  # Convert BGR to RGB
        
        # Create visualizer for the image
        visualizer = Visualizer(img, metadata=metadata, scale=1)
        vis = visualizer.draw_instance_predictions(annotations_reformatted)

        # Convert to PIL image
        img_pil = Image.fromarray(vis.get_image())
        
        # Draw the image number on the image
        draw = ImageDraw.Draw(img_pil)
        text = f"Image {image_counter - 1}"  # Image number starts at 0
        text_position = (10, 10)  # Top-left corner
        draw.text(text_position, text, font=font, fill="white")  # Draw the text with the specified font size

        # Append the image to the list
        images_list.append(img_pil)

        # Break when we have collected enough images (rows * cols)
        if len(images_list) >= rows * cols:
            break

    # Assuming all images are the same size, get width and height of an image
    w, h = images_list[0].size

    # Add padding to width and height for spacing between images
    w_padded = w + padding
    h_padded = h + padding

    # Create a new blank image that fits all images in the grid, including padding
    grid_img = Image.new('RGB', (cols * w_padded, rows * h_padded), color=(255, 255, 255))  # White background

    # Paste each image into the grid with padding
    for i, img in enumerate(images_list):
        x = (i % cols) * w_padded
        y = (i // cols) * h_padded
        grid_img.paste(img, (x, y))

    # Save the grid image as a PDF
    grid_img.save(output_file, "PDF", quality=100)

    # Save the mapping of image indices to filenames as a JSON file
    with open(json_output, 'w') as json_file:
        json.dump(image_mapping, json_file, indent=4)

    print(f"Image grid saved as {output_file}")
    print(f"Filename mapping saved as {json_output}")
    
def set_up_data_splits(train_df, val_df, test_df, cv2_annotations_dir, root_dir):

    split_name = 'train'
    print(split_name)
    # DatasetCatalog.remove('giraffe_torso_train')
    DatasetCatalog.register("giraffe_torso_train", lambda split_name=split_name: get_dataset_dicts(train_df, cv2_annotations_dir, root_dir))
    MetadataCatalog.get("giraffe_torso_" + split_name).set(thing_classes=["giraffe_torso"])

    split_name = 'val'
    print(split_name)
    # DatasetCatalog.remove('giraffe_torso_val')
    DatasetCatalog.register("giraffe_torso_val", lambda split_name=split_name: get_dataset_dicts(val_df, cv2_annotations_dir, root_dir))
    MetadataCatalog.get("giraffe_torso_" + split_name).set(thing_classes=["giraffe_torso"])

    split_name = 'test'
    print(split_name)
    # DatasetCatalog.remove('giraffe_torso_test')
    DatasetCatalog.register("giraffe_torso_test", lambda split_name=split_name: get_dataset_dicts(test_df, cv2_annotations_dir, root_dir))
    MetadataCatalog.get("giraffe_torso_" + split_name).set(thing_classes=["giraffe_torso"])
    
    print(MetadataCatalog)
    print(DatasetCatalog)
    return DatasetCatalog, MetadataCatalog
    
def get_cv2_annotations(cv2_annotations_dir, row):
    cv2_annotation_file = os.path.join(cv2_annotations_dir, ''.join(row['path'].replace('/', '___').split('.')[:-1]) + '.txt')
    if os.path.exists(cv2_annotation_file):
        with open(cv2_annotation_file, 'r') as cv2_file:
            cv2_annotation = cv2_file.readline().strip()
        return cv2_annotation.split()
    else:
        return None

def get_dataset_dicts(df_split, cv2_annotations_dir, root_dir):
    dataset_dicts = []

    for index, row in df_split.iterrows():
        
        record = {}

        filename = os.path.join(root_dir, row['path_relative_to_root'])     
        cv2_annotation = get_cv2_annotations(cv2_annotations_dir, row)
        
        if cv2_annotation:

            giraffe_id, x_min, y_min, width_bb, height_bb, width, height = cv2_annotation  
            category = 0 # just want to detect torso
            
            record["file_name"] = filename
            record["image_id"] = int(row['#Serial'])  # Use #Serial as image_id
            record["width"] = int(width)
            record["height"] = int(height)

            obj = {
                "bbox": [
                    int(x_min),
                    int(y_min),
                    int(x_min) + int(width_bb),
                    int(y_min) + int(height_bb),
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(category),
            }
            record["annotations"] = [obj]
            dataset_dicts.append(record)
    
    return dataset_dicts

def insert_subdir_with_suffix(image_path, subdir='output_image_dir', suffix='_'):
    # Get the directory, filename without extension, and extension
    dir_name = os.path.dirname(image_path)
    base_name, ext = os.path.splitext(os.path.basename(image_path))

    # Create the output directory path
    output_dir = os.path.join(dir_name, subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the new file name with suffix before the extension
    new_filename = f"{base_name}{suffix}{ext}"
    
    return os.path.join(output_dir, new_filename)