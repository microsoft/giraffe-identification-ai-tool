# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import cv2
import sys
import time
import math
import torch
import faiss
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import faiss_index_dir
from utils.helpers_matching import load_pkl_files

def train_faiss(all_descriptors_train):
    print('\nTraining faiss index started ...')
    start_time = time.time()
    
    # Normalize the descriptors by dividing by their L2 norms
    all_descriptors_train_normalized = normalize(all_descriptors_train)
    
    # Create faiss index (HNSWFlat in this case)
    faiss_index = faiss.IndexHNSWFlat(all_descriptors_train.shape[1], 16)
    
    # Add normalized descriptors to the faiss index
    faiss_index.add(all_descriptors_train_normalized)
    
    print('\nTraining time for faiss {:.6f} seconds'.format(time.time() - start_time))
    
    return faiss_index

def train_faiss_partial_data(root_dir, data_percentage_used):
    
    # Validate data_percentage_used
    if not (0 < data_percentage_used <= 1):
        raise ValueError(f"data_percentage_used for train_faiss_partial_data should be between 0 and 1, but got {data_percentage_used}")

    # Load pkl files for descriptors
    descriptors_data = load_pkl_files(root_dir)
    
    # Reshape reference data for faiss
    all_descriptors_train, all_labels_train, all_serials_train = reshape_reference_data_for_faiss(descriptors_data['reference'])

    # Check if data is available
    if len(all_serials_train) == 0:
        print("Warning: No training data available in all_serials_train.")
        return None, None, None, None

    # Compute partial data index
    partial_data_idx = int(data_percentage_used * len(all_serials_train))

    # Ensure partial data is not empty
    if partial_data_idx == 0:
        print("Warning: Selected data percentage is too small, leading to no training data.")
        return None, None, None, None

    print(f"Using {partial_data_idx} out of {len(all_serials_train)} samples for training.")

    # Slice the training data
    all_descriptors_train_init = all_descriptors_train[:partial_data_idx]
    all_labels_train_init = all_labels_train[:partial_data_idx]
    all_serials_train_init = all_serials_train[:partial_data_idx]

    print(f"all_descriptors_train_init: {all_descriptors_train_init.shape}, {type(all_descriptors_train_init)}")
    print(f"all_labels_train_init: {all_labels_train_init.shape}, {type(all_labels_train_init)}")
    print(f"all_serials_train_init: {len(all_serials_train_init)}, {type(all_serials_train_init)}")

    # Train faiss index
    faiss_index = train_faiss(all_descriptors_train_init)
    
    print(f"Number of vectors in the index after adding new items: {faiss_index.ntotal}\n")

    return faiss_index, all_descriptors_train_init, all_labels_train_init, all_serials_train_init

def write_faiss(faiss_index, all_descriptors_train, all_labels_train, all_serials_train, faiss_index_dir, subdir=None, activate=False):
    if subdir is not None:
        faiss_index_dir = os.path.join(faiss_index_dir, subdir)
    os.makedirs(faiss_index_dir, exist_ok=True)
    print('\nWriting faiss index started ...')
    print('\nPath to write faiss index: ', faiss_index_dir)
    start_time = time.time()
    faiss.write_index(faiss_index, os.path.join(faiss_index_dir, 'faiss_index.index'))
    print('\nWriting time for faiss {:.6f} seconds'.format(time.time() - start_time))
    
    print('\nWriting supplemental data for faiss index started ...')
    start_time = time.time()
    with open(os.path.join(faiss_index_dir,'all_descriptors_train.pkl'), 'wb') as file:
        pickle.dump(all_descriptors_train, file)
    with open(os.path.join(faiss_index_dir, 'all_labels_train.pkl'), 'wb') as file:
        pickle.dump(all_labels_train, file)
    with open(os.path.join(faiss_index_dir, 'all_serials_train.pkl'), 'wb') as file:
        pickle.dump(all_serials_train, file)
    print('\nWriting time for supplemental data for faiss index {:.6f} seconds'.format(time.time() - start_time))

def read_faiss():
    print('\nReading faiss index started ...')
    print('\nPath to read faiss index: ', faiss_index_dir)
    start_time = time.time()
    faiss_index = faiss.read_index(os.path.join(faiss_index_dir, 'faiss_index.index'))
    print('\nReading time for faiss {:.6f} seconds'.format(time.time() - start_time))
    
    print('\nReading supplemental data for faiss index started ...')
    start_time = time.time()
    with open(os.path.join(faiss_index_dir,'all_descriptors_train.pkl'), 'rb') as file:
        all_descriptors_train = pickle.load(file)
    with open(os.path.join(faiss_index_dir, 'all_labels_train.pkl'), 'rb') as file:
        all_labels_train = pickle.load(file)
    with open(os.path.join(faiss_index_dir, 'all_serials_train.pkl'), 'rb') as file:
        all_serials_train = pickle.load(file)
    print('\nReading time for supplemental data for faiss index {:.6f} seconds'.format(time.time() - start_time))
    
    return faiss_index, all_descriptors_train, all_labels_train, all_serials_train

def load_trained_faiss_ref(faiss_index_dir):
    filenames = [
            'all_descriptors_train.pkl',
            'all_labels_train.pkl',
            'all_serials_train.pkl',
            'faiss_index.index']

    # Check if all files exist in the specified directory
    if all(os.path.isfile(os.path.join(faiss_index_dir, filename)) for filename in filenames):
        faiss_index_ref, all_descriptors_train_ref, all_labels_train_ref, all_serials_train_ref = read_faiss()
        return faiss_index_ref, all_descriptors_train_ref, all_labels_train_ref, all_serials_train_ref
    
    # If at least one file is missing return
    else:
        print('index not available to load.')
        sys.exit()
        
def serialize_a_reference_image_data(label_id, serials_per_image, descriptors_per_image, all_descriptors, all_labels, all_serials):
    for i, descriptors in enumerate(descriptors_per_image):
        serial = serials_per_image[i]
        if descriptors is not None and len(descriptors.shape) == 2:
            num_descriptors = descriptors.shape[0]
            all_descriptors.append(descriptors)
            all_labels.extend([label_id] * num_descriptors)
            all_serials.extend([serial] * num_descriptors)
    return all_descriptors, all_labels, all_serials

def reshape_reference_data_for_faiss(ref_descriptors_data, keys=None):
    if keys is None:    
        keys = ref_descriptors_data.keys()

    print(f'\nReshaping reference data for faiss...')
    all_descriptors = []
    all_labels = []
    all_serials = []
    
    for label_id in tqdm(keys):
        if label_id in ref_descriptors_data.keys():
            descriptors_per_image, serials_per_image = ref_descriptors_data[label_id]
            all_descriptors, all_labels, all_serials = serialize_a_reference_image_data(label_id, serials_per_image, descriptors_per_image, all_descriptors, all_labels, all_serials)
        else:
            print(f"Warning: label_id {label_id} not found in ref_descriptors_data.")
    
    if all_descriptors:
        all_descriptors = np.concatenate(all_descriptors, axis=0)
        all_labels = np.array(all_labels)
        all_serials = np.array(all_serials)
    print(f'\nReference data reshaped for faiss:{len(all_descriptors)}, {len(all_labels)}, {len(all_serials)}')
    return all_descriptors, all_labels, all_serials

def serialize_and_reshape_a_query_image_data(label_id, serial, descriptors):
    all_descriptors, all_labels, all_serials = [], [], []
    if descriptors is not None and len(descriptors.shape) == 2:
        num_descriptors = descriptors.shape[0]
        all_descriptors.append(descriptors)
        all_labels.extend([label_id] * num_descriptors)
        all_serials.extend([serial] * num_descriptors)
    if all_descriptors:
        all_descriptors = np.concatenate(all_descriptors, axis=0)
        all_labels = np.array(all_labels)
        all_serials = np.array(all_serials)
    return all_descriptors, all_labels, all_serials

class Giraffe_seg_and_torso_seg_process:
    
    def __init__(self, image, instances, torso_instances):
        self.image = image
        
        # Reference for finding torso based on giraffe selected
        self.reference_torso_box = None
        
        # Full giraffe segment
        self.segm_results = instances
        self.image_masked = None
        self.instance_box = None
        self.image_cropped_sq = None
        
        # Giraffe torso object
        self.giraffe_torso_results = torso_instances
        self.image_center_torso_masked = None
        self.instance_center_torso_box = None
        self.image_center_torso_cropped_sq = None
        self.image_center_torso_cropped_sq_original = None
        
        # Get some stats
        self.giraffe_segment_counts = 0
        self.torso_detection_coverage = 0
        self.torso_segmentation_coverage = 0
        self.torso_combined_coverage = 0
        
        self.segmentation_mask = None
        self.detection_mask = None
        
    def select_giraffe_segment(self):
        animal_class = 23
        image_height, image_width, ch = self.image.shape
        min_eDistance = math.dist([0.5*image_width, 0.5*image_height], [0,0])
        min_eDistance_arg = -1
        min_eDistance_box = None
        mask = None
        image_masked = None
        for arg, istance_class in enumerate(self.segm_results['instances'].pred_classes):
            if istance_class == animal_class:
                self.giraffe_segment_counts += 1
                box = self.segm_results['instances'].pred_boxes[arg].tensor.cpu().numpy()[0]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if self.reference_torso_box is not None:
                    image_width_ref = self.reference_torso_box[2] + self.reference_torso_box[0]
                    image_height_ref = self.reference_torso_box[3] + self.reference_torso_box[1]
                else:
                    image_width_ref = image_width
                    image_height_ref = image_height
                
                eDistance = math.dist([0.5*(x1+x2), 0.5*(y1+y2)], [0.5*image_width_ref, 0.5*image_height_ref])
                
                if eDistance < min_eDistance:
                    min_eDistance = eDistance
                    min_eDistance_arg = arg
                    min_eDistance_box = box
                    mask = self.segm_results['instances'].pred_masks[min_eDistance_arg]
        
        if mask is not None and mask.any():
            
            # Mask the full giraffe based on segmentation
            mask = mask.cpu().numpy()
            self.segmentation_mask = mask
            reshaped_mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            image_masked = self.image * reshaped_mask
            self.instance_box, self.image_masked = min_eDistance_box, image_masked
            
            # Compute segmentation coverage
            covered_area = np.sum(mask)
            total_area = mask.size
            self.torso_segmentation_coverage = covered_area / total_area
        
        if self.reference_torso_box is not None and self.image_masked is not None and self.image_masked.any():
            
            # Find overlap between detection and segmentation results
            x1, y1, x2, y2 = int(self.reference_torso_box[0]), int(self.reference_torso_box[1]), int(self.reference_torso_box[2]), int(self.reference_torso_box[3])
            binary_mask = np.zeros_like(self.image, dtype=np.uint8)
            binary_mask[y1:y2, x1:x2] = 1
            final_result = self.image_masked * binary_mask
            self.instance_center_torso_box, self.image_center_torso_masked = self.reference_torso_box, final_result
            
            # Compute detection coverage
            binary_mask_single_channel = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            binary_mask_single_channel[y1:y2, x1:x2] = 1
            self.detection_mask = binary_mask_single_channel
            covered_area = np.sum(binary_mask_single_channel)
            total_area = binary_mask_single_channel.size
            self.torso_detection_coverage = covered_area / total_area
                        
    def select_giraffe_torso_segment(self):
        image_height, image_width, ch = self.image.shape
        min_eDistance = math.dist([0.5*image_width, 0.5*image_height], [0,0])
        min_eDistance_arg = -1
        min_eDistance_box = None
        mask = None
        image_masked = None
        for arg, istance_class in enumerate(self.giraffe_torso_results['instances'].pred_boxes.tensor.cpu().numpy()):
            
            box = self.giraffe_torso_results['instances'].pred_boxes[arg].tensor.cpu().numpy()[0]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            image_width_ref = image_width
            image_height_ref = image_height
            
            eDistance = math.dist([0.5*(x1+x2), 0.5*(y1+y2)], [0.5*image_width_ref, 0.5*image_height_ref])
            if eDistance < min_eDistance:
                min_eDistance = eDistance
                min_eDistance_arg = arg
                min_eDistance_box = box
            self.reference_torso_box = min_eDistance_box
        

    def _find_square_bbx(self, x1, y1, x2, y2):
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        width = x2 - x1
        height = y2 - y1
        size = max(width, height)
        size = int(size / 2)

        new_x1 = center_x - size
        new_y1 = center_y - size
        new_x2 = center_x + size
        new_y2 = center_y + size
        return new_x1, new_y1, new_x2, new_y2
    
    def rescale_giraffe_segment(self, new_height, new_width):
        if self.image_masked is not None and self.image_masked.any():
            image_height, image_width, ch = self.image.shape
            x1, y1, x2, y2 = self._find_square_bbx(self.instance_box[0],self.instance_box[1],self.instance_box[2],self.instance_box[3])
            self.image_cropped_sq = cv2.resize(self.image_masked[max(y1,0):min(y2+1,image_height), max(x1,0):min(x2+1,image_width)], (new_height, new_width))
        
    def rescale_giraffe_torso_segment(self, new_height, new_width):
        if self.image_center_torso_masked is not None and self.image_center_torso_masked.any():
            image_height, image_width, ch = self.image.shape
            x1, y1, x2, y2 = self._find_square_bbx(self.instance_center_torso_box[0],self.instance_center_torso_box[1],self.instance_center_torso_box[2],self.instance_center_torso_box[3])
            self.image_center_torso_cropped_sq = cv2.resize(self.image_center_torso_masked[max(y1,0):min(y2+1,image_height), max(x1,0):min(x2+1,image_width)], (new_height, new_width))
            self.image_center_torso_cropped_sq_original = self.image_center_torso_masked[
                        max(y1, 0) : min(y2 + 1, image_height),
                        max(x1, 0) : min(x2 + 1, image_width),
                    ]
        
    def get_torso_combined_coverage(self):
        # compute combined coverage
        if self.segmentation_mask is not None and self.detection_mask is not None:
            covered_area = np.sum(self.segmentation_mask * self.detection_mask)
            total_area = self.segmentation_mask.size
            self.torso_combined_coverage = covered_area / total_area
        return self.torso_combined_coverage

    def plot_img(self):
        if self.image is not None and self.image.any():
            plt.imshow(self.image)
            plt.show()
        if self.image_masked is not None and self.image_masked.any():
            plt.imshow(self.image_masked)
            plt.show()
        if self.image_center_torso_masked is not None and self.image_center_torso_masked.any():
            plt.imshow(self.image_center_torso_masked)
            plt.show()
        if self.image_cropped_sq is not None and self.image_cropped_sq.any():
            plt.imshow(self.image_cropped_sq)
            plt.show()
        if self.image_center_torso_cropped_sq is not None and self.image_center_torso_cropped_sq.any():
            plt.imshow(self.image_center_torso_cropped_sq)
            plt.show()

class ProcessGiraffe(torch.utils.data.Dataset):
    
    def __init__(self, a_giraffe_photo_path, a_serial_no, a_label, segmenation_model, torso_model, cropped_img_size, target_dir, output_image_dir, plot_key=False):
        self.image_path = a_giraffe_photo_path
        self.output_image_dir = output_image_dir
        self.a_label = a_label
        self.a_serial_no = a_serial_no
        self.segmenation_model = segmenation_model
        self.torso_model = torso_model
        self.target_dir = target_dir
        self.plot_key = plot_key
        self.cropped_img_size = cropped_img_size
    
    def __getitem__(self, idx=None):
        image_rgb = np.array(ImageOps.exif_transpose(Image.open(os.path.join(self.target_dir,self.image_path))).convert("RGB"))
        image = image_rgb[:, :, ::-1]
       
        segmenation_outputs = self.segmenation_model(image)
        torso_outputs = self.torso_model(image)
        
        giraffe_img_obj = Giraffe_seg_and_torso_seg_process(image, segmenation_outputs, torso_outputs)
        
        giraffe_img_obj.select_giraffe_torso_segment()
        giraffe_img_obj.select_giraffe_segment()      
        giraffe_img_obj.rescale_giraffe_torso_segment(self.cropped_img_size, self.cropped_img_size)
        giraffe_img_obj.rescale_giraffe_segment(self.cropped_img_size, self.cropped_img_size)
        
        save_cropped_torso_image(giraffe_img_obj, self.image_path, self.output_image_dir)

        if self.plot_key:
            giraffe_img_obj.plot_img()

        label = self.a_label
        universal_id = self.a_serial_no    
    
        return giraffe_img_obj, label, universal_id

def save_cropped_torso_image(giraffe_img_obj, image_dir, output_image_dir):
    
    # Save the zoomed in version of torso
    cropped_image_data = giraffe_img_obj.image_center_torso_cropped_sq_original
    if cropped_image_data is not None and cropped_image_data.any():
        
        pil_image = Image.fromarray(np.uint8(cropped_image_data[:, :, ::-1]))
        
        # Make a new name
        parts = image_dir.rsplit(".", 1)
        img_filename = os.path.basename(image_dir)
        cropped_image_dir = os.path.join(output_image_dir, "zoomed_version", img_filename).replace("." + parts[1], "_cropped_torso_zoomed." + parts[1])
        
        # Check if the directory exists
        if not os.path.exists(os.path.dirname(cropped_image_dir)):
            os.makedirs(os.path.dirname(cropped_image_dir))
        
        pil_image.save(cropped_image_dir)
        
    
    # Save the original size version of torso
    cropped_image_data = giraffe_img_obj.image_center_torso_masked
    
    if cropped_image_data is not None and cropped_image_data.any():
        
        pil_image = Image.fromarray(np.uint8(cropped_image_data[:, :, ::-1]))
        
        # make a new name
        parts = image_dir.rsplit(".", 1)
        img_filename = os.path.basename(image_dir)
        cropped_image_dir = os.path.join(output_image_dir, "original_size", img_filename).replace("." + parts[1], "_cropped_torso." + parts[1])
        
        # Check if the directory exists
        if not os.path.exists(os.path.dirname(cropped_image_dir)):
            os.makedirs(os.path.dirname(cropped_image_dir))
        
        pil_image.save(cropped_image_dir)

def normalize(all_descriptors_train):
    # Compute the L2 norms (magnitudes) of the training descriptors
    norms = np.linalg.norm(all_descriptors_train, axis=1, keepdims=True)
    
    # Normalize the descriptors by dividing by their L2 norms
    all_descriptors_train_normalized = all_descriptors_train / norms

    return all_descriptors_train_normalized.astype(np.float32)

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by rank
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
                     
def replace_negatives_with_unique_values(array, target_value=-1):
    
    some_large_number = 10000000
    
    # Step 1: Identify indices of the target value
    target_indices = np.where(array == target_value)[0]
    
    # Step 2: Collect existing values in the array, excluding the target value
    existing_values = set(array[array != target_value])
    
    # Step 3: Generate unique replacement values
    replacement_values = set(
        range(max(existing_values, default=0) + some_large_number, 
              max(existing_values, default=0) + some_large_number + len(target_indices))
    )
    
    # Ensure replacements are unique and not in the array
    unique_replacements = iter(replacement_values)
    
    # Step 4: Create a copy of the array to modify
    new_array = array.copy()
    for index in target_indices:
        new_array[index] = next(unique_replacements)
    
    return new_array

def run_union_find(col1, col2):
    uf = UnionFind()

    # Add each unique value from col1 and col2 to the Union-Find structure
    for val in np.concatenate([col1, col2]):
        uf.add(val)

    # Perform union operations based on the two columns
    for v1, v2 in zip(col1, col2):
        uf.union(v1, v2)

    # Create a mapping from each element to its root (representing its new ID)
    value_to_new_id = {}
    for val in np.concatenate([col1, col2]):
        root = uf.find(val)
        if root not in value_to_new_id:
            value_to_new_id[root] = len(value_to_new_id)  # Assign a new ID to the root
        value_to_new_id[val] = value_to_new_id[root]  # Assign the same ID to all members of the group

    return value_to_new_id