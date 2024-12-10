# This code generates the mask and bounding box extraction from WSI, given the annotations in XML format from Aperio Imagescope.

import cv2
import numpy as np
import os
from xml.dom import minidom
import matplotlib.path as mplPath
import openslide
import time
import argparse
import glob

# Supported file formats
openslide_formats = ['.ndpi', '.svs', '.tif', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', 'svslide', 'bif']

# Argument parser for input case
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--case', help='cases')
results = parser.parse_args()
cases = results.case

def get_all_regions(file_name):
    """Extract all regions and their attributes from the XML file."""
    mydoc = minidom.parse(file_name)
    annotations = mydoc.getElementsByTagName('Annotation')
    all_anns, all_orgs, all_paths, names = [], [], [], []
    
    for annotation in annotations:
        regions = annotation.getElementsByTagName('Region')
        names.append(annotation.getAttribute('Name').encode('utf-8'))
        all_regions, orgs, paths = [], [], []
        
        for region in regions:
            verticies = region.getElementsByTagName('Vertex')
            xy, xy_path = [], []
            
            for item in verticies:
                xy.append(list(map(int, [float(item.getAttribute('X').encode('utf-8')), float(item.getAttribute('Y').encode('utf-8'))])))
                xy_path.append([item.getAttribute('X'), item.getAttribute('Y')])
                
            all_regions.append(xy)
            ox, oy, wd, ht = cv2.boundingRect(np.asarray(xy))
            orgs.append([ox, oy, wd, ht])
            paths.append(mplPath.Path(xy_path))
            
        all_anns.append(all_regions)
        all_orgs.append(orgs)
        all_paths.append(paths)
        
    return all_anns, all_orgs, all_paths, names

def get_annotations_1(format_f, xml_file_path, output_directory, padd, image_path, level=0):
    """Process the XML file and generate masks and bounding boxes."""
    ffl = len(format_f)
    directory = output_directory
    
    # Get all regions from the XML file
    all_anns, all_origins, all_paths, names = get_all_regions(xml_file_path)
    
    # Open the image using OpenSlide or OpenCV
    if str(format_f) in openslide_formats:
        img = openslide.open_slide(image_path)
    else:
        img = cv2.imread(image_path)

    count = 0
    pathsplt = image_path.split('/')
    
    # Iterate through all the annotations
    for lyr in range(len(all_anns)):
        all_regions = all_anns[lyr]
        
        # Create directory for the current layer if it doesn't exist
        if not os.path.exists(output_directory + str(lyr)):
            os.makedirs(output_directory + str(lyr))
        
        if len(all_regions) == 0:
            continue
            
        dir_path = directory + str(lyr) + '/'
        origins = all_origins[lyr]
        paths = all_paths[lyr]
        n_regions = len(all_regions)
        
        # Process each region
        for region_id in range(n_regions):
            if len(origins[region_id]) == 0:
                continue
                
            ox, oy, w, h = origins[region_id]
            if w >= 40000 or h >= 40000:
                continue
            
            region = all_regions[region_id]
            start_x = max(0, ox - padd)
            start_y = max(0, oy - padd)
            shifted_region = [[item[0] - start_x, item[1] - start_y] for item in region]
            
            # Create mask image
            mask_im = np.zeros((h + 2 * padd, w + 2 * padd, 3))
            mask_im = cv2.drawContours(mask_im, np.asarray([shifted_region]), -1, (255, 255, 255), -1, 8)
            mask_im = np.array(mask_im, dtype=np.uint8)
            mask_im = cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY)
            
            img_save_name = f"{dir_path}ROI__{count}__{pathsplt[-1][0:-ffl]}__layer__{lyr}__{ox}__{oy}__{w}__{h}"
            
            if format_f in openslide_formats:
                im_object = img.read_region((start_x, start_y), level, (w + 2 * padd, h + 2 * padd))
                im_object.save(img_save_name + '__.png')
                w1t, h1t = im_object.size
            else:
                im_patch = img[start_y:start_y + h + 2 * padd, start_x:start_x + w + 2 * padd, :]
                cv2.imwrite(img_save_name + '__.png', im_patch)
                h1t, w1t, _ = im_patch.shape
                
            mask_im = mask_im[0:h1t, 0:w1t]
            cv2.imwrite(img_save_name + '__mask__.png', mask_im)
            count += 1
            
    return count

# Placeholders for paths
list_xml = glob.glob("../data/svs_and_xml/*.xml")
for item_xml in list_xml:
    print(item_xml)
    image_path = item_xml.replace(".xml", ".svs")
    print(image_path)
    c = get_annotations_1('.svs', item_xml, '/path/to/output_directory/', 0, image_path, level=0)
    print("Number of ROIs: ", c)