import os
import shutil
from typing import List

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

INPUT_ROOT = "HT21"
OUTPUT_ROOT = "YN-HT"


def calculate_center_coordinates(bb_left, bb_top, bb_width, bb_height):
    bb_center_x = bb_left + bb_width / 2
    bb_center_y = bb_top + bb_height / 2

    return bb_center_x, bb_center_y


def normalize_coordinates(bb_left, bb_top, bb_width, bb_height, image_width, image_height):
    bb_center_x, bb_center_y = calculate_center_coordinates(bb_left, bb_top, bb_width, bb_height)

    normalized_bb_center_x = bb_center_x / image_width
    normalized_bb_center_y = bb_center_y / image_height
    normalized_bb_width = bb_width / image_width
    normalized_bb_height = bb_height / image_height

    return normalized_bb_center_x, normalized_bb_center_y, normalized_bb_width, normalized_bb_height


def transform_annotation(input_file, output_dir, image_width, image_height, file_prefix):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read input file and process each line
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line into individual values
            values = line.strip().split(',')

            # Extract relevant information
            frame = values[0].zfill(6)
            class_id = 0
            bb_left = float(values[2])
            bb_top = float(values[3])
            bb_width = float(values[4])
            bb_height = float(values[5])

            # Normalize the bounding box coordinates
            normalized_bb_center_x, normalized_bb_center_y, normalized_bb_width, normalized_bb_height = \
                normalize_coordinates(bb_left, bb_top, bb_width, bb_height, image_width, image_height)

            # Create output file name
            output_file = os.path.join(output_dir, f"{file_prefix}{frame}.txt")

            # Write transformed annotation to the output file
            with open(output_file, 'a') as output:
                output.write(
                    f"{class_id} "
                    f"{normalized_bb_center_x} "
                    f"{normalized_bb_center_y} "
                    f"{normalized_bb_width} "
                    f"{normalized_bb_height}\n"
                )


def append_string_to_filenames_in_directory(directory, prefix, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Extract the file name and extension
            base_name, extension = os.path.splitext(filename)

            # Append the prefix to the file name
            new_name = f"{prefix}{base_name}{extension}"

            # Create the new file path in the output directory
            new_path = os.path.join(output_directory, new_name)

            # Copy the file to the new location with the new name
            shutil.copyfile(file_path, new_path)


def copy_every_nth_file(source_dir, destination_dir, n):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get the list of files in the source directory
    files = os.listdir(source_dir)

    # Filter files to keep only every nth file
    every_nth_file = [files[i] for i in range(0, len(files), n)]

    # Copy every nth file to the destination directory
    for file_name in every_nth_file:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copyfile(source_path, destination_path)


def create_empty_txt(input_directory, output_directory):
    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each file in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.lower().endswith('.jpg'):
            # Construct paths
            input_image_path = os.path.join(input_directory, file_name)
            file_name_without_extension = os.path.splitext(file_name)[0]
            output_txt_path = os.path.join(output_directory, f"{file_name_without_extension}.txt")

            # Create an empty text file
            with open(output_txt_path, 'w') as txt_file:
                pass  # The file is empty


# if __name__ == "__main__":
    ## Create images with no annotations; null
    # input_dir = "extract/nature/frames"
    # output_dir = "extract/nature/frames/labels"
    #
    # create_empty_txt(input_dir, output_dir)

    ## Reduce the size of training and validation datasets
    # source_directories = [
    #     "YN-HT/images/val",
    #     "YN-HT/images/train",
    #     "YN-HT/labels/val",
    #     "YN-HT/labels/train"
    # ]
    # destination_directories = [
    #     "YN-HT-s/images/val",
    #     "YN-HT-s/images/train",
    #     "YN-HT-s/labels/val",
    #     "YN-HT-s/labels/train"
    # ]
    # n_value = 10

    # for num_dir in range(len(source_directories)):
    #     copy_every_nth_file(source_directories[num_dir], destination_directories[num_dir], n_value)

    ## Copy test dataset
    # copy_every_nth_file("YN-HT/images/test", "YN-HT-s/images/test", 1)
    # copy_every_nth_file("YN-HT/labels/test", "YN-HT-s/labels/test", 1)

    ## Convert HT21 annotation to YOLO annotation
    # ht_dirs = {"HT21-01", "HT21-02", "HT21-03", "HT21-04"}
    # ht_dirs = {"HT21-11", "HT21-12", "HT21-13"}
    # what = "train"
    # ht_dirs = {"HT21-14", "HT21-15"}
    # what = "val"

    # for ht_dir in ht_dirs:
    #     parent_directory_img = f"{INPUT_ROOT}/train/{ht_dir}/img1"
    #     output_directory_img = f"{OUTPUT_ROOT}/images/train"
    #     parent_file_det = f"{INPUT_ROOT}/train/{ht_dir}/det/det.txt"
    #
    #     parent_directory_img = f"{INPUT_ROOT}/test/{ht_dir}/img1"
    #     output_directory_img = f"{OUTPUT_ROOT}/images/{what}"
    #
    #     parent_file_det = f"{INPUT_ROOT}/test/{ht_dir}/det/det.txt"
    #     output_directory_det = f"{OUTPUT_ROOT}/labels/{what}"
    #
    #     new_prefix = f"{ht_dir}_"
    #
    #     append_string_to_filenames_in_directory(parent_directory_img, new_prefix, output_directory_img)
    #     transform_annotation(parent_file_det, output_directory_det, IMAGE_WIDTH, IMAGE_HEIGHT, new_prefix)
