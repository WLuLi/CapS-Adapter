#!/bin/bash

# Specify the directory containing the .tar files
directory="./data/datasets/birdsnap/images_tar"

# Specify the directory where files will be extracted
extract_to="./data/datasets/birdsnap/birdsnap/images"

# Create the extraction directory if it doesn't exist
mkdir -p "$extract_to"

# Iterate over all .tar files in the directory and extract them to the specified directory
for tar_file in "$directory"/*.tar; do
    echo "Extracting file to: $extract_to"
    tar -xvf "$tar_file" -C "$extract_to"
done

echo "All files have been extracted to the specified directory."
