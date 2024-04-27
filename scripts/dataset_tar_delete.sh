#!/bin/bash

# Specify the directory containing the .tar files
directory="./data/datasets/birdsnap/images_tar"

# Iterate over all .tar files in the directory and delete them
for tar_file in "$directory"/*.tar; do
    echo "Deleting file: $tar_file"
    rm "$tar_file"
done

echo "All .tar files have been deleted from the specified directory."
