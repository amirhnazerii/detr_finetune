#!/bin/bash

# Script to download Cityscapes dataset essential files for DETR finetuning
set -e  # Exit on error

# Variables
USERNAME="amir20n"  # Your Cityscapes username
PASSWORD="Cityscapesn@zeri21n"  # Your Cityscapes password
DOWNLOAD_DIR="/scratch/anazeri/cityscapes"  # Where to save the downloaded files
DATASET_DIR="/scratch/anazeri/cityscapes"  # Where to extract the dataset

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$DATASET_DIR"

# Check if credentials are provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Please set your Cityscapes username and password in the script."
    echo "You need to register at https://www.cityscapes-dataset.com/register/"
    exit 1
fi

echo "Downloading Cityscapes dataset files..."

# Function to download with authentication
download_file() {
    local url="$1"
    local output="$2"
    
    echo "Downloading $output..."
    curl -L -c cookies.txt -o "$output" \
         -d "username=$USERNAME&password=$PASSWORD&submit=Login" "$url"
    
    if [ ! -f "$output" ] || [ ! -s "$output" ]; then
        echo "Failed to download $output. Please check your credentials and try again."
        exit 1
    fi
    
    echo "Successfully downloaded $output"
}

# Download essential files
FILES_TO_DOWNLOAD=(
    "leftImg8bit_trainvaltest.zip"
    "gtFine_trainvaltest.zip"
)

DOWNLOAD_LINKS=(
    "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
    "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
)

for i in "${!FILES_TO_DOWNLOAD[@]}"; do
    download_file "${DOWNLOAD_LINKS[$i]}" "$DOWNLOAD_DIR/${FILES_TO_DOWNLOAD[$i]}"
done

echo "Downloads completed. Now extracting files..."

# Extract downloaded files
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    echo "Extracting $file..."
    unzip -q "$DOWNLOAD_DIR/$file" -d "$DATASET_DIR"
done

echo "Extraction completed. Setting up proper directory structure..."

# Verify directory structure
if [ -d "$DATASET_DIR/leftImg8bit" ] && [ -d "$DATASET_DIR/gtFine" ]; then
    echo "Dataset structure looks correct!"
else
    echo "WARNING: Expected directories not found. Please check the extraction."
fi

# Show summary
echo "============================================================"
echo "Cityscapes dataset has been downloaded and prepared."
echo "Dataset location: $DATASET_DIR"
echo "Directory structure:"
echo "  $DATASET_DIR/leftImg8bit - Contains images"
echo "  $DATASET_DIR/gtFine - Contains annotations"
echo "============================================================"
echo "You should now be able to run the DETR finetuning with:"
echo "--dataset_file \"cityscapes\" --coco_path \"$DATASET_DIR\""
echo "============================================================"

# Cleanup
rm -f cookies.txt

exit 0