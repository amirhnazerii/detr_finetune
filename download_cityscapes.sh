#!/bin/bash

# Script to download Cityscapes dataset essential files for DETR finetuning
set -e  # Exit on error

# Variables
USERNAME="amir20n"  # Your Cityscapes username
PASSWORD="Cityscapesn@zeri21n"  # Your Cityscapes password
DOWNLOAD_DIR="/scratch/anazeri/cityscapes"  # Where to save the downloaded files
DATASET_DIR="/scratch/anazeri/cityscapes"  # Where to extract the dataset

DOWNLOAD_DIR="/scratch/anazeri/cityscapes/downloads"  # Where to save the downloaded files
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

# First get the login cookie
echo "Authenticating with Cityscapes website..."
wget --keep-session-cookies --save-cookies=cityscapes_cookies.txt \
     --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" \
     -O login_response.html \
     "https://www.cityscapes-dataset.com/login/"

# Check if login was successful
if grep -q "Login failed" login_response.html; then
    echo "Login failed. Please check your credentials."
    rm -f login_response.html cityscapes_cookies.txt
    exit 1
fi

echo "Authentication successful."

# Download files using the session cookie
download_file() {
    local package_id="$1"
    local output_file="$2"
    
    echo "Downloading $output_file..."
    wget --load-cookies=cityscapes_cookies.txt \
         --content-disposition \
         -O "$output_file" \
         "https://www.cityscapes-dataset.com/file-handling/?packageID=$package_id"
         
    # Verify download
    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        echo "Failed to download $output_file"
        exit 1
    fi
    
    # Check if it's a valid ZIP file
    if ! unzip -t "$output_file" &> /dev/null; then
        echo "Downloaded file $output_file is not a valid ZIP file."
        echo "Saving server response for debugging..."
        mv "$output_file" "${output_file}.response"
        cat "${output_file}.response"
        exit 1
    fi
    
    echo "Successfully downloaded $output_file"
}

# Download essential files for DETR - packageIDs from the Cityscapes website
echo "Downloading leftImg8bit_trainvaltest.zip (images)..."
download_file "3" "$DOWNLOAD_DIR/leftImg8bit_trainvaltest.zip"

echo "Downloading gtFine_trainvaltest.zip (fine annotations)..."
download_file "1" "$DOWNLOAD_DIR/gtFine_trainvaltest.zip"

echo "Downloads completed. Now extracting files..."

# Extract downloaded files
for file in "$DOWNLOAD_DIR"/*.zip; do
    echo "Extracting $file..."
    unzip -o "$file" -d "$DATASET_DIR"
done

echo "Extraction completed. Verifying directory structure..."

# Verify directory structure
if [ -d "$DATASET_DIR/leftImg8bit/train" ] && 
   [ -d "$DATASET_DIR/leftImg8bit/val" ] && 
   [ -d "$DATASET_DIR/gtFine/train" ] && 
   [ -d "$DATASET_DIR/gtFine/val" ]; then
    echo "✓ Dataset structure is correct!"
    
    # Count files for verification
    train_images=$(find "$DATASET_DIR/leftImg8bit/train" -name "*_leftImg8bit.png" | wc -l)
    val_images=$(find "$DATASET_DIR/leftImg8bit/val" -name "*_leftImg8bit.png" | wc -l)
    train_annotations=$(find "$DATASET_DIR/gtFine/train" -name "*_gtFine_polygons.json" | wc -l)
    val_annotations=$(find "$DATASET_DIR/gtFine/val" -name "*_gtFine_polygons.json" | wc -l)
    
    echo "✓ Training images: $train_images"
    echo "✓ Validation images: $val_images"
    echo "✓ Training annotations: $train_annotations"
    echo "✓ Validation annotations: $val_annotations"
else
    echo "ERROR: Expected directories not found. The extraction may have failed."
    echo "Please check the download log for errors."
    exit 1
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
rm -f cityscapes_cookies.txt login_response.html

exit 0