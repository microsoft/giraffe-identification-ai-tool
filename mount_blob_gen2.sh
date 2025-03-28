# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

#!/bin/bash  

# You can call this script by doing: sudo ./mountblob.sh <container_name> <storage_account_name> <mount_type> <app_id>

container_name=$1
storage_account_name=$2
mount_type=$3
app_id=$4

#/mnt/blobfusetmp/blobfuse2.yaml 
blobfuse_temp_path="/home/blobfusetmp"
blobfuse_cache_path="/mnt/blobfusecache"

# check if container_name is set, if not exit with a message
if [ -z "$container_name" ]; then
    echo "container_name is required"
    exit 1
fi

# check if storage_account_name is set, if not exit with a message
if [ -z "$storage_account_name" ]; then
    echo "storage_account_name is required"
    exit 1
fi

# check if mount_type is set, if not set to default
if [ -z "$mount_type" ]; then
    mount_type="block"
    echo "mount_type is not set, defaulting to block"
fi

# show the variables and ask the user to confirm
echo "container_name: $container_name"
echo "storage_account_name: $storage_account_name"
echo "mount_type: $mount_type"
echo "app_id: $app_id"

# create the blobfuse2 temp path
sudo mkdir -p $blobfuse_temp_path

# create the mount path, based on the passed in container_name variable
sudo mkdir -p /mnt/$1  

# remove the file if it exists
# sudo rm -f "$blobfuse_temp_path/blobfuse2.yaml"
if [ -f "$blobfuse_temp_path/blobfuse2.yaml" ]; then
    echo "Removing existing blobfuse2.yaml at $blobfuse_temp_path"
    sudo rm -f "$blobfuse_temp_path/blobfuse2.yaml"
fi

# create file cache directory
sudo mkdir -p $blobfuse_cache_path

# create the yaml file for the configuration
cat <<EOF > "$blobfuse_temp_path/blobfuse2.yaml"
# Disk cache related configuration
file_cache:
    # Required
    path: $blobfuse_cache_path
    timeout-sec: 0 

# Azure storage configuration
azstorage:
    # Required
    type: "$mount_type"
    account-name: "$storage_account_name"
    container: "$container_name"
    mode: "msi"
    appid: "$app_id"
EOF

echo "Configuration file available at $blobfuse_temp_path/blobfuse2.yaml"

echo "Mounting the blob container..."
# mount the blob container
sudo blobfuse2 /mnt/$1 --tmp-path=$blobfuse_cache_path --config-file="$blobfuse_temp_path/blobfuse2.yaml" -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other -o direct_io

if [ "$(ls -A /mnt/$1 2>/dev/null)" ]; then
    echo "Blob container mounted successfully at /mnt/$1"
else
    echo "Failed to mount: /mnt/$1 is empty"
fi
 
