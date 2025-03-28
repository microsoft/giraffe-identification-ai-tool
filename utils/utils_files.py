# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

def read_file(file_path: str) -> str:
    """
    Load content from a file into a string.
    
    Args:
        file_path (str): Path to the file to be loaded
        
    Returns:
        str: Content of the file as a string
    """
    with open(file_path, 'r') as file:
        return file.read()