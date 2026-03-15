"""
Module for consolidating cleaned text data into CSV files.
"""
import os
import csv
import json
from pathlib import Path
from tqdm import tqdm


class DataConsolidator:
    """
    Class for consolidating cleaned text data into CSV files.
    """
    
    def __init__(self):
        """Initialize the DataConsolidator class."""
        pass
    
    def create_oneline_csv(self, file_paths, output_path, text_extractor, cleaner=None):
        """
        Create a CSV file where each row represents one file with cleaned text.
        
        Args:
            file_paths (list): List of file paths to process
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            cleaner (callable, optional): Function to clean the extracted text
            
        Returns:
            str: Path to the created CSV file
        """
        data = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                # Extract text from the file
                text = text_extractor(file_path)
                
                # Clean text if cleaner is provided
                if cleaner and callable(cleaner):
                    text = cleaner(text)
                
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                
                # Create a row for this file
                row = {
                    'filename': metadata['filename'],
                    'text': text.replace('\n', ' ').replace('\r', ''),  # Ensure text is single line
                    'file_path': metadata['path'],
                    'file_size': metadata['size_bytes'],
                    'file_type': metadata['extension']
                }
                
                data.append(row)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save to CSV
        if data:
            # Get all unique keys to use as CSV headers
            headers = set()
            for row in data:
                headers.update(row.keys())
            headers = list(headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                writer.writerows(data)
            return output_path
        else:
            raise ValueError("No data to consolidate")
    
    def extract_metadata(self, file_path):
        """
        Extract metadata from a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Metadata dictionary
        """
        file_info = Path(file_path)
        
        metadata = {
            'filename': file_info.name,
            'extension': file_info.suffix,
            'size_bytes': file_info.stat().st_size,
            'created_time': file_info.stat().st_ctime,
            'modified_time': file_info.stat().st_mtime,
            'path': str(file_info.absolute())
        }
        
        return metadata
    
    def text_to_dict_list(self, text, metadata=None, split_paragraphs=False):
        """
        Convert text to a list of dictionaries.
        
        Args:
            text (str): Text content
            metadata (dict, optional): Metadata to include
            split_paragraphs (bool): Whether to split text into paragraphs
            
        Returns:
            list: List of dictionaries containing the text data
        """
        if split_paragraphs:
            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Create a list of dictionaries for each paragraph
            data = []
            for i, paragraph in enumerate(paragraphs):
                row = {'paragraph_id': i, 'content': paragraph}
                
                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        row[f'meta_{key}'] = value
                
                data.append(row)
            
            return data
        else:
            # Create a single row dictionary
            data = {'content': text}
            
            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    data[f'meta_{key}'] = value
            
            return [data]
    
    def consolidate_files(self, file_paths, output_path, text_extractor, split_paragraphs=False, one_file_per_row=True):
        """
        Consolidate multiple text files into a single CSV file.
        
        Args:
            file_paths (list): List of file paths to consolidate
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            split_paragraphs (bool): Whether to split text into paragraphs
            one_file_per_row (bool): Whether to ensure one file per row in output
            
        Returns:
            str: Path to the created CSV file
        """
        all_data = []
        
        for file_path in tqdm(file_paths, desc="Consolidating files"):
            try:
                # Extract text from the file
                text = text_extractor(file_path)
                
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                
                # Convert text to list of dictionaries
                if one_file_per_row:
                    # Ensure one file per row, even if split_paragraphs is True
                    row_data = [{
                        'content': text,
                        **{f'meta_{k}': v for k, v in metadata.items()}
                    }]
                else:
                    # Use the original behavior with potential paragraph splitting
                    row_data = self.text_to_dict_list(text, metadata, split_paragraphs)
                
                # Append to the list
                all_data.extend(row_data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save to CSV
        if all_data:
            # Get all unique keys to use as CSV headers
            headers = set()
            for row in all_data:
                headers.update(row.keys())
            headers = list(headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                writer.writerows(all_data)
            
            return output_path
        else:
            raise ValueError("No data to consolidate")
    
    def consolidate_directory(self, input_dir, output_path, text_extractor, 
                              file_extensions=None, recursive=False, split_paragraphs=False,
                              one_file_per_row=True):
        """
        Consolidate all text files in a directory into a single CSV file.
        
        Args:
            input_dir (str): Directory containing files to consolidate
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            file_extensions (list, optional): List of file extensions to include
            recursive (bool): Whether to search for files recursively
            split_paragraphs (bool): Whether to split text into paragraphs
            
        Returns:
            str: Path to the created CSV file
        """
        # Get all files in the directory
        if recursive:
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) 
                        for f in filenames]
        else:
            all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if os.path.isfile(os.path.join(input_dir, f))]
        
        # Filter by extension if specified
        if file_extensions:
            all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in file_extensions]
        
        return self.consolidate_files(all_files, output_path, text_extractor, split_paragraphs, one_file_per_row)
