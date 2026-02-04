"""
Main module for processing text files: converting, cleaning, and consolidating.
"""
import os
from pathlib import Path
from .converter import TextConverter
from .cleaner import TextCleaner
from .consolidator import DataConsolidator


class TextProcessor:
    """
    Class for processing text files: converting, cleaning, and consolidating.
    """
    
    def __init__(self, language='english'):
        """
        Initialize the TextProcessor class.
        
        Args:
            language (str): Language for NLP processing
        """
        self.converter = TextConverter()
        self.cleaner = TextCleaner(language=language)
        self.consolidator = DataConsolidator()
    
    def get_supported_files(self, input_dir, recursive=False, extensions=None):
        """
        Get all supported files in a directory.
        
        Args:
            input_dir (str): Directory to search in
            recursive (bool): Whether to search recursively
            extensions (list, optional): List of file extensions to include
            
        Returns:
            list: List of file paths
        """
        # Get all files in the directory
        if recursive:
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) 
                        for f in filenames]
        else:
            all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if os.path.isfile(os.path.join(input_dir, f))]
        
        # Filter by extension if specified
        if extensions:
            all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in extensions]
        else:
            # Filter by supported formats
            all_files = [f for f in all_files if self.converter.is_supported(f)]
        
        return all_files
    
    def process_file(self, file_path, output_dir=None, cleaning_operations=None):
        """
        Process a single file: convert to Word, clean, and return cleaned text.
        
        Args:
            file_path (str): Path to the input file
            output_dir (str, optional): Directory to save the Word document
            cleaning_operations (list, optional): List of cleaning operations to apply
            
        Returns:
            tuple: (word_doc_path, cleaned_text)
        """
        # Convert to Word
        if output_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            word_doc_path = os.path.join(output_dir, f"{base_name}.docx")
        else:
            word_doc_path = None
        
        word_doc_path = self.converter.convert_to_word(file_path, word_doc_path)
        
        # Extract text
        text = self.converter.extract_text(file_path)
        
        # Clean text
        cleaned_text = self.cleaner.clean_text(text, operations=cleaning_operations)
        
        return word_doc_path, cleaned_text
    
    def process_directory(self, input_dir, output_dir, csv_output_path, 
                          cleaning_operations=None, recursive=False, split_paragraphs=False):
        """
        Process all files in a directory: convert to Word, clean, and consolidate to CSV.
        
        Args:
            input_dir (str): Directory containing files to process
            output_dir (str): Directory to save the Word documents
            csv_output_path (str): Path to save the consolidated CSV file
            cleaning_operations (list, optional): List of cleaning operations to apply
            recursive (bool): Whether to search for files recursively
            split_paragraphs (bool): Whether to split text into paragraphs
            
        Returns:
            tuple: (list of Word docs, CSV path)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all files in the directory
        if recursive:
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) 
                        for f in filenames]
        else:
            all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if os.path.isfile(os.path.join(input_dir, f))]
        
        # Filter supported files
        supported_files = [f for f in all_files if self.converter.is_supported(f)]
        
        # Process each file
        word_docs = []
        processed_files = []
        
        for file_path in supported_files:
            try:
                # Get relative path to maintain directory structure
                rel_path = os.path.relpath(file_path, input_dir)
                file_output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
                
                # Create subdirectories if needed
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Process the file
                word_doc, _ = self.process_file(file_path, file_output_dir, cleaning_operations)
                word_docs.append(word_doc)
                processed_files.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Define a text extractor function for consolidation
        def text_extractor(file_path):
            text = self.converter.extract_text(file_path)
            return self.cleaner.clean_text(text, operations=cleaning_operations)
        
        # Consolidate processed files to CSV
        csv_path = self.consolidator.consolidate_files(
            processed_files, csv_output_path, text_extractor, split_paragraphs
        )
        
        return word_docs, csv_path
