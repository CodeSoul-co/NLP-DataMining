#!/usr/bin/env python
"""
Main entry point for the DataClean application.
Provides a command-line interface for converting text files to CSV with NLP cleaning.
"""
import os
import sys
import click
from dataclean.src.converter import TextConverter
from dataclean.src.cleaner import TextCleaner
from dataclean.src.consolidator import DataConsolidator


@click.group()
def cli():
    """DataClean: Convert text files to CSV with NLP cleaning."""
    pass


@cli.command('convert')
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--language', '-l', default='english', type=click.Choice(['english', 'chinese']), 
              help='Language for NLP processing')
@click.option('--clean/--no-clean', default=True, help='Apply NLP cleaning to text')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars']),
              help='Cleaning operations to apply')
def convert_command(input_path, output_csv, recursive, language, clean, operations):
    """Convert text files to CSV with optional NLP cleaning."""
    # Initialize components
    converter = TextConverter()
    cleaner = TextCleaner(language=language)
    consolidator = DataConsolidator()
    
    click.echo(f"Processing {input_path}...")
    
    # Get files to process
    if os.path.isdir(input_path):
        # Get all supported files in directory
        files = []
        if recursive:
            for root, _, filenames in os.walk(input_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if converter.is_supported(file_path):
                        files.append(file_path)
        else:
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path) and converter.is_supported(file_path):
                    files.append(file_path)
        
        click.echo(f"Found {len(files)} supported files")
    else:
        # Single file
        files = [input_path]
    
    # Process files
    if clean:
        # With cleaning
        csv_path = consolidator.create_oneline_csv(
            files,
            output_csv,
            converter.extract_text,
            lambda text: cleaner.clean_text(text, operations=operations if operations else None)
        )
    else:
        # Without cleaning
        csv_path = consolidator.create_oneline_csv(
            files,
            output_csv,
            converter.extract_text
        )
    
    click.echo(f"Conversion complete! Output saved to: {csv_path}")


@cli.command('clean')
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--language', '-l', default='english', type=click.Choice(['english', 'chinese']), 
              help='Language for NLP processing')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars']),
              help='Cleaning operations to apply')
def clean_command(input_file, output_file, language, operations):
    """Clean a text file using NLP techniques."""
    # Initialize components
    converter = TextConverter()
    cleaner = TextCleaner(language=language)
    
    click.echo(f"Extracting text from {input_file}...")
    text = converter.extract_text(input_file)
    
    click.echo("Cleaning text...")
    cleaned_text = cleaner.clean_text(text, operations=operations if operations else None)
    
    click.echo(f"Saving cleaned text to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    click.echo("Cleaning complete!")


@cli.command('batch')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--language', '-l', default='english', type=click.Choice(['english', 'chinese']), 
              help='Language for NLP processing')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars']),
              help='Cleaning operations to apply')
def batch_command(input_dir, output_dir, recursive, language, operations):
    """Process multiple files and save individual cleaned text files."""
    # Initialize components
    converter = TextConverter()
    cleaner = TextCleaner(language=language)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all supported files
    files = []
    if recursive:
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if converter.is_supported(file_path):
                    files.append(file_path)
    else:
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path) and converter.is_supported(file_path):
                files.append(file_path)
    
    click.echo(f"Found {len(files)} supported files")
    
    # Process each file
    processed_count = 0
    for file_path in files:
        try:
            # Get relative path to maintain directory structure
            rel_path = os.path.relpath(file_path, input_dir)
            output_path = os.path.join(output_dir, rel_path + '.txt')
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Extract and clean text
            text = converter.extract_text(file_path)
            cleaned_text = cleaner.clean_text(text, operations=operations if operations else None)
            
            # Save cleaned text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            processed_count += 1
            click.echo(f"Processed {rel_path}")
        except Exception as e:
            click.echo(f"Error processing {file_path}: {e}")
    
    click.echo(f"Batch processing complete! Processed {processed_count} files.")


@cli.command('supported-formats')
def supported_formats_command():
    """List all supported file formats."""
    converter = TextConverter()
    formats = converter.supported_formats
    click.echo("Supported file formats:")
    for fmt in formats:
        click.echo(f"  {fmt}")


if __name__ == '__main__':
    cli()
