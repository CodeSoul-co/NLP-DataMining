"""
Command-line interface for the text processing application.
"""
import os
import click
from pathlib import Path
from .processor import TextProcessor


@click.group()
def cli():
    """Text processing application for converting, cleaning, and consolidating text data."""
    pass


@cli.command('convert')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for Word documents')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
def convert_command(input_path, output_dir, recursive):
    """Convert text files to Word format."""
    processor = TextProcessor()
    
    if os.path.isdir(input_path):
        if not output_dir:
            output_dir = input_path
        
        word_docs = processor.converter.batch_convert(input_path, output_dir, recursive)
        click.echo(f"Converted {len(word_docs)} files to Word format.")
    else:
        if not output_dir:
            output_dir = os.path.dirname(input_path)
        
        word_doc = processor.converter.convert_to_word(input_path, 
                                                      os.path.join(output_dir, 
                                                                  os.path.basename(input_path) + '.docx'))
        click.echo(f"Converted {input_path} to Word format: {word_doc}")


@cli.command('clean')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-path', '-o', type=click.Path(), help='Output file path')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars', 'stem_text', 'lemmatize_text']),
              help='Cleaning operations to apply')
def clean_command(input_path, output_path, operations):
    """Clean text data using NLP techniques."""
    processor = TextProcessor()
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean text
    cleaned_text = processor.cleaner.clean_text(text, operations=operations if operations else None)
    
    # Write output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        click.echo(f"Cleaned text saved to {output_path}")
    else:
        click.echo(cleaned_text)


@cli.command('consolidate')
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--split-paragraphs', '-s', is_flag=True, help='Split text into paragraphs')
@click.option('--extensions', '-e', multiple=True, help='File extensions to include')
@click.option('--one-file-per-row', '-o', is_flag=True, default=True, help='Ensure one file per row in output')
def consolidate_command(input_path, output_csv, recursive, split_paragraphs, extensions, one_file_per_row):
    """Consolidate text files into a CSV file."""
    processor = TextProcessor()
    
    if os.path.isdir(input_path):
        # Process directory
        csv_path = processor.consolidator.consolidate_directory(
            input_path, output_csv, processor.converter.extract_text,
            file_extensions=extensions if extensions else None,
            recursive=recursive, split_paragraphs=split_paragraphs,
            one_file_per_row=one_file_per_row
        )
    else:
        # Process single file
        csv_path = processor.consolidator.consolidate_files(
            [input_path], output_csv, processor.converter.extract_text,
            split_paragraphs=split_paragraphs,
            one_file_per_row=one_file_per_row
        )
    
    click.echo(f"Consolidated data saved to {csv_path}")


@cli.command('process')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for Word documents')
@click.option('--csv-output', '-c', type=click.Path(), help='Output path for consolidated CSV')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--split-paragraphs', '-s', is_flag=True, help='Split text into paragraphs')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars', 'stem_text', 'lemmatize_text']),
              help='Cleaning operations to apply')
def process_command(input_path, output_dir, csv_output, recursive, split_paragraphs, operations):
    """Process text files: convert to Word, clean, and consolidate to CSV."""
    processor = TextProcessor()
    
    if not output_dir:
        if os.path.isdir(input_path):
            output_dir = os.path.join(input_path, 'processed')
        else:
            output_dir = os.path.join(os.path.dirname(input_path), 'processed')
    
    if not csv_output:
        csv_output = os.path.join(output_dir, 'consolidated.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    
    if os.path.isdir(input_path):
        # Process directory
        word_docs, csv_path = processor.process_directory(
            input_path, output_dir, csv_output,
            cleaning_operations=operations if operations else None,
            recursive=recursive, split_paragraphs=split_paragraphs
        )
        click.echo(f"Processed {len(word_docs)} files.")
    else:
        # Process single file
        word_doc, cleaned_text = processor.process_file(
            input_path, output_dir,
            cleaning_operations=operations if operations else None
        )
        
        # Consolidate to CSV
        csv_path = processor.consolidator.consolidate_files(
            [input_path], csv_output, 
            lambda f: processor.cleaner.clean_text(
                processor.converter.extract_text(f),
                operations=operations if operations else None
            ),
            split_paragraphs=split_paragraphs
        )
        
        click.echo(f"Processed file {input_path} to {word_doc}")
    
    click.echo(f"Consolidated data saved to {csv_path}")


@cli.command('to-csv')
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--language', '-l', default='english', type=click.Choice(['english', 'chinese']), 
              help='Language for NLP processing')
@click.option('--operations', '-p', multiple=True, 
              type=click.Choice(['remove_urls', 'remove_html_tags', 'remove_punctuation', 
                               'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                               'remove_special_chars', 'stem_text', 'lemmatize_text']),
              help='Cleaning operations to apply')
@click.option('--extensions', '-e', multiple=True, help='File extensions to include')
def to_csv_command(input_path, output_csv, recursive, language, operations, extensions):
    """Convert text files to CSV with NLP cleaning (one file per row)."""
    processor = TextProcessor(language=language)
    
    # Define a text extractor function that includes cleaning
    def text_extractor_with_cleaning(file_path):
        text = processor.converter.extract_text(file_path)
        return processor.cleaner.clean_text(text, operations=operations if operations else None)
    
    if os.path.isdir(input_path):
        # Process directory
        csv_path = processor.consolidator.create_oneline_csv(
            processor.get_supported_files(input_path, recursive, extensions),
            output_csv,
            processor.converter.extract_text,
            lambda text: processor.cleaner.clean_text(text, operations=operations if operations else None)
        )
    else:
        # Process single file
        csv_path = processor.consolidator.create_oneline_csv(
            [input_path],
            output_csv,
            processor.converter.extract_text,
            lambda text: processor.cleaner.clean_text(text, operations=operations if operations else None)
        )
    
    click.echo(f"Converted and cleaned data saved to {csv_path}")


if __name__ == '__main__':
    cli()
