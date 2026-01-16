"""
Topic Identification Result Table Generator

This module generates publication-quality topic identification result tables,
similar to those commonly found in LDA/topic modeling research papers.

Features:
- Generate formatted tables with topic ID, name, strength, keywords, and category
- Export to CSV, Excel, PNG, and LaTeX formats
- Support for custom topic naming and categorization
- Chinese font support for academic publications
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TopicTableGenerator:
    """
    Generator for topic identification result tables.
    
    Creates publication-ready tables showing:
    - Topic ID/序号
    - Topic Name/主题名称 (semantic label)
    - Topic Strength/主题强度 (proportion in corpus)
    - Topic Keywords/主题词项 (representative words)
    - Topic Category/类型 (optional classification)
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150,
        font_family: str = None,
        language: str = 'zh'
    ):
        """
        Initialize table generator.
        
        Args:
            output_dir: Directory to save outputs
            figsize: Default figure size for PNG export
            dpi: Default DPI for PNG export
            font_family: Font family for rendering (auto-detect if None)
            language: Language for headers ('zh' for Chinese, 'en' for English)
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.language = language
        
        # Set up font for Chinese characters
        self.font_family = font_family or self._detect_chinese_font()
        
        # Column headers based on language
        self.headers = self._get_headers(language)
        
        # Store generated data
        self.table_data = None
        self.df = None
    
    def _detect_chinese_font(self) -> str:
        """Detect available Chinese font on the system."""
        chinese_fonts = [
            'SimHei', 'SimSun', 'Microsoft YaHei', 'STHeiti', 
            'STSong', 'PingFang SC', 'Noto Sans CJK SC',
            'WenQuanYi Micro Hei', 'Source Han Sans CN'
        ]
        
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        for font in chinese_fonts:
            if font in available_fonts:
                logger.info(f"Using Chinese font: {font}")
                return font
        
        logger.warning("No Chinese font found, using default font")
        return 'DejaVu Sans'
    
    def _get_headers(self, language: str) -> Dict[str, str]:
        """Get column headers based on language."""
        if language == 'zh':
            return {
                'id': '序号',
                'name': '主题名称',
                'strength': '主题强度',
                'keywords': '主题词项',
                'category': '类型'
            }
        else:
            return {
                'id': 'ID',
                'name': 'Topic Name',
                'strength': 'Strength',
                'keywords': 'Keywords',
                'category': 'Category'
            }
    
    def generate_table(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        theta: np.ndarray = None,
        topic_names: Dict[int, str] = None,
        topic_categories: Dict[int, str] = None,
        top_n_words: int = 10,
        strength_format: str = '{:.6f}'
    ) -> pd.DataFrame:
        """
        Generate topic identification result table.
        
        Args:
            topic_words: List of (topic_id, [(word, weight), ...]) tuples
            theta: Document-topic distribution matrix (D x K) for calculating strength
            topic_names: Optional dict mapping topic_id to semantic name
            topic_categories: Optional dict mapping topic_id to category
            top_n_words: Number of top words to include per topic
            strength_format: Format string for strength values
            
        Returns:
            DataFrame with topic identification results
        """
        rows = []
        
        # Calculate topic strengths from theta if provided
        if theta is not None:
            topic_strengths = theta.mean(axis=0)
        else:
            topic_strengths = None
        
        for idx, (topic_id, words) in enumerate(topic_words):
            # Get top words
            top_words = [w for w, _ in words[:top_n_words]]
            keywords_str = ' '.join(top_words)
            
            # Get strength
            if topic_strengths is not None and topic_id < len(topic_strengths):
                strength = topic_strengths[topic_id]
                strength_str = strength_format.format(strength)
            else:
                # Calculate from word weights as fallback
                total_weight = sum(w for _, w in words[:top_n_words])
                strength_str = strength_format.format(total_weight / top_n_words if words else 0)
            
            # Get topic name (auto-generate if not provided)
            if topic_names and topic_id in topic_names:
                name = topic_names[topic_id]
            else:
                # Auto-generate name from top 2-3 words
                name = self._auto_generate_name(top_words[:3])
            
            # Get category
            category = topic_categories.get(topic_id, '') if topic_categories else ''
            
            rows.append({
                self.headers['id']: idx + 1,
                self.headers['name']: name,
                self.headers['strength']: strength_str,
                self.headers['keywords']: keywords_str,
                self.headers['category']: category
            })
        
        self.df = pd.DataFrame(rows)
        self.table_data = rows
        
        return self.df
    
    def _auto_generate_name(self, top_words: List[str]) -> str:
        """Auto-generate topic name from top words."""
        if not top_words:
            return "未命名主题"
        
        # Join top 2-3 words with appropriate separator
        if self.language == 'zh':
            # Check if words are Chinese
            if any('\u4e00' <= c <= '\u9fff' for c in ''.join(top_words)):
                return '/'.join(top_words[:2])
            else:
                return ', '.join(top_words[:2])
        else:
            return ', '.join(top_words[:2])
    
    def to_csv(self, filename: str = 'topic_table.csv', encoding: str = 'utf-8-sig') -> str:
        """
        Export table to CSV file.
        
        Args:
            filename: Output filename
            encoding: File encoding (utf-8-sig for Excel compatibility)
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        self.df.to_csv(filepath, index=False, encoding=encoding)
        logger.info(f"Table saved to CSV: {filepath}")
        return filepath
    
    def to_excel(self, filename: str = 'topic_table.xlsx', sheet_name: str = '主题识别结果') -> str:
        """
        Export table to Excel file with formatting.
        
        Args:
            filename: Output filename
            sheet_name: Excel sheet name
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]
                
                # Adjust column widths
                for idx, col in enumerate(self.df.columns):
                    max_length = max(
                        self.df[col].astype(str).map(len).max(),
                        len(col)
                    )
                    # Limit max width
                    adjusted_width = min(max_length + 2, 60)
                    worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width
            
            logger.info(f"Table saved to Excel: {filepath}")
            return filepath
            
        except ImportError:
            logger.warning("openpyxl not available, saving as CSV instead")
            return self.to_csv(filename.replace('.xlsx', '.csv'))
    
    def to_png(
        self,
        filename: str = 'topic_table.png',
        title: str = None,
        show_category: bool = True
    ) -> str:
        """
        Export table as PNG image (publication-ready).
        
        Args:
            filename: Output filename
            title: Table title
            show_category: Whether to include category column
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        # Prepare data
        df_display = self.df.copy()
        if not show_category or self.headers['category'] not in df_display.columns:
            if self.headers['category'] in df_display.columns:
                # Check if category column is empty
                if df_display[self.headers['category']].str.strip().eq('').all():
                    df_display = df_display.drop(columns=[self.headers['category']])
        
        # Set up figure
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.axis('off')
        
        # Set font
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.sans-serif'] = [self.font_family]
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create table
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4472C4'] * len(df_display.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header row
        for j in range(len(df_display.columns)):
            cell = table[(0, j)]
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        
        # Style data rows with alternating colors
        for i in range(1, len(df_display) + 1):
            for j in range(len(df_display.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#D6DCE5')
                else:
                    cell.set_facecolor('#FFFFFF')
                
                # Left-align keywords column
                if df_display.columns[j] == self.headers['keywords']:
                    cell.set_text_props(ha='left')
        
        # Add title
        if title is None:
            title = '表1 主题识别结果表' if self.language == 'zh' else 'Table 1. Topic Identification Results'
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20,
                 fontfamily=self.font_family)
        
        # Save
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Table saved to PNG: {filepath}")
        return filepath
    
    def to_png_academic(
        self,
        filename: str = 'topic_table_academic.png',
        title: str = None,
        show_category: bool = True,
        max_keywords_display: int = 10
    ) -> str:
        """
        Export table as academic-style PNG (三线表风格，类似论文表格).
        
        模仿参考图片样式：简洁的三线表，黑色边框，白色背景
        
        Args:
            filename: Output filename
            title: Table title (e.g., "表1 托育政策文本主题词项")
            show_category: Whether to include category column
            max_keywords_display: Maximum number of keywords to display per topic
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        # Prepare data
        df_display = self.df.copy()
        
        # Truncate keywords to max_keywords_display words
        keywords_col = self.headers['keywords']
        if keywords_col in df_display.columns:
            df_display[keywords_col] = df_display[keywords_col].apply(
                lambda x: ' '.join(str(x).split()[:max_keywords_display]) if pd.notna(x) else ''
            )
        
        if not show_category:
            if self.headers['category'] in df_display.columns:
                df_display = df_display.drop(columns=[self.headers['category']])
        elif self.headers['category'] in df_display.columns:
            if df_display[self.headers['category']].str.strip().eq('').all():
                df_display = df_display.drop(columns=[self.headers['category']])
        
        # Set font
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.sans-serif'] = [self.font_family]
        plt.rcParams['axes.unicode_minus'] = False
        
        # Calculate figure size based on content
        n_rows = len(df_display)
        n_cols = len(df_display.columns)
        fig_width = max(16, n_cols * 3)
        fig_height = max(6, n_rows * 0.45 + 2)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        
        # Create table with academic style (三线表)
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc='center',
            loc='center',
            edges='horizontal'  # Only horizontal lines
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        
        # Set column widths
        col_widths = []
        for j, col in enumerate(df_display.columns):
            if col == self.headers['id']:
                col_widths.append(0.05)
            elif col == self.headers['name']:
                col_widths.append(0.12)
            elif col == self.headers['strength']:
                col_widths.append(0.08)
            elif col == self.headers['keywords']:
                col_widths.append(0.55)
            elif col == self.headers['category']:
                col_widths.append(0.12)
            else:
                col_widths.append(0.1)
        
        # Style header row (bold, with top and bottom border)
        for j in range(n_cols):
            cell = table[(0, j)]
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#F0F0F0')  # Light gray header
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            cell.set_width(col_widths[j])
        
        # Style data rows (white background, thin borders)
        for i in range(1, n_rows + 1):
            for j in range(n_cols):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_edgecolor('#DDDDDD')
                cell.set_linewidth(0.5)
                cell.set_width(col_widths[j])
                
                # Left-align keywords column for readability
                if df_display.columns[j] == self.headers['keywords']:
                    cell.set_text_props(ha='left')
        
        # Add top and bottom borders (三线表)
        # Draw lines manually for cleaner look
        ax.axhline(y=0.95, xmin=0.05, xmax=0.95, color='black', linewidth=2)
        ax.axhline(y=0.05, xmin=0.05, xmax=0.95, color='black', linewidth=2)
        
        # Add title above table
        if title is None:
            title = '表1 主题识别结果表' if self.language == 'zh' else 'Table 1. Topic Identification Results'
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98,
                    fontfamily=self.font_family)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Academic table saved to PNG: {filepath}")
        return filepath
    
    def to_latex(
        self,
        filename: str = 'topic_table.tex',
        caption: str = None,
        label: str = 'tab:topics'
    ) -> str:
        """
        Export table to LaTeX format for academic papers.
        
        Args:
            filename: Output filename
            caption: Table caption
            label: LaTeX label for referencing
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        if caption is None:
            caption = '主题识别结果' if self.language == 'zh' else 'Topic Identification Results'
        
        # Generate LaTeX
        latex_str = self.df.to_latex(
            index=False,
            caption=caption,
            label=label,
            column_format='c' + 'l' * (len(self.df.columns) - 1),
            escape=False
        )
        
        # Add booktabs style
        latex_str = latex_str.replace('\\toprule', '\\hline\\hline')
        latex_str = latex_str.replace('\\midrule', '\\hline')
        latex_str = latex_str.replace('\\bottomrule', '\\hline\\hline')
        
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        logger.info(f"Table saved to LaTeX: {filepath}")
        return filepath
    
    def to_html(self, filename: str = 'topic_table.html', title: str = None) -> str:
        """
        Export table to HTML format.
        
        Args:
            filename: Output filename
            title: Page title
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No table data. Call generate_table() first.")
        
        if title is None:
            title = '主题识别结果表' if self.language == 'zh' else 'Topic Identification Results'
        
        # Generate styled HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            padding: 20px;
        }}
        h2 {{
            text-align: center;
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4472C4;
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: bold;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .keywords {{
            text-align: left;
            max-width: 400px;
        }}
        .strength {{
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    {self.df.to_html(index=False, classes='topic-table', escape=False)}
</body>
</html>
"""
        
        filepath = os.path.join(self.output_dir, filename) if self.output_dir else filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Table saved to HTML: {filepath}")
        return filepath
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the topics.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            return {}
        
        # Parse strength values
        strength_col = self.headers['strength']
        strengths = pd.to_numeric(self.df[strength_col], errors='coerce')
        
        return {
            'num_topics': len(self.df),
            'total_strength': strengths.sum(),
            'avg_strength': strengths.mean(),
            'max_strength': strengths.max(),
            'min_strength': strengths.min(),
            'top_topic': self.df.loc[strengths.idxmax(), self.headers['name']] if not strengths.isna().all() else None
        }


def generate_topic_table(
    topic_words: List[Tuple[int, List[Tuple[str, float]]]],
    theta: np.ndarray = None,
    output_dir: str = None,
    topic_names: Dict[int, str] = None,
    topic_categories: Dict[int, str] = None,
    formats: List[str] = ['csv', 'png'],
    language: str = 'zh',
    title: str = None
) -> Dict[str, str]:
    """
    Convenience function to generate topic identification table in multiple formats.
    
    Args:
        topic_words: List of (topic_id, [(word, weight), ...]) tuples
        theta: Document-topic distribution matrix
        output_dir: Directory to save outputs
        topic_names: Optional dict mapping topic_id to semantic name
        topic_categories: Optional dict mapping topic_id to category
        formats: List of output formats ('csv', 'excel', 'png', 'latex', 'html')
        language: Language for headers ('zh' or 'en')
        title: Table title
        
    Returns:
        Dictionary mapping format to file path
    """
    generator = TopicTableGenerator(output_dir=output_dir, language=language)
    generator.generate_table(
        topic_words=topic_words,
        theta=theta,
        topic_names=topic_names,
        topic_categories=topic_categories
    )
    
    output_files = {}
    
    for fmt in formats:
        try:
            if fmt == 'csv':
                output_files['csv'] = generator.to_csv()
            elif fmt == 'excel':
                output_files['excel'] = generator.to_excel()
            elif fmt == 'png':
                output_files['png'] = generator.to_png(title=title)
            elif fmt == 'latex':
                output_files['latex'] = generator.to_latex()
            elif fmt == 'html':
                output_files['html'] = generator.to_html(title=title)
        except Exception as e:
            logger.error(f"Error generating {fmt} format: {e}")
    
    return output_files
