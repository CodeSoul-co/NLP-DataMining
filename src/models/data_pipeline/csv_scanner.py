"""
Phase 1: The Sniff - CSV Upload and Column Analysis

Analyzes CSV columns to suggest their identity:
- text: Main text content for topic modeling
- time: Temporal dimension for DTM
- covariate: Structural covariates for STM
- id: Document identifier
- label: Category/label column
- ignore: Columns to skip
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


class CSVScanner:
    """Scans CSV files and suggests column identities."""
    
    # Column identity types
    IDENTITY_TEXT = 'text'
    IDENTITY_TIME = 'time'
    IDENTITY_COVARIATE = 'covariate'
    IDENTITY_ID = 'id'
    IDENTITY_LABEL = 'label'
    IDENTITY_IGNORE = 'ignore'
    
    # Date patterns for detection
    DATE_PATTERNS = [
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD-MM-YYYY or MM/DD/YYYY
        r'^\d{4}$',                         # YYYY (year only)
        r'^\d{4}[-/]\d{1,2}$',             # YYYY-MM
    ]
    
    def __init__(self, sample_rows: int = 100):
        """
        Initialize scanner.
        
        Args:
            sample_rows: Number of rows to sample for analysis
        """
        self.sample_rows = sample_rows
        self.df = None
        self.file_path = None
        self.analysis_result = None
    
    def load_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load CSV file and sample rows.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            
        Returns:
            Sampled DataFrame
        """
        self.file_path = file_path
        
        # Try different encodings if utf-8 fails
        encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for enc in encodings:
            try:
                self.df = pd.read_csv(file_path, encoding=enc, nrows=self.sample_rows)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if self.df is None:
            raise ValueError(f"Could not read CSV file with any supported encoding")
        
        return self.df
    
    def analyze_column(self, col_name: str) -> Dict[str, Any]:
        """
        Analyze a single column and suggest its identity.
        
        Args:
            col_name: Column name
            
        Returns:
            Column analysis result
        """
        if self.df is None:
            raise RuntimeError("No CSV loaded. Call load_csv() first.")
        
        col = self.df[col_name]
        non_null = col.dropna()
        
        # Basic statistics
        total_count = len(col)
        non_null_count = len(non_null)
        null_ratio = 1 - (non_null_count / total_count) if total_count > 0 else 0
        unique_count = non_null.nunique()
        unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0
        
        # Determine data type
        dtype = str(col.dtype)
        
        # Analyze based on content
        analysis = {
            'column_name': col_name,
            'dtype': dtype,
            'total_count': total_count,
            'non_null_count': non_null_count,
            'null_ratio': round(null_ratio, 4),
            'unique_count': unique_count,
            'unique_ratio': round(unique_ratio, 4),
            'sample_values': non_null.head(5).tolist() if non_null_count > 0 else [],
        }
        
        # Text analysis (for object/string types)
        if dtype == 'object':
            text_lengths = non_null.astype(str).str.len()
            analysis['avg_length'] = round(text_lengths.mean(), 2) if len(text_lengths) > 0 else 0
            analysis['max_length'] = int(text_lengths.max()) if len(text_lengths) > 0 else 0
            analysis['min_length'] = int(text_lengths.min()) if len(text_lengths) > 0 else 0
            
            # Check if it looks like a date
            analysis['is_date_like'] = self._is_date_like(non_null)
            
            # Check if it looks like text content
            analysis['is_text_content'] = self._is_text_content(non_null, analysis['avg_length'])
        else:
            analysis['avg_length'] = 0
            analysis['max_length'] = 0
            analysis['min_length'] = 0
            analysis['is_date_like'] = False
            analysis['is_text_content'] = False
        
        # Check if numeric (potential year or ID)
        analysis['is_numeric'] = pd.api.types.is_numeric_dtype(col)
        if analysis['is_numeric']:
            analysis['min_value'] = float(non_null.min()) if non_null_count > 0 else None
            analysis['max_value'] = float(non_null.max()) if non_null_count > 0 else None
            # Check if it looks like a year
            analysis['is_year_like'] = self._is_year_like(non_null)
        else:
            analysis['is_year_like'] = False
        
        # Suggest identity
        analysis['suggested_identity'] = self._suggest_identity(analysis, col_name)
        analysis['confidence'] = self._calculate_confidence(analysis)
        
        return analysis
    
    def _is_date_like(self, series: pd.Series) -> bool:
        """Check if series values look like dates."""
        if len(series) == 0:
            return False
        
        sample = series.head(20).astype(str)
        date_matches = 0
        
        for val in sample:
            for pattern in self.DATE_PATTERNS:
                if re.match(pattern, str(val).strip()):
                    date_matches += 1
                    break
        
        return date_matches / len(sample) > 0.7
    
    def _is_year_like(self, series: pd.Series) -> bool:
        """Check if numeric series looks like years."""
        if len(series) == 0:
            return False
        
        try:
            min_val = series.min()
            max_val = series.max()
            # Reasonable year range
            return 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100
        except:
            return False
    
    def _is_text_content(self, series: pd.Series, avg_length: float) -> bool:
        """Check if series looks like main text content."""
        # Text content typically has longer strings
        if avg_length < 50:
            return False
        
        # Check for word-like content
        sample = series.head(10).astype(str)
        word_counts = sample.str.split().str.len()
        avg_words = word_counts.mean()
        
        return avg_words > 5
    
    def _suggest_identity(self, analysis: Dict, col_name: str) -> str:
        """Suggest column identity based on analysis."""
        name_lower = col_name.lower()
        
        # Check column name hints
        text_hints = ['text', 'content', 'body', 'message', 'description', 'abstract', 'title', 'document']
        time_hints = ['time', 'date', 'year', 'month', 'day', 'timestamp', 'created', 'published']
        id_hints = ['id', 'index', 'key', 'uid', 'uuid']
        label_hints = ['label', 'category', 'class', 'type', 'tag', 'topic']
        
        # Name-based suggestions
        if any(hint in name_lower for hint in text_hints):
            if analysis.get('is_text_content', False) or analysis.get('avg_length', 0) > 30:
                return self.IDENTITY_TEXT
        
        if any(hint in name_lower for hint in time_hints):
            return self.IDENTITY_TIME
        
        if any(hint in name_lower for hint in id_hints):
            return self.IDENTITY_ID
        
        if any(hint in name_lower for hint in label_hints):
            return self.IDENTITY_COVARIATE
        
        # Content-based suggestions
        if analysis.get('is_text_content', False):
            return self.IDENTITY_TEXT
        
        if analysis.get('is_date_like', False) or analysis.get('is_year_like', False):
            return self.IDENTITY_TIME
        
        # Low unique ratio with categorical values -> covariate
        if analysis['unique_ratio'] < 0.1 and analysis['unique_count'] < 50:
            return self.IDENTITY_COVARIATE
        
        # High unique ratio with numeric -> ID
        if analysis['is_numeric'] and analysis['unique_ratio'] > 0.9:
            return self.IDENTITY_ID
        
        # Default to ignore
        return self.IDENTITY_IGNORE
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for the suggestion."""
        identity = analysis['suggested_identity']
        confidence = 0.5  # Base confidence
        
        if identity == self.IDENTITY_TEXT:
            if analysis.get('is_text_content', False):
                confidence += 0.3
            if analysis.get('avg_length', 0) > 100:
                confidence += 0.2
        
        elif identity == self.IDENTITY_TIME:
            if analysis.get('is_date_like', False):
                confidence += 0.4
            if analysis.get('is_year_like', False):
                confidence += 0.3
        
        elif identity == self.IDENTITY_COVARIATE:
            if analysis['unique_ratio'] < 0.05:
                confidence += 0.3
            if analysis['unique_count'] < 20:
                confidence += 0.2
        
        elif identity == self.IDENTITY_ID:
            if analysis['unique_ratio'] > 0.95:
                confidence += 0.4
        
        return min(confidence, 1.0)
    
    def scan(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform full scan of CSV file.
        
        Args:
            file_path: Path to CSV file (optional if already loaded)
            
        Returns:
            Complete analysis result as JSON-serializable dict
        """
        if file_path:
            self.load_csv(file_path)
        
        if self.df is None:
            raise RuntimeError("No CSV loaded. Provide file_path or call load_csv() first.")
        
        columns_analysis = []
        for col in self.df.columns:
            col_analysis = self.analyze_column(col)
            columns_analysis.append(col_analysis)
        
        # Find best candidates for each identity
        text_candidates = [c for c in columns_analysis if c['suggested_identity'] == self.IDENTITY_TEXT]
        time_candidates = [c for c in columns_analysis if c['suggested_identity'] == self.IDENTITY_TIME]
        covariate_candidates = [c for c in columns_analysis if c['suggested_identity'] == self.IDENTITY_COVARIATE]
        
        # Sort by confidence
        text_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        time_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.analysis_result = {
            'file_path': str(self.file_path),
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': columns_analysis,
            'recommendations': {
                'text_column': text_candidates[0]['column_name'] if text_candidates else None,
                'time_column': time_candidates[0]['column_name'] if time_candidates else None,
                'covariate_columns': [c['column_name'] for c in covariate_candidates],
            },
            'model_availability': {
                'basic_models': True,  # LDA, HDP, BTM, NVDM, GSM, ProdLDA, ETM, CTM, BERTopic
                'dtm': len(time_candidates) > 0,
                'stm': len(covariate_candidates) > 0,
            },
            'scan_time': datetime.now().isoformat(),
        }
        
        return self.analysis_result
    
    def to_json(self) -> str:
        """Export analysis result as JSON string."""
        import json
        if self.analysis_result is None:
            raise RuntimeError("No analysis result. Call scan() first.")
        return json.dumps(self.analysis_result, ensure_ascii=False, indent=2)


def scan_csv(file_path: str, sample_rows: int = 100) -> Dict[str, Any]:
    """
    Convenience function to scan a CSV file.
    
    Args:
        file_path: Path to CSV file
        sample_rows: Number of rows to sample
        
    Returns:
        Analysis result
    """
    scanner = CSVScanner(sample_rows=sample_rows)
    return scanner.scan(file_path)


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python csv_scanner.py <csv_file>")
        sys.exit(1)
    
    result = scan_csv(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
