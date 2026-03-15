"""
Phase 2: The Mapping - Column Identity Confirmation

Handles user confirmation of column mappings:
- Target Text (required)
- Time Dimension (optional, enables DTM)
- Structural Covariates (optional, enables STM)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class ColumnRole(Enum):
    """Column role types."""
    TEXT = 'text'
    TIME = 'time'
    COVARIATE = 'covariate'
    ID = 'id'
    LABEL = 'label'
    IGNORE = 'ignore'


@dataclass
class ColumnMapping:
    """Single column mapping configuration."""
    column_name: str
    role: ColumnRole
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'column_name': self.column_name,
            'role': self.role.value,
            'enabled': self.enabled
        }


@dataclass
class MappingConfig:
    """Complete mapping configuration."""
    text_column: str
    time_column: Optional[str] = None
    covariate_columns: List[str] = field(default_factory=list)
    id_column: Optional[str] = None
    label_column: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'text_column': self.text_column,
            'time_column': self.time_column,
            'covariate_columns': self.covariate_columns,
            'id_column': self.id_column,
            'label_column': self.label_column,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MappingConfig':
        return cls(
            text_column=data['text_column'],
            time_column=data.get('time_column'),
            covariate_columns=data.get('covariate_columns', []),
            id_column=data.get('id_column'),
            label_column=data.get('label_column'),
        )
    
    def validate(self) -> Dict[str, Any]:
        """Validate mapping configuration."""
        errors = []
        warnings = []
        
        # Text column is required
        if not self.text_column:
            errors.append("Text column is required for topic modeling")
        
        # Check for duplicate columns
        all_columns = [self.text_column, self.time_column, self.id_column, self.label_column]
        all_columns = [c for c in all_columns if c]
        all_columns.extend(self.covariate_columns)
        
        if len(all_columns) != len(set(all_columns)):
            errors.append("Duplicate column assignments detected")
        
        # Warnings
        if not self.time_column:
            warnings.append("No time column selected - DTM will not be available")
        
        if not self.covariate_columns:
            warnings.append("No covariate columns selected - STM will not be available")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'available_models': self.get_available_models()
        }
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get which models are available based on mapping."""
        return {
            'lda': True,
            'hdp': True,
            'btm': True,
            'nvdm': True,
            'gsm': True,
            'prodlda': True,
            'etm': True,
            'ctm': True,
            'bertopic': True,
            'dtm': self.time_column is not None,
            'stm': len(self.covariate_columns) > 0,
        }


class ColumnMapper:
    """Manages column mapping workflow."""
    
    def __init__(self, scan_result: Dict[str, Any]):
        """
        Initialize mapper with scan result.
        
        Args:
            scan_result: Result from CSVScanner.scan()
        """
        self.scan_result = scan_result
        self.columns = {c['column_name']: c for c in scan_result['columns']}
        self.mapping_config: Optional[MappingConfig] = None
        
        # Initialize with recommendations
        self._init_from_recommendations()
    
    def _init_from_recommendations(self):
        """Initialize mapping from scan recommendations."""
        rec = self.scan_result.get('recommendations', {})
        
        self.mapping_config = MappingConfig(
            text_column=rec.get('text_column'),
            time_column=rec.get('time_column'),
            covariate_columns=rec.get('covariate_columns', []),
        )
    
    def get_mapping_table(self) -> List[Dict]:
        """
        Get mapping table for frontend display.
        
        Returns:
            List of column info with current mapping
        """
        table = []
        
        for col_name, col_info in self.columns.items():
            row = {
                'column_name': col_name,
                'dtype': col_info['dtype'],
                'sample_values': col_info['sample_values'][:3],
                'suggested_role': col_info['suggested_identity'],
                'confidence': col_info['confidence'],
                'current_role': self._get_current_role(col_name),
                'selectable_roles': self._get_selectable_roles(col_info),
            }
            table.append(row)
        
        return table
    
    def _get_current_role(self, col_name: str) -> str:
        """Get current role for a column."""
        if not self.mapping_config:
            return 'ignore'
        
        if col_name == self.mapping_config.text_column:
            return 'text'
        if col_name == self.mapping_config.time_column:
            return 'time'
        if col_name in self.mapping_config.covariate_columns:
            return 'covariate'
        if col_name == self.mapping_config.id_column:
            return 'id'
        if col_name == self.mapping_config.label_column:
            return 'label'
        
        return 'ignore'
    
    def _get_selectable_roles(self, col_info: Dict) -> List[str]:
        """Get selectable roles for a column based on its characteristics."""
        roles = ['ignore']
        
        # Text role - for string columns with substantial content
        if col_info['dtype'] == 'object' and col_info.get('avg_length', 0) > 10:
            roles.append('text')
        
        # Time role - for date-like or year-like columns
        if col_info.get('is_date_like') or col_info.get('is_year_like'):
            roles.append('time')
        elif col_info['is_numeric']:
            roles.append('time')  # Allow numeric columns as time
        
        # Covariate role - for categorical columns
        if col_info['unique_ratio'] < 0.5 or col_info['unique_count'] < 100:
            roles.append('covariate')
        
        # ID role - for high-uniqueness columns
        if col_info['unique_ratio'] > 0.8:
            roles.append('id')
        
        # Label role - for categorical columns
        if col_info['unique_ratio'] < 0.3:
            roles.append('label')
        
        return list(set(roles))
    
    def set_text_column(self, column_name: str) -> bool:
        """Set text column."""
        if column_name not in self.columns:
            return False
        self.mapping_config.text_column = column_name
        return True
    
    def set_time_column(self, column_name: Optional[str]) -> bool:
        """Set time column (None to disable DTM)."""
        if column_name and column_name not in self.columns:
            return False
        self.mapping_config.time_column = column_name
        return True
    
    def set_covariate_columns(self, column_names: List[str]) -> bool:
        """Set covariate columns."""
        for col in column_names:
            if col not in self.columns:
                return False
        self.mapping_config.covariate_columns = column_names
        return True
    
    def add_covariate_column(self, column_name: str) -> bool:
        """Add a covariate column."""
        if column_name not in self.columns:
            return False
        if column_name not in self.mapping_config.covariate_columns:
            self.mapping_config.covariate_columns.append(column_name)
        return True
    
    def remove_covariate_column(self, column_name: str) -> bool:
        """Remove a covariate column."""
        if column_name in self.mapping_config.covariate_columns:
            self.mapping_config.covariate_columns.remove(column_name)
            return True
        return False
    
    def set_id_column(self, column_name: Optional[str]) -> bool:
        """Set ID column."""
        if column_name and column_name not in self.columns:
            return False
        self.mapping_config.id_column = column_name
        return True
    
    def update_mapping(self, mapping_dict: Dict) -> Dict[str, Any]:
        """
        Update mapping from frontend submission.
        
        Args:
            mapping_dict: {
                'text_column': str,
                'time_column': str or None,
                'covariate_columns': List[str],
                'id_column': str or None,
                'label_column': str or None,
            }
            
        Returns:
            Validation result
        """
        self.mapping_config = MappingConfig.from_dict(mapping_dict)
        return self.validate()
    
    def validate(self) -> Dict[str, Any]:
        """Validate current mapping configuration."""
        if not self.mapping_config:
            return {
                'valid': False,
                'errors': ['No mapping configuration'],
                'warnings': [],
                'available_models': {}
            }
        return self.mapping_config.validate()
    
    def get_config(self) -> MappingConfig:
        """Get current mapping configuration."""
        return self.mapping_config
    
    def to_dict(self) -> Dict:
        """Export mapping as dictionary."""
        return {
            'mapping': self.mapping_config.to_dict() if self.mapping_config else None,
            'validation': self.validate(),
            'mapping_table': self.get_mapping_table(),
        }


def create_mapper_from_scan(scan_result: Dict[str, Any]) -> ColumnMapper:
    """Create a ColumnMapper from scan result."""
    return ColumnMapper(scan_result)
