"""
Utility functions for the Data Visualizer application.

This package contains modules for file handling, data analysis,
text processing, and data visualization.
"""

# Import key functions from file_handlers
from .file_handlers import (
    read_file,
    extract_text_from_pdf,
    extract_text_from_docx,
    convert_df_to_csv_download_link,
    infer_column_types
)

# Import key functions from data_analysis
from .data_analysis import (
    analyze_tabular_data,
    analyze_text_data,
    generate_data_summary
)

# Import key functions from visualizations
from .visualizations import create_visualization

# Define package version
__version__ = '0.1.0'

# Define what gets imported with "from utils import *"
__all__ = [
    # File handlers
    'read_file',
    'extract_text_from_pdf',
    'extract_text_from_docx',
    'convert_df_to_csv_download_link',
    'infer_column_types',
    
    # Data analysis
    'analyze_tabular_data',
    'analyze_text_data',
    'generate_data_summary',
    
    # Visualizations
    'create_visualization'
]