import os
import pandas as pd
import PyPDF2
import docx
import io
import base64

def read_file(file_obj, file_type=None):
    """
    Read various file types and convert to appropriate format for analysis
    
    Parameters:
    -----------
    file_obj : UploadedFile
        The file object from Streamlit file uploader
    file_type : str
        The type of file (optional, will be inferred from file name if not provided)
        
    Returns:
    --------
    data : pd.DataFrame or str
        The data extracted from the file
    file_type : str
        The type of file that was processed
    error : str or None
        Error message if any occurred
    """
    if file_type is None:
        file_type = file_obj.name.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            data = pd.read_csv(file_obj)
            return data, 'tabular', None
            
        elif file_type in ['xlsx', 'xls']:
            data = pd.read_excel(file_obj)
            return data, 'tabular', None
            
        elif file_type == 'json':
            try:
                # Try standard JSON parsing first
                data = pd.read_json(file_obj)
                return data, 'tabular', None
            except ValueError as json_error:
                if "Trailing data" in str(json_error):
                    # Try to handle JSON Lines format
                    try:
                        # Reset file position to beginning
                        file_obj.seek(0)
                        data = pd.read_json(file_obj, lines=True)
                        return data, 'tabular', None
                    except:
                        # If JSON Lines also fails, provide helpful error
                        return None, None, f"Invalid JSON format: {str(json_error)}. Try validating your JSON file structure."
                else:
                    # For other JSON parsing errors
                    return None, None, f"JSON parsing error: {str(json_error)}"
            
        else:
            return None, None, f"Unsupported file type: {file_type}. Please upload CSV, Excel, or JSON."
            
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}"

def extract_text_from_pdf(file_obj):
    """Extract text from PDF file"""
    bytes_data = file_obj.getvalue()
    pdf_file = io.BytesIO(bytes_data)
    
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    
    return text

def extract_text_from_docx(file_obj):
    """Extract text from DOCX file"""
    bytes_data = file_obj.getvalue()
    doc_file = io.BytesIO(bytes_data)
    
    doc = docx.Document(doc_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    return text

def convert_df_to_csv_download_link(df):
    """
    Convert dataframe to CSV download link
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to convert
        
    Returns:
    --------
    href : str
        HTML link for downloading the data as CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data_export.csv" class="download-link">📄 Download CSV File</a>'
    return href

def convert_df_to_excel_download_link(df):
    """
    Convert dataframe to Excel download link
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to convert
        
    Returns:
    --------
    href : str
        HTML link for downloading the data as Excel
    """
    output = io.BytesIO()  # Changed from BytesIO() to io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_export.xlsx" class="download-link">📊 Download Excel File</a>'
    return href

def convert_df_to_json_download_link(df):
    """
    Convert dataframe to JSON download link
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to convert
        
    Returns:
    --------
    href : str
        HTML link for downloading the data as JSON
    """
    json_data = df.to_json(orient='records', indent=2)
    b64 = base64.b64encode(json_data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="data_export.json" class="download-link">📋 Download JSON File</a>'
    return href

def infer_column_types(df):
    """Infer column types for a DataFrame"""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 10 and df[col].nunique() / len(df[col]) < 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numerical'
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[col] = 'datetime'
        else:
            if df[col].nunique() < 10 or df[col].nunique() / len(df[col]) < 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
    
    return column_types