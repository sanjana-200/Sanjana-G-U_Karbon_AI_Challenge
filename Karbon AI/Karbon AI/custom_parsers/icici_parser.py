import pandas as pd
import pdfplumber
import re

def parse(file_path: str) -> pd.DataFrame:
    """
    Parse ICICI bank PDF/CSV file and return a pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the PDF/CSV file.
    
    Returns:
    pd.DataFrame: Parsed DataFrame.
    """

    # Check if file is a PDF or CSV
    if file_path.endswith('.pdf'):
        # Extract text from PDF using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        
        # Split text into lines
        lines = text.split('\n')
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Split lines into columns
        columns = []
        for line in lines:
            columns.append(re.split('\s+', line))
        
        # Transpose columns
        columns = list(map(list, zip(*columns)))
        
        # Create DataFrame
        df = pd.DataFrame(columns)
        
    elif file_path.endswith('.csv'):
        # Read CSV file directly
        df = pd.read_csv(file_path)
    
    else:
        raise ValueError("Invalid file type. Only PDF and CSV files are supported.")
    
    # Handle repeated headers
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.astype(object)
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    
    # Drop empty rows
    df = df.dropna(how='all')
    
    # Replace NaN in numeric columns with 0.0
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(0.0)
    
    # Replace NaN in string columns with ""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("")
    
    return df