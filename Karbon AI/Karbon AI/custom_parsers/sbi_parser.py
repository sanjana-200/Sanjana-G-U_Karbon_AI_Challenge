import pandas as pd
import pdfplumber
import re

def parse(file_path: str) -> pd.DataFrame:
    """
    Parse SBI bank PDF/CSV file and return a DataFrame.
    
    Parameters:
    file_path (str): Path to the PDF/CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the parsed data.
    """
    
    # Check if file is a PDF or CSV
    if file_path.endswith('.pdf'):
        # Extract text from PDF
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        
        # Split text into lines
        lines = text.split('\n')
        
        # Initialize lists to store data
        headers = []
        data = []
        
        # Initialize flag to track if we're in the data section
        in_data = False
        
        # Iterate over lines
        for line in lines:
            # Remove leading and trailing whitespace
            line = line.strip()
            
            # Check if line is a header
            if line and line[0].isupper():
                headers.append(line)
                in_data = False
            # Check if line is data
            elif line and not line.isspace():
                data.append(line)
                in_data = True
            
            # If we're in the data section and we've seen a header, break
            if in_data and headers:
                break
        
        # Create DataFrame
        df = pd.DataFrame([line.split('\t') for line in data], columns=headers)
        
    elif file_path.endswith('.csv'):
        # Read CSV directly into DataFrame
        df = pd.read_csv(file_path)
    
    else:
        raise ValueError("Unsupported file type. Only PDF and CSV are supported.")
    
    # Drop empty rows
    df = df.dropna(how='all')
    
    # Replace NaN in numeric columns with 0.0
    df = df.apply(lambda x: x.fillna(0.0) if x.dtype in ['int64', 'float64'] else x)
    
    # Replace NaN in non-numeric columns with ""
    df = df.apply(lambda x: x.fillna("") if x.dtype not in ['int64', 'float64'] else x)
    
    # Clean numeric columns
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df