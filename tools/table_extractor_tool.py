from smolagents import Tool
import pandas as pd
from typing import Optional
import os

class TableExtractorTool(Tool):
    """
    Extracts tables from Excel (.xlsx, .xls) or CSV files and answers queries.
    Auto-detects file type based on extension.
    """
    name = "table_extractor"
    description = "Reads Excel/CSV files and answers questions about tabular data"
    inputs = {
        "file_path": {
            "type": "string", 
            "description": "Path to Excel/CSV file"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name (Excel only, optional)",
            "required": False,
            "nullable": True
        },
        "query": {
            "type": "string",
            "description": "Question about the data (e.g., 'total sales')",
            "required": False,
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, 
               file_path: str,
               sheet_name: Optional[str] = None,
               query: Optional[str] = None) -> str:
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"
            
            # Read file based on extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ('.xlsx', '.xls'):
                df = self._read_excel(file_path, sheet_name)
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                return f"Error: Unsupported file type {ext}"
            
            if df.empty:
                return "Error: No data found in file."
            
            return self._answer_query(df, query) if query else df.to_string()
            
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def _read_excel(self, path: str, sheet_name: Optional[str]) -> pd.DataFrame:
        """Read Excel file with sheet selection logic"""
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name)
        
        # Auto-detect first non-empty sheet
        sheets = pd.ExcelFile(path).sheet_names
        for sheet in sheets:
            df = pd.read_excel(path, sheet_name=sheet)
            if not df.empty:
                return df
        return pd.DataFrame()  # Return empty if all sheets are blank

    def _answer_query(self, df: pd.DataFrame, query: str) -> str:
        """Handles queries with pandas operations"""
        query = query.lower()
        
        try:
            # SUM QUERIES (e.g., "total revenue")
            if "total" in query or "sum" in query:
                for col in df.select_dtypes(include='number').columns:
                    if col.lower() in query:
                        return f"Total {col}: {df[col].sum():.2f}"
            
            # AVERAGE QUERIES (e.g., "average price")
            elif "average" in query or "mean" in query:
                for col in df.select_dtypes(include='number').columns:
                    if col.lower() in query:
                        return f"Average {col}: {df[col].mean():.2f}"
            
            # FILTER QUERIES (e.g., "show sales > 1000")
            elif ">" in query or "<" in query:
                col = next((c for c in df.columns if c.lower() in query), None)
                if col:
                    filtered = df.query(query.replace(col, f"`{col}`"))
                    return filtered.to_string()
            
            # DEFAULT: Return full table with column names
            return f"Data:\nColumns: {', '.join(df.columns)}\n\n{df.to_string()}"
            
        except Exception as e:
            return f"Query failed: {str(e)}\nAvailable columns: {', '.join(df.columns)}"