from smolagents import Tool
from tabula import read_pdf
import pandas as pd
from typing import Optional, Dict, Any

class TableExtractorTool(Tool):
    """
    Tool to extract tables from PDFs/webpages and answer queries about them.

    Args:
        file_path (str): Path to PDF file (optional)
        url (str): URL of webpage containing tables (optional)
        query (str): Natural language question about the table data (optional)

    Returns:
        str: Extracted table data or answer to query
    """

    name = "extract_table"
    description = "Extracts tables from PDFs or webpages and answers questions about the data"
    
    inputs = {
        "file_path": {
            "type": "string", 
            "description": "Path to PDF file (either file_path or url required)",
            "required": False
        },
        "url": {
            "type": "string",
            "description": "URL of webpage containing tables (either file_path or url required)",
            "required": False
        },
        "query": {
            "type": "string",
            "description": "Natural language question about the table data",
            "required": False
        }
    }
    
    output_type = "string"

    def forward(self, file_path: Optional[str] = None, 
               url: Optional[str] = None, 
               query: Optional[str] = None) -> str:
        
        # Validate input
        if not file_path and not url:
            return "Error: Either file_path or url must be provided"
        
        try:
            # Case 1: Extract from PDF
            if file_path and file_path.endswith(".pdf"):
                tables = read_pdf(file_path, pages="all", multiple_tables=True)
                df = pd.concat(tables) if tables else None
            
            # Case 2: Extract from HTML (webpage)
            elif url:
                dfs = pd.read_html(url)
                df = dfs[0] if dfs else None
            
            if df is None:
                return "No tables found in the input source"
            
            # Answer query if provided
            if query:
                return self._answer_query(df, query)
            return df.to_string()
            
        except Exception as e:
            return f"Error processing table data: {str(e)}"

    def _answer_query(self, df: pd.DataFrame, query: str) -> str:
        """Helper method to answer questions about the table data"""
        try:
            query = query.lower()
            
            # Example simple queries - you could expand this or integrate an LLM
            if "total" in query and "sum" in query:
                if "revenue" in query:
                    col = "Revenue"
                elif "sales" in query:
                    col = "Sales"
                else:
                    # Try to find a numeric column
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if col:
                    return f"Total {col}: {df[col].sum()}"
            
            elif "average" in query or "mean" in query:
                # Find the most likely column referenced in query
                for col in df.columns:
                    if col.lower() in query:
                        return f"Average {col}: {df[col].mean():.2f}"
                
                # Default to first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    return f"Average {numeric_cols[0]}: {df[numeric_cols[0]].mean():.2f}"
            
            # Fallback: return the table
            return f"Here's the table data:\n{df.to_string()}\n\nQuery '{query}' not fully understood."
            
        except Exception as e:
            return f"Error answering query: {str(e)}\nTable data:\n{df.to_string()}"