import pandas as pd
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, FilePath

from karo.tools.base_tool import BaseTool, BaseToolInputSchema, BaseToolOutputSchema

logger = logging.getLogger(__name__)

class ExcelReaderInput(BaseToolInputSchema):
    """Input schema for the ExcelReader tool."""
    file_path: FilePath = Field(..., description="Path to the Excel file.")
    sheet_name: Optional[Union[str, int]] = Field(None, description="Specific name of sheet")
    max_rows: Optional[int] = Field(100, description="Maximum number of rows to read from each sheet.")
    max_cols: Optional[int] = Field(20, description="Maximum number of columns to read from each sheet.")

class ExcelReaderOutput(BaseToolOutputSchema):
    """Output schema for the ExcelReader tool."""
    file_path: str = Field(..., description="The path of file that was read")
    sheet_name_read: str = Field(..., description="The name of the sheet that was actually read")
    data_preview: Optional[str] = Field(None, description="A string representation of the first few rows/columns of the data")
    row_count: Optional[int] = Field(None, description="The total number of rows read")
    column_names: Optional[List[str]] = Field(None, description="List of column names read (up to max_cols)")

class ExcelReader(BaseTool):
    """Tool to read data from an Excel file."""
    name = "excel_reader"
    description: str = "Reads data from an Excel file and returns a preview of the data."
    input_schema = ExcelReaderInput
    output_schema = ExcelReaderOutput

    def __init__(self, config: Optional[Any] = None):
        """Initialize the ExcelReader tool."""
        logger.info("ExcelReaderTool initialized.")
        pass

    def run(self, input_data: ExcelReaderInput) -> ExcelReaderOutput:
        """Reads the specificied Excel file and returns a data preview."""

        # Read the specified sheet from the Excel file
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl is not installed. Please install it to read Excel files.")
            return self.output_schema(success=False, error_message="openpyxl is not installed.", file_path=str(input_data.file_path), sheet_name_read="N/A")

        if not isinstance(input_data, self.input_schema):
            return self.output_schema(success=False, error_message="Invalid input data format.", file_path=str(input_data.file_path), sheet_name_read="N/A")
        
        file_path_str = str(input_data.file_path)

        if not os.path.exists(file_path_str):
            return self.output_schema(success=False, error_message=f"File not found: {file_path_str}", file_path=file_path_str, sheet_name_read="N/A")
        
        try:
            excel_file = pd.ExcelFile(file_path_str, engine = "openpyxl")
            sheet_names = excel_file.sheet_names
            sheet_to_read: Union[str, int] = 0 
            sheet_name_read: str = sheet_names[0]

            if input_data.sheet_name is not None:
                if isinstance(input_data.sheet_name, int):
                    if 0 <= input_data.sheet_name < len(sheet_names):
                        sheet_to_read = input_data.sheet_name
                        sheet_name_read = sheet_names[sheet_to_read]
                    else:
                        return self.output_schema(success=False, error_message=f"Sheet index out of range: {input_data.sheet_name}", file_path=file_path_str, sheet_name_read="N/A")
                elif isinstance(input_data.sheet_name, str):
                    if input_data.sheet_name in sheet_names:
                        sheet_to_read = input_data.sheet_name
                        sheet_name_read = input_data.sheet_name
                    else:
                        return self.output_schema(success=False, error_message=f"Sheet name not found: {input_data.sheet_name}", file_path=file_path_str, sheet_name_read="N/A")
                    
            header_df = pd.read_excel(excel_file, sheet_name=sheet_to_read, nrows=0)
            all_columns = header_df.columns.tolist()
            cols_to_use = all_columns[:input_data.max_cols] if input_data.max_cols else all_columns

            df = pd.read_excel(excel_file, sheet_name=sheet_to_read, usecols=cols_to_use, nrows=input_data.max_rows)

            preview_rows = min(len(df), 10)
            data_preview_str = df.head(preview_rows).to_markdown(index=False)

            logger.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns from sheet '{sheet_name_read}' in '{file_path_str}'.")

            return self.output_schema(
                success=True,
                file_path=file_path_str,
                sheet_name_read=sheet_name_read,
                data_preview=data_preview_str,
                row_count=len(df),
                column_names=df.columns.tolist()
            )
        except FileNotFoundError:
            logger.error(f"File not found: {file_path_str}")
            return self.output_schema(success=False, error_message=f"File not found: {file_path_str}", file_path=file_path_str, sheet_name_read="N/A")
        except Exception as e:
            logger.error(f"Error reading Excel file '{file_path_str}': {e}", exc_info=True)
            return self.output_schema(success=False, error_message=f"Error reading Excel file: {e}", file_path=file_path_str, sheet_name_read="N/A")