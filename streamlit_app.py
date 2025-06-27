import patch
import streamlit as st
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv
from karo.prompts.system_prompt_builder import SystemPromptBuilder
from rich.console import Console
from pydantic import Field
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

from karo.core.base_agent import BaseAgent, BaseAgentConfig
from karo.providers.openai_provider import OpenAIProvider, OpenAIProviderConfig
from karo.providers.anthropic_provider import AnthropicProvider, AnthropicProviderConfig
from karo.schemas.base_schemas import BaseInputSchema, BaseOutputSchema, AgentErrorSchema
from excel_tool_reader import ExcelReaderInput, ExcelReaderOutput, ExcelReader

console = Console()

st.set_page_config(page_title="Excel Reader Tool", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.title("Excel Reader Tool")
st.markdown("This tool reads data from an Excel file and returns a preview of the data.")

class SummarizationOutput(BaseOutputSchema):
    summary: str = Field(..., description="The summary of the data read from the Excel file.")
    key_takeaways: List[str] = Field(default_factory=list, description="Key takeaways from the data.")

with st.sidebar:
    st.header("About Karo Framework")
    st.info("Karo is a framework for building and deploying AI agents.")

    st.header("Settings")
    provider_type = st.selectbox("Provider", ["OpenAI"], help="Select the provider for the agent.")

    if provider_type == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key.")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        model_choice = st.selectbox("AI Model", ["gpt-4-turbo", "gpt-4"], help="Select the model for summarization.")
    
    else:
        api_key = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key for summarization.")

        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        model_choice = st.selectbox("AI Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229"], help="Select the model for summarization.")

    max_rows = st.slider("Max Rows to process", 10, 50, 100, help="Limit the number of rows to process (higher = more complete but slower)")

    max_cols = st.slider("Max Columns to process", 1, 20, 5, help="Limit the number of columns to process")

    show_debug = st.checkbox("Show Debug Info", value=False, help="Show system prompts and messages being sent to the LLM")

def run_summarization(file_path, max_rows, max_cols, provider_type="OpenAI", model_choice="gpt-4-turbo", show_debug=False):
    """Function to summarize the data read from the Excel file."""
    if provider_type == "OpenAI":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key is not set. Please enter your API key in the sidebar.")
            return None
        
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Anthropic API key is not set. Please enter your API key in the sidebar.")
            return None
            
    with st.spinner("Initializing Excel Reader Tool..."):
        excel_reader_tool = ExcelReader()
        st.success("Excel Reader Tool initialized successfully.")

    with st.spinner("Reading Excel file..."):
        tool_input = ExcelReaderInput(file_path=file_path, max_rows=max_rows, max_cols=max_cols)
        tool_output = excel_reader_tool.run(tool_input)

    if not tool_output.success:
        st.error(f"Error reading Excel file: {tool_output.error_message}")
        return None
    
    if not tool_output.data_preview:
        st.warning("No data found in the Excel file.")
        return None
    
    st.success(f"Successfully read sheet '{tool_output.sheet_name_read}' {tool_output.row_count} rows, {len(tool_output.column_names)} columns from the Excel file.")
    st.markdown("### Data Preview")
    st.markdown(tool_output.data_preview)

    with st.spinner(f"Initializing {provider_type} provider..."):
        if provider_type == "OpenAI":
            provider_config = OpenAIProviderConfig(model=model_choice)
            provider = OpenAIProvider(config=provider_config)
        else:
            provider_config = AnthropicProviderConfig(model=model_choice)
            provider = AnthropicProvider(config=provider_config)

        st.success(f"{provider_type} provider initialized (Model: {model_choice} successfully.")

    system_prompt_content = "You are an expert data analyst. Your task is to analyze Excel data and provide accurate summaries and key takeaways."

    system_prompt_builder = SystemPromptBuilder(role_description=system_prompt_content)

    data_message = (
        "Please analyze this Excel data:\n\n"
        "Data preview:\n"
        "```markdown\n"
        f"{tool_output.data_preview}\n"
        "```\n\n"
        f"Column names: {', '.join(tool_output.column_names)}\n"
        f"Sheet name: {tool_output.sheet_name_read}\n"
        f"(Note: Only the first {tool_output.row_count} rows are shown in the preview\n\n"
        "Generate a summary that specifically analyzes the data patterns. "
        "Include insights about key metrics and patterns visible in the data. "
    )

    if show_debug:
        st.markdown("### debug information")
        st.markdown("#### System Prompt")
        st.code(system_prompt_content)
        st.markdown("#### User Message")
        st.code(data_message)

    with st.spinner("Configuring Agent..."):
        agent_config = BaseAgentConfig(
            provider_config=provider_config,
            system_prompt=system_prompt_builder,
            output_schema=SummarizationOutput,
        )

        summarization_agent = BaseAgent(config=agent_config)
        st.success("Agent configured successfully.")

    external_history = [ {"role": "user", "content": data_message }]

    with st.spinner(f"Generating summary using {provider_type} Agent..."):
        simple_input = BaseInputSchema(chat_message="")

        result = summarization_agent.run(
            input_data=simple_input,
            history=external_history,
        )

    return result

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], help="Upload an Excel file to read data from.")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        tmp_filepath = temp_file.name

    try:
        st.success(f"File uploaded: {uploaded_file.name}")

        api_var = "OPENAI_API_KEY" if provider_type == "OpenAI" else "ANTHROPIC_API_KEY"
        if api_var not in os.environ or not os.environ[api_var]:
            st.warning(f"{provider_type} API key is not set. Please enter your API key in the sidebar.")

        else:
            if st.button("Generate Summary"):
                result = run_summarization(tmp_filepath, max_rows, max_cols, provider_type, model_choice, show_debug)

                if result is None:
                    pass

                elif isinstance(result, SummarizationOutput):
                    st.subheader("Summary")
                    st.write(result.summary)

                    st.subheader("Key Takeaways")
                    for i, takeaway in enumerate(result.key_takeaways, start=1):
                        st.markdown(f"**{i}.** {takeaway}")
                elif isinstance(result, AgentErrorSchema):
                    st.error(f"Error from Karo Agent: {result.error_message}")
                else:
                    st.warning(f"Unexpected result type from agent: {type(result)}")

    finally:
        if os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)
else:

    st.info("Please upload an Excel file to read data from.")

    st.subheader("Agent summary will appear here")
    st.text("Upload a file and click 'Generate Summary' to analyze your data.")
