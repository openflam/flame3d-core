import os

from server.search.llm_reasoning.tools import get_tool_descriptions

_current_dir = os.path.dirname(os.path.abspath(__file__))
_prompt_file_path = os.path.join(_current_dir, "tools_prompt.md")

with open(_prompt_file_path, "r", encoding="utf-8") as _f:
    TOOL_CALLING_SYSTEM_TEMPLATE = _f.read()

def get_tools_prompt(allowed_tools=None):
    descriptions = get_tool_descriptions(allowed_tools)
    
    strategy_prompt = ""
    for tool_name, desc in descriptions.items():
        strategy_prompt += desc + "\n\n"
        
    return TOOL_CALLING_SYSTEM_TEMPLATE.replace("{tool_descriptions}", strategy_prompt.strip())


