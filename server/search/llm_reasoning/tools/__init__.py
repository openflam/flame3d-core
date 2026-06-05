"""Tools module."""

from .bm25_search import search_terms, SEARCH_TERMS_TOOL, SEARCH_THINKING_TEXT, SEARCH_TERMS_DESCRIPTION
from .distance import get_distance, GET_DISTANCE_TOOL, GET_DISTANCE_THINKING_TEXT, GET_DISTANCE_DESCRIPTION
from .search_around import search_around_component, SEARCH_AROUND_COMPONENT_TOOL, SEARCH_AROUND_COMPONENT_THINKING_TEXT, SEARCH_AROUND_COMPONENT_DESCRIPTION
from .component_info import get_component_info, GET_COMPONENT_INFO_TOOL, GET_COMPONENT_INFO_THINKING_TEXT, GET_COMPONENT_INFO_DESCRIPTION
from .image import get_images, GET_IMAGES_TOOL, GET_IMAGES_THINKING_TEXT, GET_IMAGES_DESCRIPTION
from .execute_python import execute_python, EXECUTE_PYTHON_TOOL, EXECUTE_PYTHON_THINKING_TEXT, EXECUTE_PYTHON_DESCRIPTION

_TOOLS_LIST = [SEARCH_TERMS_TOOL, GET_DISTANCE_TOOL, SEARCH_AROUND_COMPONENT_TOOL, GET_COMPONENT_INFO_TOOL, GET_IMAGES_TOOL, EXECUTE_PYTHON_TOOL]

_TOOL_FUNCTIONS = {
    "search_terms": search_terms,
    "get_distance": get_distance,
    "search_around_component": search_around_component,
    "get_component_info": get_component_info,
    "get_images": get_images,
    "execute_python": execute_python,
}

_THINKING_TEXTS = {
    "search_terms": SEARCH_THINKING_TEXT,
    "get_distance": GET_DISTANCE_THINKING_TEXT,
    "search_around_component": SEARCH_AROUND_COMPONENT_THINKING_TEXT,
    "get_component_info": GET_COMPONENT_INFO_THINKING_TEXT,
    "get_images": GET_IMAGES_THINKING_TEXT,
    "execute_python": EXECUTE_PYTHON_THINKING_TEXT,
}

_TOOL_DESCRIPTIONS = {
    "search_terms": SEARCH_TERMS_DESCRIPTION,
    "get_distance": GET_DISTANCE_DESCRIPTION,
    "search_around_component": SEARCH_AROUND_COMPONENT_DESCRIPTION,
    "get_component_info": GET_COMPONENT_INFO_DESCRIPTION,
    "get_images": GET_IMAGES_DESCRIPTION,
    "execute_python": EXECUTE_PYTHON_DESCRIPTION,
}

def get_tools(allowed_tools=None):
    if allowed_tools is None:
        return _TOOLS_LIST
    return [t for t in _TOOLS_LIST if t["name"] in allowed_tools]

def get_tool_functions(allowed_tools=None):
    if allowed_tools is None:
        return _TOOL_FUNCTIONS
    return {k: v for k, v in _TOOL_FUNCTIONS.items() if k in allowed_tools}

def get_thinking_texts(allowed_tools=None):
    if allowed_tools is None:
        return _THINKING_TEXTS
    return {k: v for k, v in _THINKING_TEXTS.items() if k in allowed_tools}

def get_tool_descriptions(allowed_tools=None):
    if allowed_tools is None:
        return _TOOL_DESCRIPTIONS
    return {k: v for k, v in _TOOL_DESCRIPTIONS.items() if k in allowed_tools}
