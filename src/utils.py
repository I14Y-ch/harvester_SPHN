
from datetime import datetime 
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, List, Optional, Union

def remove_html_tags(text: str) -> str:
    """Remove HTML tags using BeautifulSoup."""
    return BeautifulSoup(text, "html.parser").get_text()

def normalize_uri(uri: str) -> str:
    """Normalize URI by stripping whitespace and trailing slashes."""
    return uri.strip().rstrip('/') if uri else uri

def is_valid_uri(uri: str) -> bool:
    """Check if the string looks like a valid URI."""
    try:
        result = urlparse(normalize_uri(uri))
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def format_date(value_str: str) -> Optional[str]:
    """Helper to format date strings consistently."""
    if not value_str:
        return None
    try:
        if len(value_str) == 10:
            return datetime.strptime(value_str, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
        return datetime.fromisoformat(value_str.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return value_str

def remove_empty_fields(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Recursively remove empty lists, empty dicts, and None values from a dictionary.
    """
    if isinstance(data, dict):
        return {
            k: remove_empty_fields(v)
            for k, v in data.items()
            if v not in (None, [], {}, "") and remove_empty_fields(v) not in (None, [], {})
        }
    elif isinstance(data, list):
        return [remove_empty_fields(v) for v in data if remove_empty_fields(v) not in (None, [], {})]
    return data
