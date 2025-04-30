"""
Utilities for handling code-related queries and responses.
This module provides functionality for detecting, parsing, and generating code-related content.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK resources: {e}")

# Regular expressions for code detection
CODE_BLOCK_PATTERN = r"```(?:\w+)?\n([\s\S]*?)\n```"
INLINE_CODE_PATTERN = r"`([^`]+)`"
FUNCTION_DEF_PATTERNS = {
    'python': r"def\s+\w+\s*\(.*?\):",
    'javascript': r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:function|\(.*?\)\s*=>))",
    'typescript': r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:function|\(.*?\)\s*=>))",
    'java': r"\b(?:public|private|protected|static)?\s+\w+\s+\w+\s*\(.*?\)\s*\{",
    'c': r"\w+\s+\w+\s*\(.*?\)\s*\{",
    'cpp': r"\w+(?:\s+\w+)+\s*\(.*?\)\s*(?:const)?\s*\{",
    'csharp': r"(?:public|private|protected|static)?\s+\w+\s+\w+\s*\(.*?\)\s*\{",
    'go': r"func\s+\w+\s*\(.*?\)\s*(?:\w+)?\s*\{",
    'ruby': r"def\s+\w+(?:\(.*?\))?\s*$",
    'php': r"function\s+\w+\s*\(.*?\)\s*\{",
    'rust': r"fn\s+\w+\s*\(.*?\)(?:\s*->\s*\w+)?\s*\{",
}
CLASS_DEF_PATTERNS = {
    'python': r"class\s+\w+(?:\(.*?\))?:",
    'javascript': r"class\s+\w+(?:\s+extends\s+\w+)?",
    'typescript': r"class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?",
    'java': r"(?:public|private|protected)?\s+class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?",
    'cpp': r"class\s+\w+(?:\s*:\s*(?:public|private|protected)\s+\w+)?",
    'csharp': r"(?:public|private|protected|internal)?\s+class\s+\w+",
    'php': r"class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?",
    'ruby': r"class\s+\w+(?:\s*<\s*\w+)?",
    'rust': r"struct\s+\w+|enum\s+\w+|trait\s+\w+",
}
IMPORT_PATTERNS = {
    'python': r"(?:import|from)\s+\w+",
    'javascript': r"(?:import|require)\s*\(?\s*['\"]",
    'typescript': r"(?:import|require)\s*\(?\s*['\"]",
    'java': r"import\s+[\w\.]+",
    'go': r"import\s+[\"(]",
    'rust': r"use\s+[\w:]+",
    'ruby': r"require\s+['\"]",
    'php': r"(?:require|include)(?:_once)?\s*\(?\s*['\"]",
}

def contains_code(text: str) -> bool:
    """
    Detect if text contains code blocks or patterns.
    
    Args:
        text: The text to analyze
        
    Returns:
        bool: True if the text likely contains code
    """
    # Check for markdown code blocks
    if re.search(CODE_BLOCK_PATTERN, text):
        return True
    
    # Check for inline code
    inline_code_matches = re.findall(INLINE_CODE_PATTERN, text)
    if len(inline_code_matches) > 2:  # Multiple inline code segments suggest code discussion
        return True
    
    # Check for common code patterns
    for lang, pattern in FUNCTION_DEF_PATTERNS.items():
        if re.search(pattern, text):
            return True
    
    for lang, pattern in CLASS_DEF_PATTERNS.items():
        if re.search(pattern, text):
            return True
    
    for lang, pattern in IMPORT_PATTERNS.items():
        if re.search(pattern, text):
            return True
    
    # Check for high symbol-to-text ratio (code often has more symbols)
    symbol_count = len(re.findall(r'[{}\[\]()<>:;.,=+\-*/&|^%!]', text))
    text_length = len(text)
    
    if text_length > 0 and symbol_count / text_length > 0.1:  # More than 10% symbols is suspicious
        return True
    
    # Check for indentation patterns typical in code
    lines = text.split('\n')
    indent_pattern = re.compile(r'^( {2,}|\t+)')
    indented_lines = sum(1 for line in lines if indent_pattern.match(line))
    
    if len(lines) > 3 and indented_lines / len(lines) > 0.3:  # 30% indented lines suggests code
        return True
    
    return False

def extract_code_blocks(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract code blocks from text.
    
    Args:
        text: The text to extract code blocks from
        
    Returns:
        List of tuples (code_content, language)
    """
    blocks = []
    
    # Extract markdown code blocks with language specification
    for match in re.finditer(r"```(\w+)?\n([\s\S]*?)\n```", text):
        language = match.group(1)
        code = match.group(2)
        blocks.append((code, language))
    
    # Extract markdown code blocks without language specification
    for match in re.finditer(r"```\n([\s\S]*?)\n```", text):
        code = match.group(1)
        # Try to guess the language
        language = detect_language(code)
        blocks.append((code, language))
    
    # If no code blocks, check if the entire text might be code
    if not blocks and contains_code(text):
        language = detect_language(text)
        blocks.append((text, language))
    
    return blocks

def detect_language(code: str) -> Optional[str]:
    """
    Detect the programming language of a code snippet.
    
    Args:
        code: The code to analyze
        
    Returns:
        The detected language or None if detection failed
    """
    try:
        lexer = guess_lexer(code)
        return lexer.name.lower()
    except ClassNotFound:
        # Try to detect based on patterns
        for lang, pattern in FUNCTION_DEF_PATTERNS.items():
            if re.search(pattern, code):
                return lang
        
        for lang, pattern in CLASS_DEF_PATTERNS.items():
            if re.search(pattern, code):
                return lang
                
        for lang, pattern in IMPORT_PATTERNS.items():
            if re.search(pattern, code):
                return lang
        
        return None

def format_code_for_response(code: str, language: Optional[str] = None) -> str:
    """
    Format code for inclusion in a response.
    
    Args:
        code: The code to format
        language: The programming language
        
    Returns:
        Formatted code as markdown
    """
    if not language:
        language = detect_language(code) or ""
    
    # Ensure the code is properly indented and trimmed
    code = code.strip()
    
    # Return markdown-formatted code block
    return f"```{language}\n{code}\n```"

def is_code_question(query: str) -> bool:
    """
    Determine if a query is about code.
    
    Args:
        query: The user's query
        
    Returns:
        True if the query is likely about code
    """
    # Check for code blocks
    if contains_code(query):
        return True
    
    # Check for programming-related terms
    code_terms = [
        'code', 'function', 'class', 'method', 'variable', 'compile', 'error',
        'debug', 'exception', 'syntax', 'implement', 'algorithm', 'programming',
        'javascript', 'python', 'java', 'c#', 'typescript', 'html', 'css', 'sql',
        'database', 'api', 'framework', 'library', 'npm', 'pip', 'git', 'docker'
    ]
    
    query_lower = query.lower()
    words = word_tokenize(query_lower)
    
    code_term_count = sum(1 for term in code_terms if term in words or term in query_lower)
    
    # If more than 2 code terms, likely a code question
    return code_term_count >= 2

def categorize_code_question(query: str) -> Dict[str, any]:
    """
    Categorize a code-related question to better tailor the response.
    
    Args:
        query: The user's query
        
    Returns:
        Dictionary with category information
    """
    query_lower = query.lower()
    
    categories = {
        'debug': ['error', 'bug', 'fix', 'issue', 'problem', 'debug', 'not working', 'fails'],
        'explain': ['explain', 'how does', 'what is', 'understand', 'mean', 'purpose'],
        'implement': ['implement', 'create', 'write', 'develop', 'build', 'make', 'code'],
        'optimize': ['optimize', 'improve', 'better', 'efficient', 'performance', 'faster'],
        'compare': ['versus', 'vs', 'compare', 'difference', 'better choice', 'prefer'],
    }
    
    # Check for code blocks - if present, likely asking about specific code
    has_code = contains_code(query)
    
    # Default category
    result = {
        'primary_category': 'general',
        'has_code': has_code,
        'language': None,
        'categories': []
    }
    
    # Extract code to detect language
    if has_code:
        code_blocks = extract_code_blocks(query)
        if code_blocks:
            result['language'] = code_blocks[0][1]
    
    # Detect categories
    matched_categories = []
    for category, terms in categories.items():
        for term in terms:
            if term in query_lower:
                matched_categories.append(category)
                break
    
    if matched_categories:
        result['categories'] = matched_categories
        result['primary_category'] = matched_categories[0]
    
    return result

def generate_code_response(query: str, context: str = "") -> Tuple[str, str]:
    """
    Generate a structured response for code-related questions.
    This is a placeholder function that should be replaced with actual
    ML model integration.
    
    Args:
        query: The user's question
        context: Additional context (e.g., from web search)
        
    Returns:
        Tuple of (answer, detected_language)
    """
    # This should be replaced with actual model implementation
    categorization = categorize_code_question(query)
    language = categorization.get('language', 'unknown')
    
    # In a real implementation, you would:
    # 1. Process the query with the ML model
    # 2. Format the code appropriately
    # 3. Return the response
    
    # For now, return a placeholder
    return (
        "I would need to process this with the ML model to give you a proper code answer.",
        language
    )
