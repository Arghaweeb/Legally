import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import json
import re
from tavily import TavilyClient

# Trusted Indian legal domains
LEGAL_SOURCES = [
    "indiankanoon.org",
    "sci.gov.in",           # Supreme Court of India
    "judgments.ecourts.gov.in",  # District Courts
    "hcservices.ecourts.gov.in", # High Courts
    "lawmin.gov.in",        # Ministry of Law
    "legislative.gov.in",   # Legislative Department
    "legalserviceindia.com",
    "casemine.com",
    "manupatra.com",
    "scconline.com"
]

def _is_legal_source(url: str) -> bool:
    """Check if a URL belongs to one of the trusted legal domains."""
    if not url:
        return False
    return any(domain in url.lower() for domain in LEGAL_SOURCES)


def _extract_case_citation(text: str) -> Optional[str]:
    """Extract case citation from text."""
    # Common citation patterns
    patterns = [
        r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',  # (2024) 5 SCC 123
        r'AIR\s*\d{4}\s*SC\s*\d+',        # AIR 2024 SC 123
        r'\d{4}\s*\(\d+\)\s*\w+\s*\d+',   # 2024 (5) ALT 123
        r'W\.P\.\s*No\.\s*\d+\s*of\s*\d{4}',  # W.P. No. 123 of 2024
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group()
    return None


@tool("Legal Precedent Search Tool")
def search_legal_precedents(
    query: str, 
    ipc_sections: Optional[str] = None,
    max_results: int = 5
) -> str:
    """
    Search for relevant legal precedents and case laws from Indian courts.
    Uses Tavily Search API to find precedent cases from trusted legal sources.
    
    Args:
        query (str): The legal issue or case description to search for precedents
        ipc_sections (str, optional): Comma-separated IPC sections to include in search
        max_results (int): Maximum number of precedents to return (default: 5)
    
    Returns:
        str: Formatted string with relevant case precedents including case names,
             citations, summaries, and links to full judgments
    
    Example:
        search_legal_precedents("murder due to sudden provocation", "302,300")
    """
    load_dotenv()
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "❌ Error: 'TAVILY_API_KEY' not found in .env file"
    
    try:
        client = TavilyClient(api_key=api_key)
        
        # Build comprehensive search query
        search_terms = [query]
        
        # Add IPC sections if provided
        if ipc_sections:
            sections = ipc_sections.replace(",", " OR IPC ")
            search_terms.append(f"IPC {sections}")
        
        # Add legal keywords for better results
        search_terms.append("judgment precedent case law India")
        
        # Construct the full query
        full_query = " ".join(search_terms)
        
        # Search with domain restrictions
        site_query = " OR ".join([f"site:{domain}" for domain in LEGAL_SOURCES[:3]])
        search_query = f"({site_query}) {full_query}"
        
        # Perform search
        response = client.search(
            query=search_query,
            max_results=max_results * 2,  # Get extra results for filtering
            search_depth="advanced",
            include_domains=LEGAL_SOURCES
        )
        
        raw_results = response.get("results", [])
        
        # Filter and process results
        legal_results = []
        seen_titles = set()
        
        for item in raw_results:
            if not _is_legal_source(item.get("url", "")):
                continue
            
            title = item.get("title", "")
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Extract case citation if present
            citation = _extract_case_citation(title) or _extract_case_citation(item.get("content", ""))
            
            legal_results.append({
                "title": title,
                "citation": citation,
                "summary": item.get("content", "")[:500],  # Limit summary length
                "url": item.get("url"),
                "source": next((domain for domain in LEGAL_SOURCES if domain in item.get("url", "")), "Unknown")
            })
            
            if len(legal_results) >= max_results:
                break
        
        # Format results
        if not legal_results:
            return "No relevant legal precedents found from trusted Indian legal sources."
        
        formatted_results = []
        for i, result in enumerate(legal_results, 1):
            formatted = f"""
**Precedent {i}:**
⚖️ **Case**: {result['title']}
📑 **Citation**: {result['citation'] or 'Citation not found'}
🏛️ **Source**: {result['source']}

**Summary**: {result['summary']}

🔗 **Full Judgment**: {result['url']}
---"""
            formatted_results.append(formatted)
        
        header = f"Found {len(legal_results)} relevant legal precedent(s):\n"
        return header + "\n".join(formatted_results)
        
    except Exception as e:
        return f"❌ Error searching legal precedents: {str(e)}"


# Enhanced Precedent Searcher with additional features
class LegalPrecedentSearcher:
    """
    Advanced legal precedent searcher with caching and categorization.
    """
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in .env file")
        self.client = TavilyClient(api_key=self.api_key)
        self.cache = {}
    
    def search_by_court(
        self, 
        query: str, 
        court_level: str = "all",
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search precedents filtered by court level.
        
        Args:
            query: Search query
            court_level: 'supreme', 'high', 'district', or 'all'
            max_results: Maximum results to return
        """
        court_domains = {
            "supreme": ["sci.gov.in", "indiankanoon.org/search/?formInput=doctypes:sc"],
            "high": ["hcservices.ecourts.gov.in", "indiankanoon.org/search/?formInput=doctypes:hc"],
            "district": ["judgments.ecourts.gov.in"],
            "all": LEGAL_SOURCES
        }
        
        domains = court_domains.get(court_level, LEGAL_SOURCES)
        
        # Build domain-specific query
        site_query = " OR ".join([f"site:{domain}" for domain in domains[:3]])
        search_query = f"({site_query}) {query} judgment India"
        
        response = self.client.search(
            query=search_query,
            max_results=max_results * 2,
            search_depth="advanced"
        )
        
        results = []
        for item in response.get("results", []):
            if _is_legal_source(item.get("url", "")):
                results.append({
                    "title": item.get("title"),
                    "summary": item.get("content"),
                    "url": item.get("url"),
                    "court_level": court_level,
                    "relevance_score": item.get("score", 0)
                })
        
        return results[:max_results]
    
    def search_similar_cases(
        self,
        case_facts: str,
        ipc_sections: List[str],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases based on facts and IPC sections.
        """
        # Create a comprehensive query
        query_parts = [case_facts]
        
        if ipc_sections:
            sections_str = " ".join([f"IPC-{s}" for s in ipc_sections])
            query_parts.append(sections_str)
        
        query = " ".join(query_parts) + " similar cases precedent India"
        
        return self.search_by_court(query, "all", max_results)


# Testing functions (commented out for production)
"""
# Test IPC Search
if __name__ == "__main__":
    # Test IPC search
    print("Testing IPC Search Tool...")
    result = search_ipc_sections("theft of property")
    print(result)
    
    # Test Precedent search
    print("\nTesting Legal Precedent Search Tool...")
    result = search_legal_precedents("murder with sudden provocation", "302,304")
    print(result)
"""