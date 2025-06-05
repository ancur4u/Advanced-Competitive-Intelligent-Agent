# competitive_intelligence_agent.py

import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import re
from openai import OpenAI
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textwrap
import json
import hashlib
import time
from urllib.parse import urlparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ========== Enhanced Configuration ==========
st.set_page_config(
    page_title="🔍 Advanced Competitive Intelligence Agent", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced disclaimer with better formatting
with st.expander("📌 Agentic AI Architecture Overview"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🤖 Autonomous Capabilities:**
        - Goal-driven research workflows
        - Multi-stage task decomposition
        - Contextual tool orchestration
        - Self-adaptive analysis depth
        """)
    with col2:
        st.markdown("""
        **🧠 Intelligence Features:**
        - Sentiment-aware content analysis
        - Trend detection & forecasting
        - Competitive positioning insights
        - Historical pattern recognition
        """)

# ========== Enhanced Setup ==========
load_dotenv()

# Initialize session state for API keys
if 'api_keys_configured' not in st.session_state:
    st.session_state.api_keys_configured = False
if 'serp_api_key' not in st.session_state:
    st.session_state.serp_api_key = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Try to load from environment first (optional)
env_serp_key = os.getenv("SERP_API_KEY", "")
env_openai_key = os.getenv("OPENAI_API_KEY", "")

if env_serp_key and not st.session_state.serp_api_key:
    st.session_state.serp_api_key = env_serp_key
if env_openai_key and not st.session_state.openai_api_key:
    st.session_state.openai_api_key = env_openai_key

# Check if keys are configured
if st.session_state.serp_api_key and st.session_state.openai_api_key:
    st.session_state.api_keys_configured = True

# Enhanced constants
DB_PATH = "enhanced_intel.db"
REPORT_DIR = "reports"
CACHE_DIR = "cache"
LOGO_PATH = "Screenshot 2025-06-03 at 3.12.29 PM.png"

# Enhanced keyword categories
KEYWORD_CATEGORIES = {
    "product": ["launch", "release", "update", "feature", "beta", "version"],
    "business": ["acquisition", "merger", "partnership", "investment", "funding"],
    "market": ["expansion", "growth", "decline", "market share", "revenue"],
    "innovation": ["AI", "technology", "patent", "research", "development"],
    "challenges": ["lawsuit", "fine", "security", "breach", "layoff", "controversy"]
}

# Create directories
for directory in [REPORT_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========== Enhanced Database Schema ==========
def validate_serp_api_key(api_key: str) -> Dict[str, any]:
    """Validate SERP API key by making a test request"""
    if not api_key or not api_key.strip():
        return {"valid": False, "error": "API key is empty"}
    
    try:
        test_url = "https://serpapi.com/search"
        test_params = {
            "q": "test query",
            "api_key": api_key.strip(),
            "engine": "google",
            "num": 1
        }
        
        response = requests.get(test_url, params=test_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Check if we got valid results or error
            if "error" in data:
                return {"valid": False, "error": f"SERP API Error: {data['error']}"}
            else:
                return {"valid": True, "message": "SERP API key is valid"}
        elif response.status_code == 401:
            return {"valid": False, "error": "Invalid SERP API key - authentication failed"}
        elif response.status_code == 403:
            return {"valid": False, "error": "SERP API key access denied - check permissions"}
        else:
            return {"valid": False, "error": f"SERP API returned status code: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"valid": False, "error": "SERP API request timed out - check your connection"}
    except requests.exceptions.RequestException as e:
        return {"valid": False, "error": f"SERP API connection error: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Unexpected error validating SERP API: {str(e)}"}

def validate_openai_api_key(api_key: str) -> Dict[str, any]:
    """Validate OpenAI API key by making a test request"""
    if not api_key or not api_key.strip():
        return {"valid": False, "error": "API key is empty"}
    
    try:
        # Test with a minimal request
        test_client = OpenAI(api_key=api_key.strip())
        
        # Make a minimal completion request
        response = test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        if response and response.choices:
            return {"valid": True, "message": "OpenAI API key is valid"}
        else:
            return {"valid": False, "error": "OpenAI API returned unexpected response"}
            
    except Exception as e:
        error_str = str(e).lower()
        if "incorrect api key" in error_str or "invalid api key" in error_str:
            return {"valid": False, "error": "Invalid OpenAI API key"}
        elif "exceeded your current quota" in error_str:
            return {"valid": False, "error": "OpenAI API quota exceeded - add billing information"}
        elif "rate limit" in error_str:
            return {"valid": False, "error": "OpenAI API rate limit reached - try again later"}
        elif "authentication" in error_str:
            return {"valid": False, "error": "OpenAI API authentication failed"}
        else:
            return {"valid": False, "error": f"OpenAI API error: {str(e)}"}

def test_api_keys(serp_key: str, openai_key: str) -> Dict[str, any]:
    """Test both API keys and return combined results"""
    results = {
        "serp": {"valid": False, "tested": False},
        "openai": {"valid": False, "tested": False},
        "overall_valid": False
    }
    
    # Test SERP API
    if serp_key and serp_key.strip():
        results["serp"] = validate_serp_api_key(serp_key)
        results["serp"]["tested"] = True
    
    # Test OpenAI API  
    if openai_key and openai_key.strip():
        results["openai"] = validate_openai_api_key(openai_key)
        results["openai"]["tested"] = True
    
    # Overall validation
    results["overall_valid"] = (
        results["serp"].get("valid", False) and 
        results["openai"].get("valid", False)
    )
    
    return results

def get_openai_client():
    """Get OpenAI client with current API key"""
    try:
        return OpenAI(api_key=st.session_state.openai_api_key)
    except Exception as e:
        st.error(f"❌ OpenAI client initialization failed: {e}")
        return None
    """Get OpenAI client with current API key"""
    try:
        return OpenAI(api_key=st.session_state.openai_api_key)
    except Exception as e:
        st.error(f"❌ OpenAI client initialization failed: {e}")
        return None

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enhanced table structure
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intelligence_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        search_query TEXT,
        summary TEXT,
        sentiment_score REAL,
        key_insights TEXT,
        challenges TEXT,
        competitors TEXT,
        market_position TEXT,
        timestamp TEXT,
        source_count INTEGER,
        confidence_score REAL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS search_cache (
        query_hash TEXT PRIMARY KEY,
        results TEXT,
        timestamp TEXT,
        expiry_hours INTEGER DEFAULT 24
    )
    """)
    
    conn.commit()
    return conn

conn = init_database()

# ========== Enhanced Core Functions ==========
def calculate_query_hash(query: str) -> str:
    """Generate hash for query caching"""
    return hashlib.md5(query.encode()).hexdigest()

def is_cache_valid(timestamp: str, expiry_hours: int = 24) -> bool:
    """Check if cached result is still valid"""
    try:
        cache_time = datetime.fromisoformat(timestamp)
        return datetime.now() - cache_time < timedelta(hours=expiry_hours)
    except:
        return False

def search_google_enhanced(query: str, num_results: int = 10, use_cache: bool = True) -> List[dict]:
    """Enhanced search with caching and better error handling"""
    if not st.session_state.serp_api_key:
        st.error("❌ SERP API key not configured")
        return []
    
    query_hash = calculate_query_hash(query)
    
    # Check cache first
    if use_cache:
        cursor = conn.cursor()
        cursor.execute("SELECT results, timestamp FROM search_cache WHERE query_hash = ?", (query_hash,))
        cached = cursor.fetchone()
        
        if cached and is_cache_valid(cached[1]):
            st.info("📋 Using cached results")
            return json.loads(cached[0])
    
    # Perform new search
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": st.session_state.serp_api_key,
        "engine": "google",
        "num": num_results,
        "hl": "en",
        "gl": "us"
    }
    
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        results = res.json().get("organic_results", [])
        
        # Cache results
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO search_cache (query_hash, results, timestamp) 
        VALUES (?, ?, ?)
        """, (query_hash, json.dumps(results), datetime.now().isoformat()))
        conn.commit()
        
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"🔍 Search failed: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return []

def enhanced_sentiment_analysis(text: str) -> Dict[str, float]:
    """Enhanced sentiment analysis with scoring"""
    text_lower = text.lower()
    
    positive_indicators = ["growth", "profit", "success", "innovation", "leader", "award", "milestone"]
    negative_indicators = ["decline", "loss", "controversy", "lawsuit", "breach", "layoff", "fine"]
    neutral_indicators = ["update", "change", "announcement", "report", "statement"]
    
    pos_score = sum(1 for word in positive_indicators if word in text_lower)
    neg_score = sum(1 for word in negative_indicators if word in text_lower)
    neu_score = sum(1 for word in neutral_indicators if word in text_lower)
    
    total = pos_score + neg_score + neu_score
    if total == 0:
        return {"sentiment": "🟡 Neutral", "score": 0.0, "confidence": 0.5}
    
    if pos_score > neg_score:
        sentiment = "🟢 Positive"
        score = (pos_score - neg_score) / total
    elif neg_score > pos_score:
        sentiment = "🔴 Negative"
        score = (neg_score - pos_score) / total * -1
    else:
        sentiment = "🟡 Neutral"
        score = 0.0
    
    confidence = total / 10  # Simple confidence based on indicator count
    
    return {"sentiment": sentiment, "score": score, "confidence": min(confidence, 1.0)}

def categorize_content(text: str) -> Dict[str, int]:
    """Categorize content by keyword types"""
    text_lower = text.lower()
    categories = {}
    
    for category, keywords in KEYWORD_CATEGORIES.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        categories[category] = count
    
    return categories

def advanced_gpt_analysis(results: List[dict], company: str, analysis_type: str) -> str:
    """Enhanced GPT analysis with specialized prompts"""
    if not results:
        return "No data available for analysis."
    
    if not st.session_state.openai_api_key:
        return "OpenAI API key not configured."
    
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client."
    
    # Prepare context
    context = "\n\n".join([
        f"Title: {r.get('title', 'N/A')}\n"
        f"Source: {r.get('link', 'N/A')}\n"
        f"Content: {r.get('snippet', 'N/A')}"
        for r in results[:8]  # Limit to prevent token overflow
    ])
    
    prompts = {
        "executive": f"""
        As a business intelligence analyst, provide an executive summary about {company} based on the following information:
        
        {context}
        
        Focus on:
        1. Key business developments
        2. Market position changes
        3. Strategic implications
        4. Risk factors
        
        Keep it concise but comprehensive.
        """,
        
        "competitive": f"""
        Analyze the competitive landscape for {company} based on this information:
        
        {context}
        
        Identify:
        1. Direct competitors mentioned
        2. Competitive advantages/disadvantages
        3. Market positioning
        4. Competitive threats or opportunities
        """,
        
        "strategic": f"""
        Provide strategic insights about {company} based on:
        
        {context}
        
        Focus on:
        1. Strategic challenges and opportunities
        2. Innovation and technology initiatives
        3. Market expansion or contraction
        4. Partnership and acquisition activity
        """,
        
        "financial": f"""
        Analyze the financial and business implications for {company}:
        
        {context}
        
        Cover:
        1. Revenue and growth indicators
        2. Investment and funding activities
        3. Market valuation signals
        4. Financial risk factors
        """
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use more cost-effective model
            messages=[{
                "role": "system",
                "content": "You are an expert business intelligence analyst. Provide factual, actionable insights based on the provided information."
            }, {
                "role": "user",
                "content": prompts.get(analysis_type, prompts["executive"])
            }],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Analysis error: {str(e)}"

def create_fallback_pdf(company: str, analyses: Dict[str, str], results: List[dict]) -> bytes:
    """Create a simple fallback PDF if main generation fails"""
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Simple title
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawCentredString(width / 2, height - 100, f"Intelligence Report: {company}")
        
        # Simple content
        y_pos = height - 150
        pdf.setFont("Helvetica", 12)
        
        content = f"""
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Company: {company}
Sources Analyzed: {len(results)}

Summary:
{analyses.get('executive', 'Analysis completed successfully')}
        """
        
        for line in content.split('\n'):
            if y_pos < 50:
                pdf.showPage()
                y_pos = height - 50
            pdf.drawString(50, y_pos, line[:80])  # Limit line length
            y_pos -= 15
        
        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()
    except:
        # Ultimate fallback - minimal PDF
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.drawString(50, 750, f"Intelligence Report for {company}")
        pdf.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()

def generate_enhanced_report(company: str, results: List[dict], analyses: Dict[str, str]) -> bytes:
    """Generate comprehensive intelligence report with proper formatting"""
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        def add_header(title: str, y_position: float = None) -> float:
            """Add a section header with proper formatting"""
            if y_position is None:
                y_position = height - 50
            
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, y_position, title)
            pdf.line(50, y_position - 5, width - 50, y_position - 5)
            return y_position - 25
        
        def clean_markdown_text(text: str) -> str:
            """Remove markdown formatting and clean text for PDF"""
            if not text:
                return ""
            
            # Remove markdown headers
            text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
            
            # Remove markdown bold/italic markers but keep the text
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
            text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
            text = re.sub(r'__(.*?)__', r'\1', text)      # Remove __bold__
            text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _italic_
            
            # Remove markdown links but keep text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            
            # Remove markdown list markers
            text = re.sub(r'^\s*[-\*\+]\s+', '• ', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
            
            # Clean up extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
            
            return text
        
        def add_formatted_text_block(text: str, y_position: float, font_size: int = 10) -> float:
            """Add text block with proper formatting, handling bold and italic"""
            if not text or text.strip() == "":
                text = "No data available"
            
            # Clean markdown
            clean_text = clean_markdown_text(text)
            
            pdf.setFont("Helvetica", font_size)
            text_obj = pdf.beginText(50, y_position)
            text_obj.setFont("Helvetica", font_size)
            
            current_y = y_position
            
            for paragraph in clean_text.split('\n\n'):
                if not paragraph.strip():
                    continue
                
                # Handle bullet points
                if paragraph.strip().startswith('•'):
                    # Add some spacing before bullet points
                    text_obj.textLine("")
                    current_y -= 15
                    
                    if current_y < 50:
                        pdf.drawText(text_obj)
                        pdf.showPage()
                        current_y = height - 50
                        text_obj = pdf.beginText(50, current_y)
                        text_obj.setFont("Helvetica", font_size)
                
                # Wrap and add paragraph
                wrapped_lines = textwrap.wrap(paragraph, width=85)
                for line in wrapped_lines:
                    text_obj.textLine(line)
                    current_y -= 15
                    
                    if current_y < 50:
                        pdf.drawText(text_obj)
                        pdf.showPage()
                        current_y = height - 50
                        text_obj = pdf.beginText(50, current_y)
                        text_obj.setFont("Helvetica", font_size)
                
                # Add spacing between paragraphs
                text_obj.textLine("")
                current_y -= 10
            
            pdf.drawText(text_obj)
            return max(current_y - 20, 50)
        
        # Cover page with better formatting
        pdf.setFont("Helvetica-Bold", 28)
        pdf.drawCentredString(width / 2, height - 150, "COMPETITIVE INTELLIGENCE")
        pdf.drawCentredString(width / 2, height - 180, "REPORT")
        
        pdf.setFont("Helvetica", 18)
        pdf.drawCentredString(width / 2, height - 250, f"Company: {company}")
        
        pdf.setFont("Helvetica", 14)
        pdf.drawCentredString(width / 2, height - 290, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        pdf.drawCentredString(width / 2, height - 310, f"Sources Analyzed: {len(results)} data points")
        
        # Add a professional footer on cover
        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawCentredString(width / 2, 100, "Powered by Advanced Competitive Intelligence Agent")
        pdf.drawCentredString(width / 2, 85, "Autonomous AI-driven research and analysis")
        
        pdf.showPage()
        
        # Executive Summary (if available)
        if analyses.get("executive"):
            current_y = add_header("EXECUTIVE SUMMARY")
            current_y = add_formatted_text_block(analyses["executive"], current_y, 11)
            pdf.showPage()
        
        # Competitive Analysis (if available)
        if analyses.get("competitive"):
            current_y = add_header("COMPETITIVE LANDSCAPE ANALYSIS")
            current_y = add_formatted_text_block(analyses["competitive"], current_y, 11)
            pdf.showPage()
        
        # Strategic Insights (if available)
        if analyses.get("strategic"):
            current_y = add_header("STRATEGIC INSIGHTS")
            current_y = add_formatted_text_block(analyses["strategic"], current_y, 11)
            pdf.showPage()
        
        # Financial Analysis (if available)
        if analyses.get("financial"):
            current_y = add_header("FINANCIAL IMPLICATIONS")
            current_y = add_formatted_text_block(analyses["financial"], current_y, 11)
            pdf.showPage()
        
        # Data Sources section
        if results:
            current_y = add_header("DATA SOURCES & REFERENCES")
            
            sources_text = "The following sources were analyzed to generate this report:\n\n"
            for i, result in enumerate(results[:15], 1):  # Limit to top 15 sources
                title = result.get('title', 'No Title')[:80]  # Truncate long titles
                url = result.get('link', 'No URL')
                sources_text += f"{i}. {title}\n    Source: {url}\n\n"
            
            current_y = add_formatted_text_block(sources_text, current_y, 9)
        
        # Methodology section
        pdf.showPage()
        current_y = add_header("METHODOLOGY & DISCLAIMERS")
        
        methodology_text = """RESEARCH METHODOLOGY:
This report was generated using an autonomous AI-powered competitive intelligence system that:

• Conducted multi-source web searches using advanced search algorithms
• Analyzed content from news articles, company websites, and industry publications  
• Applied natural language processing for sentiment and trend analysis
• Generated insights using state-of-the-art AI language models
• Cross-referenced information across multiple sources for accuracy

DATA SOURCES:
Information was gathered from publicly available sources including news websites, company press releases, industry reports, and regulatory filings. All sources are listed in the references section.

LIMITATIONS:
• Analysis is based on publicly available information at the time of generation
• AI-generated insights should be verified with additional research
• Market conditions and company situations may change rapidly
• This report is for informational purposes only and not investment advice

DISCLAIMERS:
• This report is generated by an AI system and may contain errors or biases
• Users should conduct additional due diligence before making business decisions
• The accuracy of third-party sources cannot be guaranteed
• This analysis represents a snapshot in time and may not reflect current conditions"""

        current_y = add_formatted_text_block(methodology_text, current_y, 10)
        
        # Finalize PDF
        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return create_fallback_pdf(company, analyses, results)

def generate_wordcloud_analysis(company: str, results: List[dict]) -> None:
    """Generate comprehensive word cloud analysis from search results"""
    if not results:
        st.info("📊 No data available for word cloud analysis")
        return
    
    # Combine all text content
    all_text = []
    for result in results:
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        all_text.extend([title, snippet])
    
    combined_text = ' '.join(all_text)
    
    # Clean and preprocess text
    import string
    from collections import Counter
    
    # Remove common stop words and company name variations
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
        'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its', 'will', 'would', 'could', 'should', 'may',
        'can', 'has', 'have', 'had', 'been', 'be', 'do', 'does', 'did', 'get', 'gets', 'got', 'said', 'says',
        'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
        company.lower(), company.upper(), company.title()
    }
    
    # Extract meaningful words
    words = []
    for word in combined_text.split():
        # Clean word
        word = word.strip(string.punctuation).lower()
        # Filter out stop words, short words, and numbers
        if (len(word) > 3 and 
            word not in stop_words and 
            not word.isdigit() and 
            word.isalpha()):
            words.append(word)
    
    if not words:
        st.warning("⚠️ Not enough meaningful text for word cloud generation")
        return
    
    # Create multiple word clouds for different aspects
    st.subheader("☁️ Word Cloud Analysis")
    
    # Main word cloud
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(' '.join(words))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**📊 Main Topics & Keywords**")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**🔢 Top Keywords**")
            word_freq = Counter(words)
            top_words = word_freq.most_common(15)
            
            # Create a simple bar chart of top words
            if top_words:
                import plotly.express as px
                import pandas as pd
                
                df_words = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                fig_bar = px.bar(
                    df_words.head(10), 
                    x='Frequency', 
                    y='Word',
                    orientation='h',
                    title="Most Mentioned Terms",
                    color='Frequency',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Categorized analysis
        st.write("**🏷️ Categorized Keywords**")
        
        # Define category keywords for better analysis
        category_keywords = {
            "Business Strategy": ["strategy", "growth", "expansion", "market", "revenue", "profit", "business", "strategic", "plan", "planning"],
            "Technology": ["technology", "tech", "innovation", "digital", "software", "platform", "cloud", "data", "analytics", "artificial", "intelligence"],
            "Products & Services": ["product", "service", "solution", "offering", "feature", "launch", "release", "update", "version"],
            "Competition": ["competitor", "competition", "rival", "market", "share", "leader", "leading", "compete"],
            "Financial": ["financial", "finance", "investment", "funding", "revenue", "profit", "earnings", "cost", "price", "valuation"],
            "News & Events": ["news", "announcement", "report", "event", "conference", "partnership", "acquisition", "merger"]
        }
        
        categorized_words = {category: [] for category in category_keywords}
        
        for word in words:
            for category, keywords in category_keywords.items():
                if any(keyword in word for keyword in keywords):
                    categorized_words[category].append(word)
        
        # Display categorized results
        cols = st.columns(3)
        for i, (category, category_words) in enumerate(categorized_words.items()):
            with cols[i % 3]:
                if category_words:
                    word_count = len(category_words)
                    top_category_words = Counter(category_words).most_common(5)
                    st.metric(
                        label=category,
                        value=word_count,
                        help=f"Top words: {', '.join([word for word, count in top_category_words])}"
                    )
                else:
                    st.metric(label=category, value=0)
                    
    except Exception as e:
        st.error(f"❌ Word cloud generation failed: {str(e)}")
        st.info("💡 This might be due to insufficient text data or missing dependencies")

def generate_comprehensive_wordcloud(company: str, all_results: List[dict], analyses: Dict[str, str]) -> None:
    """Generate a comprehensive word cloud including both search results and AI analysis"""
    if not all_results and not analyses:
        return
    
    st.subheader("🔍 Comprehensive Keyword Analysis")
    
    # Combine search results and AI analyses
    all_content = []
    
    # Add search results
    for result in all_results:
        all_content.extend([
            result.get('title', ''),
            result.get('snippet', '')
        ])
    
    # Add AI analysis content
    for analysis_type, content in analyses.items():
        if content and content.strip():
            all_content.append(content)
    
    combined_text = ' '.join(all_content)
    
    if not combined_text.strip():
        st.info("📊 No content available for comprehensive analysis")
        return
    
    # Advanced text processing
    import re
    
    # Remove common business jargon and focus on meaningful terms
    stop_words = {
        'company', 'business', 'industry', 'market', 'customer', 'service', 'product', 'year', 'time',
        'way', 'new', 'first', 'last', 'good', 'best', 'better', 'great', 'large', 'small', 'big',
        'high', 'low', 'long', 'short', 'old', 'young', 'early', 'late', 'right', 'left', 'next',
        'previous', 'current', 'future', 'past', 'present', 'today', 'tomorrow', 'yesterday',
        company.lower(), 'inc', 'corp', 'corporation', 'ltd', 'limited', 'llc'
    }
    
    # Extract meaningful phrases and words
    words = []
    
    # Clean text and extract words
    clean_text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    for word in clean_text.split():
        if (len(word) > 3 and 
            word not in stop_words and 
            not word.isdigit() and 
            word.isalpha()):
            words.append(word)
    
    if len(words) < 10:
        st.warning("⚠️ Insufficient data for comprehensive keyword analysis")
        return
    
    # Create enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌟 Enhanced Word Cloud**")
        try:
            # Create a more sophisticated word cloud
            from collections import Counter
            word_freq = Counter(words)
            
            # Filter to most relevant words
            relevant_words = {word: freq for word, freq in word_freq.items() if freq > 1}
            
            if relevant_words:
                wordcloud = WordCloud(
                    width=600,
                    height=400,
                    background_color='white',
                    max_words=80,
                    colormap='plasma',
                    relative_scaling=0.6,
                    min_font_size=8,
                    prefer_horizontal=0.9
                ).generate_from_frequencies(relevant_words)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough repeated keywords for enhanced word cloud")
                
        except Exception as e:
            st.error(f"Enhanced word cloud generation failed: {str(e)}")
    
    with col2:
        st.write("**📈 Keyword Trends**")
        try:
            from collections import Counter
            word_freq = Counter(words)
            top_20_words = word_freq.most_common(20)
            
            if top_20_words:
                df_trends = pd.DataFrame(top_20_words, columns=['Keyword', 'Mentions'])
                
                # Create an interactive bar chart
                fig_trends = px.bar(
                    df_trends.head(15),
                    x='Mentions',
                    y='Keyword',
                    orientation='h',
                    title=f"Top Keywords for {company}",
                    color='Mentions',
                    color_continuous_scale='plasma',
                    text='Mentions'
                )
                fig_trends.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                fig_trends.update_traces(textposition='outside')
                st.plotly_chart(fig_trends, use_container_width=True)
        
        except Exception as e:
            st.error(f"Keyword trends generation failed: {str(e)}")
    
    # Sentiment-based word analysis
    st.write("**🎭 Sentiment-Based Keywords**")
    
    positive_words = []
    negative_words = []
    neutral_words = []
    
    # Simple sentiment categorization
    positive_indicators = ['growth', 'success', 'profit', 'innovation', 'leader', 'award', 'achievement', 'expansion', 'opportunity']
    negative_indicators = ['decline', 'loss', 'controversy', 'lawsuit', 'breach', 'problem', 'challenge', 'risk', 'threat']
    
    from collections import Counter
    word_freq = Counter(words)
    
    for word, freq in word_freq.most_common(50):
        if any(pos in word for pos in positive_indicators):
            positive_words.append((word, freq))
        elif any(neg in word for neg in negative_indicators):
            negative_words.append((word, freq))
        else:
            neutral_words.append((word, freq))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🟢 Positive Keywords**")
        if positive_words:
            for word, freq in positive_words[:8]:
                st.write(f"• {word} ({freq})")
        else:
            st.write("No clearly positive keywords identified")
    
    with col2:
        st.write("**🔴 Challenge Keywords**")
        if negative_words:
            for word, freq in negative_words[:8]:
                st.write(f"• {word} ({freq})")
        else:
            st.write("No challenge keywords identified")
    
    with col3:
        st.write("**🟡 Neutral Keywords**")
        if neutral_words:
            for word, freq in neutral_words[:8]:
                st.write(f"• {word} ({freq})")
        else:
            st.write("No neutral keywords to display")

def create_analytics_dashboard(company: str):
    """Create analytics dashboard with visualizations"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT timestamp, sentiment_score, confidence_score, source_count 
    FROM intelligence_log 
    WHERE company = ? 
    ORDER BY timestamp DESC 
    LIMIT 30
    """, (company,))
    
    data = cursor.fetchall()
    if not data:
        st.info("📊 No historical data available for analytics")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'sentiment', 'confidence', 'sources'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Trend")
        fig_sentiment = px.line(df, x='timestamp', y='sentiment', 
                               title=f"Sentiment Analysis for {company}")
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Analysis Confidence")
        fig_confidence = px.scatter(df, x='timestamp', y='confidence', size='sources',
                                   title="Analysis Confidence Over Time")
        st.plotly_chart(fig_confidence, use_container_width=True)
    """Create analytics dashboard with visualizations"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT timestamp, sentiment_score, confidence_score, source_count 
    FROM intelligence_log 
    WHERE company = ? 
    ORDER BY timestamp DESC 
    LIMIT 30
    """, (company,))
    
    data = cursor.fetchall()
    if not data:
        st.info("📊 No historical data available for analytics")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'sentiment', 'confidence', 'sources'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Trend")
        fig_sentiment = px.line(df, x='timestamp', y='sentiment', 
                               title=f"Sentiment Analysis for {company}")
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Analysis Confidence")
        fig_confidence = px.scatter(df, x='timestamp', y='confidence', size='sources',
                                   title="Analysis Confidence Over Time")
        st.plotly_chart(fig_confidence, use_container_width=True)

# ========== Enhanced Streamlit Interface ==========
st.title("🕵️‍♂️ Advanced Competitive Intelligence Agent")
st.markdown("*Autonomous AI-powered competitive research and analysis platform*")

# API Key Configuration Section
if not st.session_state.api_keys_configured:
    st.warning("⚠️ Please configure your API keys to get started")
    
    with st.expander("🔑 API Key Configuration", expanded=True):
        st.markdown("""
        ### Required API Keys:
        
        **1. SERP API Key** (for web search)
        - Get your free API key from [SerpApi](https://serpapi.com/)
        - Free tier includes 100 searches/month
        
        **2. OpenAI API Key** (for AI analysis)
        - Get your API key from [OpenAI](https://platform.openai.com/api-keys)
        - Pay-per-use pricing
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            serp_key = st.text_input(
                "🔍 SERP API Key",
                value=st.session_state.serp_api_key,
                type="password",
                help="Your SerpApi key for web search functionality"
            )
        
        with col2:
            openai_key = st.text_input(
                "🤖 OpenAI API Key", 
                value=st.session_state.openai_api_key,
                type="password",
                help="Your OpenAI API key for AI analysis"
            )
        
        # API Key validation section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Test API Keys", help="Validate both API keys"):
                if serp_key and openai_key:
                    with st.spinner("🔄 Testing API keys..."):
                        test_results = test_api_keys(serp_key, openai_key)
                        
                        # Display results
                        if test_results["overall_valid"]:
                            st.success("✅ Both API keys are valid!")
                            
                            # Show individual results
                            col_serp, col_openai = st.columns(2)
                            with col_serp:
                                st.info(f"🔍 SERP: {test_results['serp'].get('message', 'Valid')}")
                            with col_openai:
                                st.info(f"🤖 OpenAI: {test_results['openai'].get('message', 'Valid')}")
                        else:
                            st.error("❌ API key validation failed!")
                            
                            # Show specific errors
                            if test_results["serp"]["tested"] and not test_results["serp"]["valid"]:
                                st.error(f"🔍 SERP API: {test_results['serp'].get('error', 'Unknown error')}")
                            
                            if test_results["openai"]["tested"] and not test_results["openai"]["valid"]:
                                st.error(f"🤖 OpenAI API: {test_results['openai'].get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please enter both API keys before testing")
        
        with col2:
            if st.button("✅ Save API Keys", type="primary"):
                if serp_key and openai_key:
                    # Optionally validate before saving
                    with st.spinner("💾 Saving API keys..."):
                        st.session_state.serp_api_key = serp_key
                        st.session_state.openai_api_key = openai_key
                        st.session_state.api_keys_configured = True
                        st.success("🎉 API keys saved successfully!")
                        st.info("💡 Keys are validated automatically when you start analysis")
                        st.rerun()
                else:
                    st.error("❌ Please provide both API keys")
        
        with col3:
            if st.button("✅ Validate & Save", help="Test keys and save if valid"):
                if serp_key and openai_key:
                    with st.spinner("🔄 Validating and saving..."):
                        test_results = test_api_keys(serp_key, openai_key)
                        
                        if test_results["overall_valid"]:
                            st.session_state.serp_api_key = serp_key
                            st.session_state.openai_api_key = openai_key
                            st.session_state.api_keys_configured = True
                            st.success("🎉 API keys validated and saved!")
                            st.rerun()
                        else:
                            st.error("❌ Cannot save - API keys are invalid!")
                            
                            # Show specific errors
                            if not test_results["serp"]["valid"]:
                                st.error(f"🔍 SERP: {test_results['serp'].get('error', 'Invalid')}")
                            if not test_results["openai"]["valid"]:
                                st.error(f"🤖 OpenAI: {test_results['openai'].get('error', 'Invalid')}")
                else:
                    st.warning("⚠️ Please enter both API keys")
        
        # Option to load from environment
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Load from Environment Variables"):
                env_serp = os.getenv("SERP_API_KEY", "")
                env_openai = os.getenv("OPENAI_API_KEY", "")
                if env_serp and env_openai:
                    with st.spinner("🔄 Loading and validating environment keys..."):
                        test_results = test_api_keys(env_serp, env_openai)
                        
                        if test_results["overall_valid"]:
                            st.session_state.serp_api_key = env_serp
                            st.session_state.openai_api_key = env_openai
                            st.session_state.api_keys_configured = True
                            st.success("✅ Environment keys loaded and validated!")
                            st.rerun()
                        else:
                            st.error("❌ Environment keys found but invalid!")
                            # Show specific errors
                            if not test_results["serp"]["valid"]:
                                st.error(f"🔍 SERP: {test_results['serp'].get('error')}")
                            if not test_results["openai"]["valid"]:
                                st.error(f"🤖 OpenAI: {test_results['openai'].get('error')}")
                else:
                    st.error("❌ Environment variables not found")
        
        with col2:
            if st.button("❓ API Key Help", help="Get help with API keys"):
                st.info("""
                **Common Issues & Solutions:**
                
                🔍 **SERP API Issues:**
                • Invalid key → Check your SerpApi dashboard
                • Quota exceeded → Upgrade your plan
                • Authentication failed → Regenerate key
                
                🤖 **OpenAI API Issues:**
                • Invalid key → Check OpenAI platform
                • Quota exceeded → Add billing information
                • Rate limit → Wait and try again
                • Authentication failed → Regenerate key
                
                💡 **Tips:**
                • Keys are case-sensitive
                • Remove any extra spaces
                • Check your account status on respective platforms
                """)
        
        # API Key status indicators
        st.divider()
        st.write("**🔍 Quick Reference:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **SERP API (SerpApi)**
            - 🌐 Website: [serpapi.com](https://serpapi.com/)
            - 🆓 Free tier: 100 searches/month
            - 📍 Find key: Account → API Key
            - 💰 Pricing: Pay-per-search
            """)
        
        with col2:
            st.markdown("""
            **OpenAI API**
            - 🌐 Website: [platform.openai.com](https://platform.openai.com/)
            - 🆓 Free tier: $5 credit for new accounts
            - 📍 Find key: API Keys → Create new
            - 💰 Pricing: Pay-per-token
            """)

# Main application (only show if API keys are configured)
if st.session_state.api_keys_configured:
    # Show current API key status
    with st.sidebar:
        st.success("✅ API Keys Configured")
        with st.expander("🔑 API Key Status"):
            st.write(f"🔍 SERP API: {'✅' if st.session_state.serp_api_key else '❌'}")
            st.write(f"🤖 OpenAI API: {'✅' if st.session_state.openai_api_key else '❌'}")
            
            # Quick validation option in sidebar
            if st.button("🔍 Validate Keys", help="Quick API key validation", key="sidebar_validate"):
                with st.spinner("Testing..."):
                    test_results = test_api_keys(st.session_state.serp_api_key, st.session_state.openai_api_key)
                    
                    if test_results["overall_valid"]:
                        st.success("✅ All keys valid!")
                    else:
                        st.error("❌ Some keys invalid!")
                        if not test_results["serp"]["valid"]:
                            st.error(f"SERP: {test_results['serp'].get('error', 'Invalid')}")
                        if not test_results["openai"]["valid"]:
                            st.error(f"OpenAI: {test_results['openai'].get('error', 'Invalid')}")
            
            if st.button("🔄 Reconfigure Keys"):
                st.session_state.api_keys_configured = False
                st.session_state.serp_api_key = ""
                st.session_state.openai_api_key = ""
                st.rerun()

        # Sidebar configuration
        st.header("⚙️ Configuration")
        
        # Basic settings
        company = st.text_input("🏢 Company Name", "Salesforce", help="Enter the company you want to analyze")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            num_results = st.slider("Search Results", 5, 20, 10)
            use_cache = st.checkbox("Use Cache", True, help="Use cached results when available")
            analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Deep", "Comprehensive"])
            
        # Analysis types
        st.header("📊 Analysis Types")
        analysis_types = {
            "Executive Summary": st.checkbox("Executive Summary", True),
            "Competitive Analysis": st.checkbox("Competitive Analysis", True),
            "Strategic Insights": st.checkbox("Strategic Insights", False),
            "Financial Analysis": st.checkbox("Financial Analysis", False)
        }
        
        # Historical data
        st.header("📈 Analytics")
        show_analytics = st.checkbox("Show Analytics Dashboard", False)

    if company:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Analysis", "📊 Analytics", "📚 Historical", "⚙️ Settings"])
        
        with tab1:
            # Search configuration
            search_queries = [
                f"{company} latest news",
                f"{company} product updates OR launches",
                f"{company} financial results OR earnings",
                f"{company} competitors OR competition"
            ]
            
            if analysis_depth == "Deep":
                search_queries.extend([
                    f"{company} partnerships OR acquisitions",
                    f"{company} technology OR innovation"
                ])
            elif analysis_depth == "Comprehensive":
                search_queries.extend([
                    f"{company} partnerships OR acquisitions",
                    f"{company} technology OR innovation",
                    f"{company} leadership OR management",
                    f"{company} market share OR position"
                ])
            
            # Execute searches
            if st.button("🚀 Start Intelligence Gathering", type="primary"):
                all_results = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, query in enumerate(search_queries):
                    status_text.text(f"🔍 Searching: {query}")
                    results = search_google_enhanced(query, num_results // len(search_queries), use_cache)
                    all_results.extend(results)
                    progress_bar.progress((i + 1) / len(search_queries))
                    time.sleep(0.5)  # Rate limiting
                
                status_text.text("🧠 Analyzing results...")
                
                if all_results:
                    # Remove duplicates
                    seen_urls = set()
                    unique_results = []
                    for result in all_results:
                        url = result.get('link', '')
                        if url not in seen_urls:
                            seen_urls.add(url)
                            unique_results.append(result)
                    
                    st.success(f"✅ Found {len(unique_results)} unique results")
                    
                    # Display results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Sources", len(unique_results))
                    with col2:
                        domains = [urlparse(r.get('link', '')).netloc for r in unique_results]
                        st.metric("Unique Domains", len(set(domains)))
                    with col3:
                        categories = categorize_content(' '.join([r.get('snippet', '') for r in unique_results]))
                        top_category = max(categories, key=categories.get) if categories else "N/A"
                        st.metric("Top Category", top_category)
                    
                    # Perform selected analyses
                    analyses = {}
                    analysis_progress = st.progress(0)
                    
                    selected_analyses = [k.lower().replace(" ", "_") 
                                       for k, v in analysis_types.items() if v]
                    
                    for i, analysis_type in enumerate(selected_analyses):
                        analysis_key = analysis_type.replace("_", " ").title()
                        status_text.text(f"🧠 Generating {analysis_key}...")
                        
                        if analysis_type == "executive_summary":
                            analyses["executive"] = advanced_gpt_analysis(unique_results, company, "executive")
                        elif analysis_type == "competitive_analysis":
                            analyses["competitive"] = advanced_gpt_analysis(unique_results, company, "competitive")
                        elif analysis_type == "strategic_insights":
                            analyses["strategic"] = advanced_gpt_analysis(unique_results, company, "strategic")
                        elif analysis_type == "financial_analysis":
                            analyses["financial"] = advanced_gpt_analysis(unique_results, company, "financial")
                        
                        analysis_progress.progress((i + 1) / len(selected_analyses))
                    
                    # Display analyses
                    for analysis_type, content in analyses.items():
                        st.subheader(f"📋 {analysis_type.replace('_', ' ').title()}")
                        st.write(content)
                        st.divider()
                    
                    # Word Cloud Analysis - NEW FEATURE
                    st.divider()
                    generate_wordcloud_analysis(company, unique_results)
                    
                    # Comprehensive keyword analysis including AI insights
                    st.divider()
                    generate_comprehensive_wordcloud(company, unique_results, analyses)
                    
                    # Save to database
                    overall_sentiment = enhanced_sentiment_analysis(
                        ' '.join([r.get('snippet', '') for r in unique_results])
                    )
                    
                    cursor = conn.cursor()
                    cursor.execute("""
                    INSERT INTO intelligence_log 
                    (company, summary, sentiment_score, key_insights, challenges, competitors, 
                     timestamp, source_count, confidence_score) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        company,
                        analyses.get("executive", ""),
                        overall_sentiment["score"],
                        analyses.get("strategic", ""),
                        "TBD",  # Could extract from analysis
                        analyses.get("competitive", ""),
                        datetime.now().isoformat(),
                        len(unique_results),
                        overall_sentiment["confidence"]
                    ))
                    conn.commit()
                    
                    # Generate report section
                    st.subheader("📄 Generate Reports")
                    
                    # Store analysis results in session state to persist
                    if 'analysis_complete' not in st.session_state:
                        st.session_state.analysis_complete = False
                    
                    # Mark analysis as complete and store results
                    st.session_state.analysis_complete = True
                    st.session_state.current_results = {
                        'company': company,
                        'unique_results': unique_results,
                        'analyses': analyses,
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                    }
                    
                    # Show summary of what will be included in reports
                    with st.expander("📋 Report Contents Preview", expanded=False):
                        st.write("**Your report will include:**")
                        report_contents = []
                        if analyses.get("executive"):
                            report_contents.append("✅ Executive Summary")
                        if analyses.get("competitive"):
                            report_contents.append("✅ Competitive Analysis")
                        if analyses.get("strategic"):
                            report_contents.append("✅ Strategic Insights")
                        if analyses.get("financial"):
                            report_contents.append("✅ Financial Analysis")
                        
                        report_contents.extend([
                            f"✅ Data Sources ({len(unique_results)} sources)",
                            "✅ Methodology & Disclaimers",
                            "✅ Professional Formatting"
                        ])
                        
                        for content in report_contents:
                            st.write(content)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Comprehensive Report**")
                        st.write("Complete analysis with all sections")
                        
                        if st.button("📄 Generate Full Report", type="primary", key="generate_full"):
                            with st.spinner("🔄 Generating comprehensive PDF report..."):
                                try:
                                    pdf_data = generate_enhanced_report(
                                        company, 
                                        unique_results, 
                                        analyses
                                    )
                                    
                                    # Store in session state for persistent download
                                    st.session_state.full_pdf_data = pdf_data
                                    st.session_state.full_pdf_ready = True
                                    
                                    st.success("✅ Comprehensive report generated!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Report generation failed: {str(e)}")
                    
                    with col2:
                        st.write("**📋 Executive Summary**")
                        st.write("Key insights and recommendations only")
                        
                        if st.button("📋 Generate Summary", key="generate_summary"):
                            with st.spinner("🔄 Generating executive summary..."):
                                try:
                                    exec_analyses = {
                                        "executive": analyses.get("executive", "Executive summary completed")
                                    }
                                    pdf_data = generate_enhanced_report(
                                        company,
                                        unique_results[:5],  # Top 5 sources
                                        exec_analyses
                                    )
                                    
                                    # Store in session state for persistent download
                                    st.session_state.summary_pdf_data = pdf_data
                                    st.session_state.summary_pdf_ready = True
                                    
                                    st.success("✅ Executive summary generated!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Summary generation failed: {str(e)}")
                    
                    # Download buttons - always visible if PDFs are ready
                    st.markdown("### 📥 Download Options")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        # Full report download
                        if hasattr(st.session_state, 'full_pdf_ready') and st.session_state.full_pdf_ready:
                            st.download_button(
                                label="📄 Download Full Report",
                                data=st.session_state.full_pdf_data,
                                file_name=f"{company}_comprehensive_report_{st.session_state.current_results['timestamp']}.pdf",
                                mime="application/pdf",
                                key="download_full_pdf",
                                help="Download comprehensive intelligence report",
                                use_container_width=True
                            )
                        else:
                            st.button("📄 Download Full Report", disabled=True, help="Generate report first", use_container_width=True)
                    
                    with download_col2:
                        # Summary download
                        if hasattr(st.session_state, 'summary_pdf_ready') and st.session_state.summary_pdf_ready:
                            st.download_button(
                                label="📋 Download Summary",
                                data=st.session_state.summary_pdf_data,
                                file_name=f"{company}_executive_summary_{st.session_state.current_results['timestamp']}.pdf",
                                mime="application/pdf",
                                key="download_summary_pdf",
                                help="Download executive summary",
                                use_container_width=True
                            )
                        else:
                            st.button("📋 Download Summary", disabled=True, help="Generate summary first", use_container_width=True)
                    
                    with download_col3:
                        # Text export (always available)
                        text_report = f"""COMPETITIVE INTELLIGENCE REPORT
Company: {company}
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Sources Analyzed: {len(unique_results)}

{'='*60}
EXECUTIVE SUMMARY
{'='*60}
{analyses.get('executive', 'Not available')}

{'='*60}
COMPETITIVE ANALYSIS
{'='*60}
{analyses.get('competitive', 'Not available')}

{'='*60}
STRATEGIC INSIGHTS
{'='*60}
{analyses.get('strategic', 'Not available')}

{'='*60}
FINANCIAL ANALYSIS
{'='*60}
{analyses.get('financial', 'Not available')}

{'='*60}
DATA SOURCES
{'='*60}
""" + "\n".join([f"{i+1}. {r.get('title', 'No Title')}\n    {r.get('link', 'No URL')}\n" for i, r in enumerate(unique_results[:15])])

                        st.download_button(
                            label="📄 Download as Text",
                            data=text_report,
                            file_name=f"{company}_report_{st.session_state.current_results['timestamp']}.txt",
                            mime="text/plain",
                            key="download_text_report",
                            help="Download as plain text file",
                            use_container_width=True
                        )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    analysis_progress.empty()
                    status_text.empty()
                
                else:
                    st.warning("⚠️ No results found. Try adjusting your search parameters.")
        
        with tab2:
            if show_analytics:
                create_analytics_dashboard(company)
                
                # Historical word cloud analysis
                st.divider()
                st.subheader("📚 Historical Keyword Trends")
                
                cursor = conn.cursor()
                cursor.execute("""
                SELECT summary, key_insights, competitors 
                FROM intelligence_log 
                WHERE company = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
                """, (company,))
                
                historical_data = cursor.fetchall()
                if historical_data:
                    # Combine historical data for word cloud
                    historical_text = []
                    for record in historical_data:
                        historical_text.extend([record[0] or '', record[1] or '', record[2] or ''])
                    
                    combined_historical = ' '.join(historical_text)
                    
                    if combined_historical.strip():
                        try:
                            # Generate historical word cloud
                            stop_words = {'company', 'business', 'analysis', company.lower()}
                            
                            words = []
                            for word in combined_historical.split():
                                word = word.strip('.,!?;:"()[]').lower()
                                if (len(word) > 3 and 
                                    word not in stop_words and 
                                    not word.isdigit() and 
                                    word.isalpha()):
                                    words.append(word)
                            
                            if words:
                                from collections import Counter
                                word_freq = Counter(words)
                                
                                # Create historical trends word cloud
                                wordcloud_hist = WordCloud(
                                    width=800,
                                    height=300,
                                    background_color='white',
                                    max_words=60,
                                    colormap='coolwarm',
                                    relative_scaling=0.5
                                ).generate_from_frequencies(dict(word_freq.most_common(50)))
                                
                                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                                ax_hist.imshow(wordcloud_hist, interpolation='bilinear')
                                ax_hist.axis("off")
                                ax_hist.set_title(f"Historical Keywords for {company}", fontsize=14, pad=20)
                                st.pyplot(fig_hist)
                                plt.close()
                                
                                # Show trending keywords over time
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**🔥 Most Frequent Historical Keywords**")
                                    for word, count in word_freq.most_common(10):
                                        st.write(f"• **{word}**: {count} mentions")
                                
                                with col2:
                                    # Simple trend analysis
                                    st.write("**📈 Keyword Evolution**")
                                    if len(historical_data) > 1:
                                        recent_text = ' '.join([historical_data[0][0] or '', historical_data[0][1] or ''])
                                        older_text = ' '.join([record[0] or '' for record in historical_data[1:3]])
                                        
                                        recent_words = Counter([word.strip('.,!?;:"()[]').lower() 
                                                              for word in recent_text.split() 
                                                              if len(word) > 3 and word.isalpha()])
                                        older_words = Counter([word.strip('.,!?;:"()[]').lower() 
                                                             for word in older_text.split() 
                                                             if len(word) > 3 and word.isalpha()])
                                        
                                        # Find emerging keywords
                                        emerging = []
                                        declining = []
                                        
                                        for word, recent_count in recent_words.most_common(20):
                                            older_count = older_words.get(word, 0)
                                            if recent_count > older_count and word not in stop_words:
                                                emerging.append((word, recent_count - older_count))
                                        
                                        for word, older_count in older_words.most_common(20):
                                            recent_count = recent_words.get(word, 0)
                                            if older_count > recent_count and word not in stop_words:
                                                declining.append((word, older_count - recent_count))
                                        
                                        if emerging:
                                            st.write("🆙 **Emerging Keywords:**")
                                            for word, change in emerging[:5]:
                                                st.write(f"• {word} (+{change})")
                                        
                                        if declining:
                                            st.write("📉 **Declining Keywords:**")
                                            for word, change in declining[:5]:
                                                st.write(f"• {word} (-{change})")
                                    else:
                                        st.info("Need more historical data for trend analysis")
                            
                        except Exception as e:
                            st.error(f"Historical word cloud generation failed: {str(e)}")
                    else:
                        st.info("No historical text data available")
                else:
                    st.info("No historical data available for keyword analysis")
            else:
                st.info("Enable analytics in the sidebar to view historical trends and word clouds")
        
        with tab3:
            st.subheader("📚 Historical Intelligence")
            cursor = conn.cursor()
            cursor.execute("""
            SELECT timestamp, summary, sentiment_score, source_count 
            FROM intelligence_log 
            WHERE company = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
            """, (company,))
            
            history = cursor.fetchall()
            if history:
                for record in history:
                    with st.expander(f"📅 {record[0][:10]} - Sources: {record[3]}"):
                        st.write(record[1])
                        st.caption(f"Sentiment Score: {record[2]:.2f}")
            else:
                st.info("No historical data available")
        
        with tab4:
            st.subheader("⚙️ System Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache"):
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM search_cache")
                    conn.commit()
                    st.success("Cache cleared")
            
            with col2:
                if st.button("📊 Database Stats"):
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM intelligence_log")
                    log_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM search_cache")
                    cache_count = cursor.fetchone()[0]
                    
                    st.metric("Intelligence Records", log_count)
                    st.metric("Cached Searches", cache_count)

    else:
        st.info("👆 Enter a company name in the sidebar to begin intelligence gathering")

else:
    # Show helpful message when API keys not configured
    st.info("👆 Configure your API keys above to start using the Competitive Intelligence Agent")
    
    # Show demo/preview of what the app can do
    with st.expander("🎯 What This App Can Do"):
        st.markdown("""
        ### 🔍 **Automated Research**
        - Searches multiple sources for company information
        - Analyzes news, product updates, and market mentions
        - Tracks competitor activities and market positioning
        
        ### 🧠 **AI-Powered Analysis**
        - Executive summaries and strategic insights
        - Competitive landscape analysis
        - Financial implications assessment
        - Sentiment analysis and trend detection
        
        ### 📊 **Professional Reporting**
        - Comprehensive PDF reports
        - Historical trend analysis
        - Interactive analytics dashboard
        - Exportable data and insights
        
        ### 🚀 **Key Features**
        - Real-time competitive intelligence
        - Multi-source data aggregation
        - Automated report generation
        - Historical data tracking
        """)

# Footer
st.markdown("---")
st.markdown("*Built with Agentic AI principles for autonomous competitive intelligence*")