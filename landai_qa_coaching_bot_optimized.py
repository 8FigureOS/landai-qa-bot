#!/usr/bin/env python3
"""
LandAI QA Coaching Chatbot - OPTIMIZED VERSION
A fast, feature-rich Streamlit app for AI-powered coaching insights
"""

import streamlit as st
import requests
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import openai
from collections import Counter

# Configuration - Use Streamlit Secrets (falls back to hardcoded for local dev)
try:
    # Try to load from Streamlit secrets (for deployment)
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except:
    # Fallback to hardcoded values (for local development)
    SUPABASE_URL = "https://yeflauigtjsexadhiqiq.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InllZmxhdWlndGpzZXhhZGhpcWlxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzNjA0MjMsImV4cCI6MjA2MzkzNjQyM30.W-6c2_rsa1TRM1S7pvEy4vT1kGx7bevpqRKgbHio1gE"
    OPENAI_API_KEY = "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"

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Supabase headers
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept-Profile": "readymode_data",
    "Content-Profile": "readymode_data"
}

# Cache timeout settings (in seconds)
CACHE_TTL_AGENTS = 3600  # 1 hour
CACHE_TTL_CALL_DATA = 900  # 15 minutes
CACHE_TTL_METRICS = 300  # 5 minutes

@st.cache_data(ttl=CACHE_TTL_AGENTS, show_spinner="Loading agents...")
def get_all_agents_optimized() -> List[Dict[str, any]]:
    """
    OPTIMIZED: Fetch all unique agents quickly (without call counts initially)
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/call_logs"
        
        # Strategy: Scan through ALL records to find every unique agent
        seen_agents = set()
        offset = 0
        limit = 1000
        max_records = 150000  # Scan entire database (we have 148k records)
        
        st.info("🔍 Scanning database for all agents...")
        
        # Discover ALL unique agents
        while offset < max_records:
            params = {
                "select": "agent_name",
                "agent_name": "not.is.null",
                "limit": limit,
                "offset": offset
            }
            
            try:
                response = requests.get(url, headers=HEADERS, params=params, timeout=30)
                
                if response.status_code in [200, 206]:  # 206 = Partial Content is OK
                    data = response.json()
                    if not data:
                        # Reached end of data
                        break
                    
                    # Collect unique agent names
                    for record in data:
                        agent_name = record.get('agent_name')
                        if agent_name and agent_name.strip():
                            seen_agents.add(agent_name)
                    
                    # Progress update every 10k records
                    if (offset + limit) % 10000 == 0:
                        st.info(f"   Scanned {offset + limit:,} records... Found {len(seen_agents)} unique agents so far")
                    
                    # If we got less than limit, we've reached the end
                    if len(data) < limit:
                        break
                    
                    offset += limit
                else:
                    st.warning(f"⚠️ API returned status {response.status_code}. Using {len(seen_agents)} agents discovered so far.")
                    break
                    
            except requests.exceptions.ConnectionError as e:
                st.error("🔌 **Cannot connect to database!**")
                st.error(f"Error: {str(e)[:200]}")
                return []
            except requests.exceptions.Timeout:
                st.warning(f"⏱️ Request timed out after {offset:,} records. Using {len(seen_agents)} agents discovered so far.")
                break
            except Exception as e:
                st.error(f"❌ Error at offset {offset:,}: {str(e)}")
                break
        
        if not seen_agents:
            st.error("❌ **No agents found in database**")
            return []
        
        st.success(f"✅ Found {len(seen_agents)} unique agents from {offset:,} records!")
        
        # Return agents sorted alphabetically (no call counts)
        agents_list = [{'name': agent, 'total_calls': 0} for agent in sorted(seen_agents)]
        
        return agents_list
        
    except Exception as e:
        st.error(f"❌ **Unexpected error:** {str(e)}")
        import traceback
        with st.expander("📋 Full error details"):
            st.code(traceback.format_exc())
        return []

@st.cache_data(ttl=CACHE_TTL_CALL_DATA, show_spinner="Loading call data...")
def get_agent_call_data_optimized(
    agent_name: str, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    campaign: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = 20
) -> List[Dict]:
    """
    OPTIMIZED: Fetch call data starting from QA evaluations (only calls with QA scores)
    """
    try:
        # STEP 1: Get call_log_ids for this agent with filters
        call_params = {
            "select": "call_log_id,log_time,phone_number,customer_name,disposition,campaign_name",
            "agent_name": f"eq.{agent_name}",
            "limit": 1000  # Get more to ensure we have enough with QA evals
        }
        
        # Apply date filters
        if start_date:
            call_params['log_time'] = f'gte.{start_date}'
            if end_date:
                call_params['and'] = f'(log_time.lt.{end_date})'
        elif end_date:
            call_params['log_time'] = f'lt.{end_date}'
        
        # Apply campaign filter
        if campaign and campaign != "All Campaigns":
            call_params['campaign_name'] = f'eq.{campaign}'
        
        call_logs_url = f"{SUPABASE_URL}/rest/v1/call_logs"
        call_response = requests.get(call_logs_url, headers=HEADERS, params=call_params, timeout=20)
        
        if call_response.status_code not in [200, 206]:
            st.error(f"Failed to fetch call logs: {call_response.status_code}")
            return []
        
        all_call_logs = call_response.json()
        
        if not all_call_logs:
            return []
        
        # Create lookup dictionary for call logs
        call_logs_dict = {call['call_log_id']: call for call in all_call_logs}
        call_ids = [str(call['call_log_id']) for call in all_call_logs]
        
        # STEP 2: Get QA evaluations - PRIMARY DATA SOURCE
        qa_params = {
            "select": "call_log_id,overall_score_percentage,politeness_score,compliance_score,enthusiasm_score,clarity_score,created_at",
            "call_log_id": f"in.({','.join(call_ids)})",
            "order": "created_at.desc",
            "limit": limit
        }
        
        # Apply score filters
        if min_score is not None:
            qa_params['overall_score_percentage'] = f'gte.{min_score}'
        if max_score is not None:
            if 'overall_score_percentage' in qa_params:
                qa_params['and'] = f'(overall_score_percentage.lte.{max_score})'
            else:
                qa_params['overall_score_percentage'] = f'lte.{max_score}'
        
        qa_url = f"{SUPABASE_URL}/rest/v1/qa_evaluations"
        qa_response = requests.get(qa_url, headers=HEADERS, params=qa_params, timeout=20)
        
        if qa_response.status_code not in [200, 206]:
            st.warning(f"No QA evaluations found for {agent_name}")
            return []
        
        qa_data = qa_response.json()
        
        if not qa_data:
            st.info(f"No QA evaluations found for {agent_name}")
            return []
        
        # Get the call_log_ids that have QA evaluations
        qa_call_ids = [str(qa['call_log_id']) for qa in qa_data]
        
        # STEP 3: Get transcripts for QA-evaluated calls only
        transcripts = {}
        if qa_call_ids:
            transcript_url = f"{SUPABASE_URL}/rest/v1/transcripts"
            transcript_params = {
                "select": "call_log_id,text",
                "call_log_id": f"in.({','.join(qa_call_ids)})",
                "order": "start.asc",
                "limit": len(qa_call_ids) * 3  # Get first 3 segments per call
            }
            
            transcript_response = requests.get(transcript_url, headers=HEADERS, params=transcript_params, timeout=20)
            if transcript_response.status_code in [200, 206]:
                transcript_data = transcript_response.json()
                
                for transcript in transcript_data:
                    call_id = transcript['call_log_id']
                    if call_id not in transcripts:
                        transcripts[call_id] = []
                    transcripts[call_id].append(transcript['text'])
        
        # STEP 4: Combine data (only for calls with QA evaluations)
        combined_data = []
        for qa in qa_data:
            call_id = qa['call_log_id']
            call_log = call_logs_dict.get(call_id)
            
            if not call_log:
                continue  # Skip if call log not found
            
            call_transcript_parts = transcripts.get(call_id, [])
            full_transcript = " ".join(call_transcript_parts)[:1000] if call_transcript_parts else "No transcript available"
            
            combined_data.append({
                'call_log_id': call_id,
                'log_time': call_log.get('log_time'),
                'phone_number': call_log.get('phone_number'),
                'customer_name': call_log.get('customer_name'),
                'disposition': call_log.get('disposition'),
                'campaign_name': call_log.get('campaign_name'),
                'qa_evaluation': qa,
                'transcript': full_transcript
            })
        
        return combined_data
        
    except Exception as e:
        st.error(f"Error fetching call data: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()[:500]}")
        return []

@st.cache_data(ttl=CACHE_TTL_METRICS, show_spinner="Calculating metrics...")
def get_agent_metrics_optimized(
    agent_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    OPTIMIZED: Get comprehensive agent metrics - PRIMARY focus on QA evaluations
    """
    try:
        # Build base query for call logs
        base_params = {
            "agent_name": f"eq.{agent_name}",
            "limit": 1000
        }
        
        if start_date:
            base_params['log_time'] = f'gte.{start_date}'
            if end_date:
                base_params['and'] = f'(log_time.lt.{end_date})'
        elif end_date:
            base_params['log_time'] = f'lt.{end_date}'
        
        call_url = f"{SUPABASE_URL}/rest/v1/call_logs"
        
        # Get call_log_ids and basic info
        call_response = requests.get(
            call_url,
            headers=HEADERS,
            params={**base_params, 'select': 'call_log_id,disposition,campaign_name'},
            timeout=15
        )
        
        if call_response.status_code not in [200, 206]:
            return {}
        
        call_data = call_response.json()
        call_ids = [str(r['call_log_id']) for r in call_data]
        
        # Get disposition and campaign breakdowns
        dispositions = [r.get('disposition') for r in call_data if r.get('disposition')]
        disposition_counts = Counter(dispositions)
        campaigns = [r.get('campaign_name') for r in call_data if r.get('campaign_name')]
        campaign_counts = Counter(campaigns)
        
        # PRIMARY: Get QA evaluations for this agent
        qa_metrics = {
            'qa_evaluated_calls': 0,
            'avg_score': 0,
            'latest_score': 0,
            'score_distribution': {},
            'avg_politeness': 0,
            'avg_compliance': 0,
            'avg_enthusiasm': 0,
            'avg_clarity': 0
        }
        
        if call_ids:
            qa_url = f"{SUPABASE_URL}/rest/v1/qa_evaluations"
            
            # Get count of QA evaluations
            qa_count_response = requests.get(
                qa_url,
                headers={**HEADERS, 'Prefer': 'count=exact'},
                params={
                    'select': 'count',
                    'call_log_id': f"in.({','.join(call_ids)})"
                },
                timeout=15
            )
            
            if qa_count_response.status_code in [200, 206]:
                qa_count = int(qa_count_response.headers.get('Content-Range', '0/0').split('/')[-1])
                qa_metrics['qa_evaluated_calls'] = qa_count
            
            # Get QA data for calculations
            qa_response = requests.get(
                qa_url,
                headers=HEADERS,
                params={
                    'select': 'overall_score_percentage,politeness_score,compliance_score,enthusiasm_score,clarity_score,created_at',
                    'call_log_id': f"in.({','.join(call_ids)})",
                    'order': 'created_at.desc',
                    'limit': 500  # Get more for better statistics
                },
                timeout=15
            )
            
            if qa_response.status_code in [200, 206]:
                qa_data = qa_response.json()
                
                if qa_data:
                    scores = [float(q['overall_score_percentage']) for q in qa_data if q.get('overall_score_percentage') is not None]
                    
                    if scores:
                        qa_metrics['avg_score'] = round(sum(scores) / len(scores), 1)
                        qa_metrics['latest_score'] = round(scores[0], 1)
                        
                        # Score distribution
                        for score in scores:
                            bracket = f"{int(score//10)*10}-{int(score//10)*10+9}%"
                            qa_metrics['score_distribution'][bracket] = qa_metrics['score_distribution'].get(bracket, 0) + 1
                        
                        # Average criteria scores (convert from decimal to percentage)
                        politeness_scores = [float(q.get('politeness_score', 0) or 0) * 100 for q in qa_data]
                        compliance_scores = [float(q.get('compliance_score', 0) or 0) * 100 for q in qa_data]
                        enthusiasm_scores = [float(q.get('enthusiasm_score', 0) or 0) * 100 for q in qa_data]
                        clarity_scores = [float(q.get('clarity_score', 0) or 0) * 100 for q in qa_data]
                        
                        qa_metrics['avg_politeness'] = round(sum(politeness_scores) / len(politeness_scores), 1) if politeness_scores else 0
                        qa_metrics['avg_compliance'] = round(sum(compliance_scores) / len(compliance_scores), 1) if compliance_scores else 0
                        qa_metrics['avg_enthusiasm'] = round(sum(enthusiasm_scores) / len(enthusiasm_scores), 1) if enthusiasm_scores else 0
                        qa_metrics['avg_clarity'] = round(sum(clarity_scores) / len(clarity_scores), 1) if clarity_scores else 0
        
        return {
            'total_calls': len(call_data),
            'disposition_breakdown': dict(disposition_counts.most_common()),
            'campaign_breakdown': dict(campaign_counts.most_common()),
            **qa_metrics
        }
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()[:500]}")
        return {}

@st.cache_data(ttl=CACHE_TTL_CALL_DATA)
def get_available_campaigns(agent_name: str) -> List[str]:
    """Get list of campaigns for an agent"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/call_logs"
        params = {
            "select": "campaign_name",
            "agent_name": f"eq.{agent_name}",
            "campaign_name": "not.is.null",
            "limit": 100
        }
        
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            campaigns = list(set([r['campaign_name'] for r in response.json() if r.get('campaign_name')]))
            return sorted(campaigns)
        return []
    except:
        return []

def generate_coaching_response(agent_name: str, question: str, agent_metrics: Dict, sample_calls: List[Dict]) -> str:
    """Generate AI coaching response using OpenAI with optimized context"""
    
    # Build concise context
    context = f"""Agent: {agent_name}
Total Calls: {agent_metrics.get('total_calls', 0)}
QA Evaluated: {agent_metrics.get('qa_evaluated_calls', 0)}
Avg Score: {agent_metrics.get('avg_score', 0)}%
Latest Score: {agent_metrics.get('latest_score', 0)}%

Top Dispositions: {str(list(agent_metrics.get('disposition_breakdown', {}).items())[:5])}
Campaigns: {str(list(agent_metrics.get('campaign_breakdown', {}).keys())[:3])}

Recent Calls (Sample):
{chr(10).join([f"- Call {i+1}: Score {call.get('qa_evaluation', {}).get('overall_score_percentage', 'N/A')}%, Disposition: {call.get('disposition', 'N/A')}" for i, call in enumerate(sample_calls[:5])])}"""
    
    system_prompt = f"""You are an expert QA coach for LandAI real estate cold calling. Provide specific, actionable coaching advice based on the data.

{context}

Keep responses concise, specific, and actionable."""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=800  # Reduced for faster responses
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error generating the coaching response: {e}"

def main():
    st.set_page_config(
        page_title="LandAI QA Coaching Bot",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better dark theme UI
    st.markdown("""
    <style>
    /* Main metrics cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #B0B0B0;
        font-weight: 500;
    }
    
    /* Metric containers - dark background */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #3A3A3A;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Column containers */
    [data-testid="column"] {
        padding: 0.25rem;
    }
    
    /* Section headers */
    h4 {
        color: #FAFAFA !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Disposition and campaign sections */
    .element-container {
        color: #E0E0E0;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #2D2D2D;
        border: 1px solid #3A3A3A;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3A3A3A 0%, #2D2D2D 100%);
        color: #FAFAFA;
        border: 1px solid #4A4A4A;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #4A4A4A 0%, #3A3A3A 100%);
        border-color: #5A5A5A;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    /* Primary button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #FF4B4B 0%, #E63946 100%);
        border-color: #FF4B4B;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4B4B 100%);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    
    /* Divider */
    hr {
        border-color: #3A3A3A;
        margin: 2rem 0;
    }
    
    /* Success/info boxes */
    [data-testid="stSuccess"], [data-testid="stInfo"] {
        background-color: #2D2D2D;
        border: 1px solid #3A3A3A;
        border-radius: 8px;
    }
    
    /* Input fields */
    input, textarea, select {
        background-color: #2D2D2D !important;
        color: #FAFAFA !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 8px !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background-color: #3A3A3A;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 LandAI QA Coaching Bot")
    st.markdown("*AI-powered coaching insights with advanced filtering*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = None
    if 'agent_metrics' not in st.session_state:
        st.session_state.agent_metrics = {}
    if 'call_data' not in st.session_state:
        st.session_state.call_data = []
    
    # Sidebar - Agent Selection & Filters
    with st.sidebar:
        st.header("🎯 Agent Selection")
        
        # Fetch top agents
        agents_data = get_all_agents_optimized()
        
        if agents_data:
            st.success(f"Found {len(agents_data)} agents")
            
            # Show ALL agents sorted alphabetically
            # Just show agent names (no call counts for faster loading)
            agent_options = [""] + [a['name'] for a in agents_data]
            agent_names = [""] + [a['name'] for a in agents_data]
            
            selected_display = st.selectbox(
                "Select an agent:",
                options=agent_options,
                index=0,
                help="Top agents by call volume"
            )
            
            selected_agent = agent_names[agent_options.index(selected_display)] if selected_display else None
            
            if selected_agent and selected_agent != st.session_state.selected_agent:
                st.session_state.selected_agent = selected_agent
                st.session_state.messages = []
                st.rerun()
            
            # Advanced Filters (only show if agent selected)
            if st.session_state.selected_agent:
                st.divider()
                st.header("🔍 Filters")
                
                # Date filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=None,
                        help="Filter calls from this date"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=None,
                        help="Filter calls until this date"
                    )
                
                # Campaign filter
                available_campaigns = get_available_campaigns(st.session_state.selected_agent)
                selected_campaign = st.selectbox(
                    "Campaign",
                    options=["All Campaigns"] + available_campaigns,
                    help="Filter by campaign"
                )
                
                # QA Score filter
                st.subheader("QA Score Range")
                score_range = st.slider(
                    "Score Range (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=5,
                    help="Filter calls by QA score"
                )
                
                # Apply filters button
                if st.button("🔄 Apply Filters", type="primary"):
                    with st.spinner("Applying filters..."):
                        # Convert dates to ISO format
                        start_date_str = start_date.isoformat() if start_date else None
                        end_date_str = end_date.isoformat() if end_date else None
                        
                        # Clear cache for this agent to force reload
                        cache_key_call_data = f"get_agent_call_data_optimized_{st.session_state.selected_agent}"
                        cache_key_metrics = f"get_agent_metrics_optimized_{st.session_state.selected_agent}"
                        
                        # Force reload data with filters (bypass cache)
                        st.session_state.call_data = get_agent_call_data_optimized(
                            st.session_state.selected_agent,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            campaign=selected_campaign if selected_campaign != "All Campaigns" else None,
                            min_score=score_range[0] if score_range != (0, 100) else None,
                            max_score=score_range[1] if score_range != (0, 100) else None,
                            limit=20
                        )
                        
                        st.session_state.agent_metrics = get_agent_metrics_optimized(
                            st.session_state.selected_agent,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )
                        
                        # Store filter state for display
                        st.session_state.current_filters = {
                            'start_date': start_date_str,
                            'end_date': end_date_str,
                            'campaign': selected_campaign,
                            'score_range': score_range
                        }
                    
                    st.success("✅ Filters applied!")
                    st.rerun()
                
                # Clear filters
                if st.button("🗑️ Clear Filters"):
                    # Clear filter state
                    if 'current_filters' in st.session_state:
                        del st.session_state.current_filters
                    
                    # Reload data without filters
                    st.session_state.call_data = get_agent_call_data_optimized(
                        st.session_state.selected_agent,
                        limit=20
                    )
                    st.session_state.agent_metrics = get_agent_metrics_optimized(
                        st.session_state.selected_agent
                    )
                    st.success("✅ Filters cleared!")
                    st.rerun()
                
                # Show active filters
                if 'current_filters' in st.session_state and st.session_state.current_filters:
                    st.markdown("---")
                    st.markdown("**🔍 Active Filters:**")
                    filters = st.session_state.current_filters
                    
                    if filters.get('start_date') or filters.get('end_date'):
                        date_range = f"{filters.get('start_date', 'Any')} to {filters.get('end_date', 'Any')}"
                        st.write(f"📅 Date: {date_range}")
                    
                    if filters.get('campaign') and filters['campaign'] != "All Campaigns":
                        st.write(f"🎯 Campaign: {filters['campaign']}")
                    
                    if filters.get('score_range') and filters['score_range'] != (0, 100):
                        st.write(f"📊 Score: {filters['score_range'][0]}% - {filters['score_range'][1]}%")
        else:
            st.error("No agents found in database")
    
    # Main content
    if st.session_state.selected_agent:
        agent_name = st.session_state.selected_agent
        
        # Load initial data if not loaded
        if not st.session_state.call_data:
            with st.spinner(f"Loading data for {agent_name}..."):
                st.session_state.call_data = get_agent_call_data_optimized(agent_name, limit=20)
                st.session_state.agent_metrics = get_agent_metrics_optimized(agent_name)
        
        metrics = st.session_state.agent_metrics
        
        # Enhanced Metrics Dashboard
        dashboard_title = f"📊 Performance Dashboard - {agent_name}"
        
        # Add filter indicator if filters are active
        if 'current_filters' in st.session_state and st.session_state.current_filters:
            dashboard_title += " 🔍 (Filtered)"
        
        st.subheader(dashboard_title)
        
        # Row 1: Core metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Calls", f"{metrics.get('total_calls', 0):,}")
        with col2:
            st.metric("QA Evaluated", f"{metrics.get('qa_evaluated_calls', 0):,}")
        with col3:
            st.metric("Avg Score", f"{metrics.get('avg_score', 0)}%")
        with col4:
            st.metric("Latest Score", f"{metrics.get('latest_score', 0)}%")
        with col5:
            qa_rate = (metrics.get('qa_evaluated_calls', 0) / metrics.get('total_calls', 1)) * 100 if metrics.get('total_calls', 0) > 0 else 0
            st.metric("QA Coverage", f"{qa_rate:.1f}%")
        
        # Row 2: Criteria scores
        if metrics.get('qa_evaluated_calls', 0) > 0:
            st.markdown("#### 📈 QA Criteria Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Politeness", f"{metrics.get('avg_politeness', 0)}%")
            with col2:
                st.metric("Compliance", f"{metrics.get('avg_compliance', 0)}%")
            with col3:
                st.metric("Enthusiasm", f"{metrics.get('avg_enthusiasm', 0)}%")
            with col4:
                st.metric("Clarity", f"{metrics.get('avg_clarity', 0)}%")
        
        # Row 3: Disposition & Campaign breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📞 Top Dispositions")
            disposition_breakdown = metrics.get('disposition_breakdown', {})
            if disposition_breakdown:
                for disp, count in list(disposition_breakdown.items())[:5]:
                    percentage = (count / metrics.get('total_calls', 1)) * 100
                    st.write(f"**{disp}**: {count:,} calls ({percentage:.1f}%)")
            else:
                st.write("No disposition data available")
        
        with col2:
            st.markdown("#### 🎯 Campaign Performance")
            campaign_breakdown = metrics.get('campaign_breakdown', {})
            if campaign_breakdown:
                for campaign, count in list(campaign_breakdown.items())[:5]:
                    percentage = (count / metrics.get('total_calls', 1)) * 100
                    st.write(f"**{campaign}**: {count:,} calls ({percentage:.1f}%)")
            else:
                st.write("No campaign data available")
        
        st.divider()
        
        # Chat interface
        st.subheader(f"💬 Chat with QA Coach")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Ask me anything about {agent_name}'s performance..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing and generating insights..."):
                    # Get 10 random sample calls that have QA evaluations for context
                    calls_with_qa = [call for call in st.session_state.call_data if call.get('qa_evaluation')]
                    sample_calls = random.sample(calls_with_qa, min(10, len(calls_with_qa))) if calls_with_qa else []
                    
                    response = generate_coaching_response(
                        agent_name,
                        prompt,
                        metrics,
                        sample_calls
                    )
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Suggested questions
        st.subheader("💡 Quick Actions")
        
        suggestions = [
            f"What are {agent_name}'s top 3 strengths?",
            f"What should {agent_name} focus on improving?",
            f"Analyze {agent_name}'s disposition patterns",
            f"How can {agent_name} improve their QA scores?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            if col.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                
                # Get 10 random sample calls that have QA evaluations for context
                calls_with_qa = [call for call in st.session_state.call_data if call.get('qa_evaluation')]
                sample_calls = random.sample(calls_with_qa, min(10, len(calls_with_qa))) if calls_with_qa else []
                
                response = generate_coaching_response(
                    agent_name,
                    suggestion,
                    metrics,
                    sample_calls
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to LandAI QA Coaching Bot! 🎯
        
        **Select an agent from the sidebar to get started.**
        
        ### ⚡ New Features:
        
        - **🚀 Lightning Fast**: Optimized queries load in 2-3 seconds
        - **📊 Top 10 Agents**: See top performers by call volume
        - **🔍 Advanced Filters**: Filter by date, campaign, and QA score
        - **📈 Enhanced Metrics**: Comprehensive performance dashboard
        - **💾 Smart Caching**: Data loads instantly on subsequent visits
        - **🎯 Pagination**: Efficient data loading
        
        ### What can this bot do?
        
        - 📊 **Performance Analysis**: View comprehensive metrics
        - 💬 **AI Coaching**: Get personalized coaching advice
        - 🔍 **Custom Filters**: Analyze specific time periods and campaigns
        - 📈 **Data-Driven Insights**: Based on real QA evaluations
        
        **Select an agent to get started!**
        """)

if __name__ == "__main__":
    main()

