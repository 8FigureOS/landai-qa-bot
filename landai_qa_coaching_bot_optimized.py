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

# Configuration - Load from Streamlit Secrets
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

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
CACHE_TTL_AGENTS = 300  # 5 minutes (reduced from 1 hour to refresh faster)
CACHE_TTL_CALL_DATA = 900  # 15 minutes
CACHE_TTL_METRICS = 300  # 5 minutes

@st.cache_data(ttl=CACHE_TTL_AGENTS, show_spinner=False)
def get_all_agents_optimized() -> List[Dict[str, any]]:
    """
    OPTIMIZED: Fetch all unique agents quickly (without call counts initially)
    Uses DISTINCT query for better performance
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/call_logs"
        
        # Use a more efficient approach: Get distinct agent names directly
        # First, check if there are any records at all
        count_response = requests.get(
            url,
            headers={**HEADERS, 'Prefer': 'count=exact'},
            params={'select': 'count', 'limit': 1},
            timeout=10
        )
        
        # If no records exist, return empty list immediately
        if count_response.status_code in [200, 206]:
            total_count = count_response.headers.get('Content-Range', '0/0').split('/')[-1]
            if total_count == '0' or not total_count.isdigit() or int(total_count) == 0:
                return []
        
        # Strategy: Scan through records to find unique agents
        seen_agents = set()
        offset = 0
        limit = 1000
        max_records = 150000  # Scan entire database
        
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
                    
                    # If we got less than limit, we've reached the end
                    if len(data) < limit:
                        break
                    
                    offset += limit
                elif response.status_code == 404:
                    # Table doesn't exist or no data
                    return []
                else:
                    # API error - use agents discovered so far
                    break
                    
            except requests.exceptions.ConnectionError:
                # Connection error - return empty if no agents found yet
                return []
            except requests.exceptions.Timeout:
                # Timeout - use agents discovered so far
                break
            except Exception:
                # Other error - use agents discovered so far
                break
        
        if not seen_agents:
            return []
        
        # Filter out specific agents to exclude
        excluded_agents = {'Charlie', 'Quality Assurance', 'Rose'}
        filtered_agents = seen_agents - excluded_agents
        
        # Return agents sorted alphabetically (no call counts)
        agents_list = [{'name': agent, 'total_calls': 0} for agent in sorted(filtered_agents)]
        
        return agents_list
        
    except Exception:
        return []

@st.cache_data(ttl=CACHE_TTL_CALL_DATA, show_spinner="Loading call data...")
def get_agent_call_data_optimized(
    agent_name: str, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    campaign: Optional[str] = None,
    disposition: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = 20
) -> List[Dict]:
    """
    OPTIMIZED: Fetch call data starting from QA evaluations (only calls with QA scores)
    """
    try:
        # STEP 1: Get call_log_ids for this agent with filters
        # Normalize agent_name to ensure exact match
        normalized_agent_name = agent_name.strip()
        call_params = {
            "select": "call_log_id,log_time,phone_number,customer_name,disposition,campaign_name,recording_url,agent_name",
            "agent_name": f"eq.{normalized_agent_name}",
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
        
        # Apply disposition filter
        if disposition and disposition != "All Dispositions":
            call_params['disposition'] = f'eq.{disposition}'
        
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
        # CRITICAL: Filter to ensure ONLY calls for the selected agent are returned
        combined_data = []
        
        for qa in qa_data:
            call_id = qa['call_log_id']
            call_log = call_logs_dict.get(call_id)
            
            if not call_log:
                continue  # Skip if call log not found
            
            # CRITICAL FILTER: Verify agent_name matches exactly (case-insensitive, trimmed)
            call_agent_name = call_log.get('agent_name', '').strip()
            if call_agent_name.lower() != normalized_agent_name.lower():
                # Skip this call - agent mismatch (data inconsistency protection)
                continue
            
            call_transcript_parts = transcripts.get(call_id, [])
            full_transcript = " ".join(call_transcript_parts)[:1000] if call_transcript_parts else "No transcript available"
            
            combined_data.append({
                'call_log_id': call_id,
                'agent_name': call_agent_name,  # Include agent_name for display
                'log_time': call_log.get('log_time'),
                'phone_number': call_log.get('phone_number'),
                'customer_name': call_log.get('customer_name'),
                'disposition': call_log.get('disposition'),
                'campaign_name': call_log.get('campaign_name'),
                'recording_url': call_log.get('recording_url'),
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
        # Normalize agent_name to ensure exact match
        normalized_agent_name = agent_name.strip()
        base_params = {
            "agent_name": f"eq.{normalized_agent_name}",
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

@st.cache_data(ttl=CACHE_TTL_CALL_DATA)
def get_available_dispositions(agent_name: str) -> List[str]:
    """Get list of unique dispositions for an agent"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/call_logs"
        params = {
            "select": "disposition",
            "agent_name": f"eq.{agent_name}",
            "disposition": "not.is.null",
            "limit": 100
        }
        
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            dispositions = list(set([r['disposition'] for r in response.json() if r.get('disposition')]))
            return sorted(dispositions)
        return []
    except:
        return []

@st.cache_data(ttl=CACHE_TTL_METRICS, show_spinner=False)
def get_overall_statistics() -> Dict:
    """Get overall statistics for all agents (excluding Charlie, Quality Assurance, Rose)"""
    try:
        # Excluded agents
        excluded_agents = ['Charlie', 'Quality Assurance', 'Rose']
        
        # Get total calls (excluding specific agents)
        call_count_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/call_logs",
            headers={**HEADERS, 'Prefer': 'count=exact'},
            params={
                'select': 'count',
                'agent_name': f'not.in.({",".join(excluded_agents)})'
            },
            timeout=15
        )
        total_calls = int(call_count_response.headers.get('Content-Range', '0/0').split('/')[-1]) if call_count_response.status_code in [200, 206] else 0
        
        # Get call_log_ids for excluded agents to filter them out from QA stats
        excluded_call_ids = []
        try:
            excluded_calls_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/call_logs",
                headers=HEADERS,
                params={
                    'select': 'call_log_id',
                    'agent_name': f'in.({",".join(excluded_agents)})',
                    'limit': 5000
                },
                timeout=10
            )
            if excluded_calls_response.status_code in [200, 206]:
                excluded_call_ids = [str(c['call_log_id']) for c in excluded_calls_response.json()]
        except:
            pass
        
        # Get QA evaluations (excluding specific agents)
        qa_params = {'select': 'count'}
        if excluded_call_ids:
            qa_params['call_log_id'] = f'not.in.({",".join(excluded_call_ids)})'
        
        qa_count_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/qa_evaluations",
            headers={**HEADERS, 'Prefer': 'count=exact'},
            params=qa_params,
            timeout=15
        )
        total_qa = int(qa_count_response.headers.get('Content-Range', '0/0').split('/')[-1]) if qa_count_response.status_code in [200, 206] else 0
        
        # Get average scores (excluding specific agents)
        qa_score_params = {'select': 'overall_score_percentage', 'limit': 1000}
        if excluded_call_ids:
            qa_score_params['call_log_id'] = f'not.in.({",".join(excluded_call_ids[:500])})'  # Limit to avoid URL length issues
        
        qa_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/qa_evaluations",
            headers=HEADERS,
            params=qa_score_params,
            timeout=15
        )
        
        avg_score = 0
        if qa_response.status_code in [200, 206]:
            scores = [float(q['overall_score_percentage']) for q in qa_response.json() if q.get('overall_score_percentage')]
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        
        # Get unique agents count
        agents = get_all_agents_optimized()
        total_agents = len(agents)
        
        return {
            'total_calls': total_calls,
            'total_qa_evaluations': total_qa,
            'avg_qa_score': avg_score,
            'total_agents': total_agents,
            'qa_coverage': round((total_qa / total_calls * 100), 1) if total_calls > 0 else 0
        }
    except Exception as e:
        return {
            'total_calls': 0,
            'total_qa_evaluations': 0,
            'avg_qa_score': 0,
            'total_agents': 0,
            'qa_coverage': 0
        }

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

def text_to_sql_query(question: str) -> str:
    """Convert natural language question to SQL query"""
    
    # Database schema
    schema = """
    Database Schema (readymode_data):
    
    1. call_logs table:
       - call_log_id (integer, primary key)
       - agent_name (text)
       - log_time (timestamp)
       - campaign_name (text)
       - disposition (text)
       - log_type (text)
       - phone_number (text)
       - customer_name (text)
    
    2. qa_evaluations table:
       - call_log_id (integer, foreign key to call_logs)
       - overall_score_percentage (numeric)
       - politeness_score (numeric)
       - compliance_score (numeric)
       - enthusiasm_score (numeric)
       - clarity_score (numeric)
       - grammar_score (numeric)
       - simplicity_score (numeric)
       - dispositions_score (numeric)
       - documentation_score (numeric)
       - info_verified_score (numeric)
       - created_at (timestamp)
    
    3. transcripts table:
       - call_log_id (integer, foreign key to call_logs)
       - speaker (text)
       - text (text)
       - start (numeric)
       - end_time (numeric)
       - confidence (numeric)
    """
    
    system_prompt = f"""You are a SQL expert. Convert the user's question into a valid PostgreSQL query.

{schema}

CRITICAL RULES:
1. **ALWAYS start queries from the qa_evaluations table first** - only QA-evaluated calls should be included
2. Join to call_logs ONLY to get additional fields like agent_name, log_time, campaign_name
3. Return ONLY the SQL query, no explanations
4. Use proper PostgreSQL syntax
5. Use aliases for readability (q for qa_evaluations, c for call_logs)
6. Limit results to 10 unless asked otherwise
7. For date ranges, use log_time column from call_logs after joining
8. For "this week", use: WHERE c.log_time >= date_trunc('week', CURRENT_DATE)
9. For "last 7 days", use: WHERE c.log_time >= CURRENT_DATE - INTERVAL '7 days'
10. Always use readymode_data schema prefix (e.g., readymode_data.qa_evaluations)

Examples:
Question: "Top performing agent this week"
SQL: SELECT c.agent_name, AVG(q.overall_score_percentage) as avg_score, COUNT(*) as call_count FROM readymode_data.qa_evaluations q JOIN readymode_data.call_logs c ON q.call_log_id = c.call_log_id WHERE c.log_time >= date_trunc('week', CURRENT_DATE) GROUP BY c.agent_name ORDER BY avg_score DESC LIMIT 10;

Question: "Agents with score above 90%"
SQL: SELECT c.agent_name, AVG(q.overall_score_percentage) as avg_score, COUNT(*) as calls FROM readymode_data.qa_evaluations q JOIN readymode_data.call_logs c ON q.call_log_id = c.call_log_id GROUP BY c.agent_name HAVING AVG(q.overall_score_percentage) > 90 ORDER BY avg_score DESC LIMIT 10;

Question: "Which campaigns had the most QA evaluated calls?"
SQL: SELECT c.campaign_name, COUNT(*) as qa_eval_count, AVG(q.overall_score_percentage) as avg_score FROM readymode_data.qa_evaluations q JOIN readymode_data.call_logs c ON q.call_log_id = c.call_log_id GROUP BY c.campaign_name ORDER BY qa_eval_count DESC LIMIT 10;
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,  # More deterministic for SQL
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return sql_query
    except Exception as e:
        return f"Error generating SQL: {e}"

def execute_sql_query(sql_query: str) -> Tuple[bool, any]:
    """Execute SQL query against Supabase using REST API"""
    try:
        import re
        from datetime import datetime, timedelta
        
        # Extract table name, conditions, and aggregations from SQL
        sql_lower = sql_query.lower()
        
        # For now, we'll support common aggregate queries
        if "avg(q.overall_score_percentage)" in sql_lower and "group by" in sql_lower:
            # This is an aggregate query - we need to fetch data and compute in Python
            
            # ALWAYS START FROM qa_evaluations table first (only QA-evaluated calls)
            qa_url = f"{SUPABASE_URL}/rest/v1/qa_evaluations"
            qa_params = {
                "select": "call_log_id,overall_score_percentage",
                "limit": "5000"  # Get up to 5000 QA evaluations
            }
            
            qa_response = requests.get(qa_url, headers=HEADERS, params=qa_params, timeout=30)
            
            if qa_response.status_code not in [200, 206]:
                return False, f"Error fetching QA evaluations: {qa_response.status_code} - {qa_response.text[:200]}"
            
            qa_evals = qa_response.json()
            
            # Debug: check if we have QA evaluations
            if not qa_evals:
                return True, {
                    "debug": "No QA evaluations found in database",
                    "sql": sql_query,
                    "note": "The qa_evaluations table appears to be empty"
                }
            
            # Get the call_log_ids from QA evaluations
            call_ids = [str(q['call_log_id']) for q in qa_evals]
            
            # Now get call_logs for these QA-evaluated calls only
            url = f"{SUPABASE_URL}/rest/v1/call_logs"
            params = {
                "select": "call_log_id,agent_name,log_time,campaign_name,disposition",
                "call_log_id": f"in.({','.join(call_ids[:1000])})"  # Limit to 1000 for URL length
            }
            
            # Add date filter if present - but let's make it flexible
            # Instead of filtering for "this week" only, let's get recent data
            if "date_trunc('week'" in sql_lower or "current_date" in sql_lower or "where" in sql_lower:
                # Get data from the last 30 days instead of just this week
                thirty_days_ago = datetime.now() - timedelta(days=30)
                params["log_time"] = f"gte.{thirty_days_ago.isoformat()}"
            
            call_response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            
            if call_response.status_code not in [200, 206]:
                return False, f"Error fetching call logs: {call_response.status_code} - {call_response.text[:200]}"
            
            calls = call_response.json()
            
            # Debug info
            if not calls:
                return True, {
                    "debug": f"Found {len(qa_evals)} QA evaluations but no matching call logs",
                    "qa_count": len(qa_evals),
                    "sample_qa_ids": call_ids[:5]
                }
            
            # Create a mapping of call_log_id to QA score
            qa_map = {q['call_log_id']: q['overall_score_percentage'] for q in qa_evals}
            
            # Aggregate by agent
            from collections import defaultdict
            agent_stats = defaultdict(lambda: {'scores': [], 'count': 0})
            
            for call in calls:
                call_id = call['call_log_id']
                if call_id in qa_map:
                    agent = call['agent_name']
                    score = qa_map[call_id]
                    agent_stats[agent]['scores'].append(score)
                    agent_stats[agent]['count'] += 1
            
            # Calculate averages and format results
            results = []
            for agent, stats in agent_stats.items():
                if stats['scores']:
                    avg_score = sum(stats['scores']) / len(stats['scores'])
                    results.append({
                        'agent_name': agent,
                        'avg_score': round(avg_score, 2),
                        'call_count': stats['count']
                    })
            
            # Check if we have results
            if not results:
                return True, {
                    "debug": "Query executed but no matching results",
                    "info": f"Found {len(calls)} calls, {len(qa_evals)} QA evaluations, but couldn't match them by agent",
                    "agents_in_calls": list(set([c.get('agent_name') for c in calls[:10]])),
                    "qa_eval_count": len(qa_evals)
                }
            
            # Sort by avg_score descending
            results.sort(key=lambda x: x['avg_score'], reverse=True)
            
            # Limit results
            if "limit" in sql_lower:
                limit_match = re.search(r'limit\s+(\d+)', sql_lower)
                if limit_match:
                    limit = int(limit_match.group(1))
                    results = results[:limit]
            else:
                results = results[:10]
            
            return True, results
        
        return False, "Query type not yet supported for direct execution"
        
    except Exception as e:
        import traceback
        return False, f"Error executing query: {str(e)}\n{traceback.format_exc()[:500]}"

def interpret_sql_results(question: str, sql_query: str, results: list) -> str:
    """Use LLM to interpret SQL results and provide a natural language answer"""
    
    if not results:
        return "No results found for your query."
    
    # Format results for LLM
    results_text = json.dumps(results, indent=2)
    
    system_prompt = """You are a data analyst. The user asked a question, we generated SQL and executed it.
Now provide a clear, concise answer to their question based on the results.

Format your response as:
1. Direct answer to their question
2. Key findings (bullet points)
3. Any relevant insights

Keep it conversational and easy to understand."""
    
    user_prompt = f"""Question: {question}

SQL Query:
{sql_query}

Results:
{results_text}

Please provide a clear, natural language answer to the user's question based on these results."""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple formatting
        return f"**Results:**\n\n" + "\n".join([f"- {r}" for r in results[:10]])

def check_authentication():
    """Check if user is authenticated"""
    # Define authorized users (username: password)
    AUTHORIZED_USERS = {
        "landai-admin": "Merxh!en4Lifn",
        "qa-manager": "QA@Manager2025!",
        "coach-lead": "Coach#Lead2025",
        "supervisor": "Super@Visor2025",
        "maria": "Merxh!en4LifnLAND"
    }
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
    
    if not st.session_state.authenticated:
        # Show login form
        st.title("🔐 LandAI QA Coaching Bot - Login")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", type="primary")
            
            if submit:
                # Check credentials
                if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome, {username}! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.caption("🔒 Secure access required")
        return False
    
    return True

def main():
    st.set_page_config(
        page_title="LandAI QA Coaching Bot",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check authentication first
    if not check_authentication():
        return
    
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
        # Show logged-in user and logout button at top
        st.markdown("---")
        if 'username' in st.session_state and st.session_state.username:
            st.caption(f"👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.markdown("---")
        
        st.header("🎯 Agent Selection")
        
        # Add cache clear button for debugging
        if st.button("🔄 Clear Cache & Refresh", help="Clear Streamlit cache and refresh agent list"):
            st.cache_data.clear()
            st.success("Cache cleared! Refreshing...")
            st.rerun()
        
        # Fetch top agents
        agents_data = get_all_agents_optimized()
        
        if agents_data:
            st.caption(f"📋 {len(agents_data)} agents available")
            
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
                
                # Disposition filter
                available_dispositions = get_available_dispositions(st.session_state.selected_agent)
                selected_disposition = st.selectbox(
                    "Disposition",
                    options=["All Dispositions"] + available_dispositions,
                    help="Filter by call disposition (e.g., live transfer, not interested, etc.)"
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
                            disposition=selected_disposition if selected_disposition != "All Dispositions" else None,
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
                            'disposition': selected_disposition,
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
                    
                    if filters.get('disposition') and filters['disposition'] != "All Dispositions":
                        st.write(f"📞 Disposition: {filters['disposition']}")
                    
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
        
        # Recent Calls with Recordings Section
        st.subheader("🎧 Recent Calls with Recordings")
        st.markdown("*Listen to actual calls to understand the QA scores in context*")
        
        if st.session_state.call_data:
            # Filter calls that have QA evaluations and recordings
            calls_with_recordings = [
                call for call in st.session_state.call_data 
                if call.get('qa_evaluation') and call.get('recording_url')
            ]
            
            if calls_with_recordings:
                # Show top 10 most recent calls
                display_calls = calls_with_recordings[:10]
                
                for idx, call in enumerate(display_calls, 1):
                    qa_eval = call.get('qa_evaluation', {})
                    score = qa_eval.get('overall_score_percentage', 'N/A')
                    recording_url = call.get('recording_url', '')
                    log_time = call.get('log_time', '')
                    disposition = call.get('disposition', 'N/A')
                    campaign = call.get('campaign_name', 'N/A')
                    customer = call.get('customer_name', 'N/A')
                    
                    # Format date if available
                    date_str = ""
                    if log_time:
                        try:
                            from datetime import datetime
                            if isinstance(log_time, str):
                                dt = datetime.fromisoformat(log_time.replace('Z', '+00:00'))
                                date_str = dt.strftime("%Y-%m-%d %H:%M")
                            else:
                                date_str = str(log_time)
                        except:
                            date_str = str(log_time)[:16] if log_time else ""
                    
                    # Create a card-like display for each call
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                        
                        with col1:
                            st.markdown(f"**Call #{idx}**")
                            if date_str:
                                st.caption(f"📅 {date_str}")
                            if customer and customer != 'N/A':
                                st.caption(f"👤 {customer}")
                        
                        with col2:
                            # Score with color coding
                            if isinstance(score, (int, float)):
                                if score >= 90:
                                    score_color = "🟢"
                                elif score >= 75:
                                    score_color = "🟡"
                                else:
                                    score_color = "🔴"
                                st.markdown(f"**{score_color} Score: {score}%**")
                            else:
                                st.markdown(f"**Score: {score}**")
                            
                            st.caption(f"📞 {disposition}")
                            if campaign and campaign != 'N/A':
                                st.caption(f"🎯 {campaign}")
                        
                        with col3:
                            # QA Criteria breakdown (if available)
                            if qa_eval:
                                criteria = []
                                if qa_eval.get('politeness_score'):
                                    pol = float(qa_eval.get('politeness_score', 0) or 0) * 100
                                    criteria.append(f"Politeness: {pol:.0f}%")
                                if qa_eval.get('clarity_score'):
                                    clar = float(qa_eval.get('clarity_score', 0) or 0) * 100
                                    criteria.append(f"Clarity: {clar:.0f}%")
                                
                                if criteria:
                                    st.caption(" | ".join(criteria))
                        
                        with col4:
                            # Recording link button
                            if recording_url:
                                # Use markdown link that opens in new tab
                                st.markdown(f'<a href="{recording_url}" target="_blank" style="display: inline-block; text-decoration: none; padding: 0.5rem 1rem; background: linear-gradient(135deg, #FF4B4B 0%, #E63946 100%); color: white; border-radius: 8px; font-weight: 600; text-align: center;">🎧 Listen</a>', unsafe_allow_html=True)
                            else:
                                st.caption("❌ No recording")
                        
                        st.markdown("---")
                
                # Show additional calls in expander if more than 10
                if len(calls_with_recordings) > 10:
                    st.caption(f"*Showing 10 of {len(calls_with_recordings)} calls with recordings*")
                    with st.expander(f"📋 View All {len(calls_with_recordings)} Calls"):
                        for idx, call in enumerate(calls_with_recordings[10:], 11):
                            qa_eval = call.get('qa_evaluation', {})
                            score = qa_eval.get('overall_score_percentage', 'N/A')
                            recording_url = call.get('recording_url', '')
                            log_time = call.get('log_time', '')
                            disposition = call.get('disposition', 'N/A')
                            
                            date_str = ""
                            if log_time:
                                try:
                                    from datetime import datetime
                                    if isinstance(log_time, str):
                                        dt = datetime.fromisoformat(log_time.replace('Z', '+00:00'))
                                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                                    else:
                                        date_str = str(log_time)
                                except:
                                    date_str = str(log_time)[:16] if log_time else ""
                            
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"**Call #{idx}** - {date_str} | {disposition}")
                            with col2:
                                if isinstance(score, (int, float)):
                                    st.write(f"Score: **{score}%**")
                                else:
                                    st.write(f"Score: {score}")
                            with col3:
                                if recording_url:
                                    st.markdown(f'<a href="{recording_url}" target="_blank" style="text-decoration: none; color: #FF4B4B; font-weight: 600;">🎧 Listen</a>', unsafe_allow_html=True)
                                else:
                                    st.caption("No recording")
                            st.markdown("---")
            else:
                st.info("No calls with both QA evaluations and recordings found.")
        else:
            st.info("No call data available. Please apply filters or select a different agent.")
        
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
        # Welcome screen with overall statistics
        st.markdown("""
        ## Welcome to LandAI QA Coaching Bot! 🎯
        
        **Select an agent from the sidebar to get started with personalized coaching insights.**
        """)
        
        st.divider()
        
        # Get and display overall statistics
        st.subheader("📊 Overall Performance Dashboard")
        
        # Load statistics silently (no spinner)
        overall_stats = get_overall_statistics()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Agents", f"{overall_stats['total_agents']:,}")
        with col2:
            st.metric("Total Calls", f"{overall_stats['total_calls']:,}")
        with col3:
            st.metric("QA Evaluations", f"{overall_stats['total_qa_evaluations']:,}")
        with col4:
            st.metric("Avg QA Score", f"{overall_stats['avg_qa_score']}%")
        with col5:
            st.metric("QA Coverage", f"{overall_stats['qa_coverage']}%")
        
        st.divider()
        
        # Natural Language Query Feature
        st.subheader("🔍 Ask Questions About Your Data")
        st.markdown("*Ask questions in plain English and get instant insights from your database*")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - "Who is the top performing agent this week?"
            - "Show me agents with scores above 90%"
            - "Which campaigns had the most calls last week?"
            - "List agents with the highest politeness scores"
            - "Show me call volume by disposition type"
            """)
        
        nl_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., Top performing agent this week",
            key="nl_query"
        )
        
        if st.button("🔍 Search", type="primary") and nl_question:
            with st.spinner("🤖 Analyzing your question..."):
                # Step 1: Generate SQL
                sql_query = text_to_sql_query(nl_question)
                
                if "Error" in sql_query:
                    st.error(f"❌ {sql_query}")
                else:
                    # Show generated SQL in expander
                    with st.expander("📝 Generated SQL Query"):
                        st.code(sql_query, language="sql")
                    
                    # Step 2: Execute the query
                    with st.spinner("⚡ Executing query and fetching results..."):
                        success, results = execute_sql_query(sql_query)
                    
                    if success:
                        # Step 3: Interpret results with LLM
                        with st.spinner("💭 Analyzing results..."):
                            interpretation = interpret_sql_results(nl_question, sql_query, results)
                        
                        # Display the AI's interpretation
                        st.markdown("### 💡 Answer:")
                        st.markdown(interpretation)
                        
                        # Show raw data in expander
                        with st.expander("📊 View Raw Data"):
                            if results:
                                import pandas as pd
                                df = pd.DataFrame(results)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("No results found")
                        
                        st.success("✅ Query executed successfully!")
                    else:
                        st.error(f"❌ Query execution failed: {results}")
                        st.info("**💡 You can still copy the SQL above and run it in Supabase SQL Editor**")
        
        st.divider()
        
        # What can this bot do
        st.markdown("""
        ### 🎯 What can this bot do?
        
        - **📊 Performance Analysis**: View detailed metrics for each agent
        - **💬 AI Coaching**: Get personalized, data-driven coaching advice
        - **🔍 Custom Filters**: Analyze specific time periods, campaigns, and score ranges
        - **📈 Real-time Insights**: Based on actual QA evaluations and call data
        - **🔍 Natural Language Queries**: Ask questions in plain English (see above)
        
        **👈 Select an agent from the sidebar to begin!**
        """)

if __name__ == "__main__":
    main()

