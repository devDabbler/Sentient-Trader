"""
Supabase Client

Handles the connection and interaction with the Supabase backend.
"""

import os
from supabase import create_client, Client
from loguru import logger


def get_supabase_client() -> Client:
    """
    Initializes and returns the Supabase client.

    Reads connection details from:
    1. Streamlit secrets (preferred for cloud deployment)
    2. Environment variables (for local development)
    
    Streamlit secrets format:
    [supabase]
    url = "https://your-project.supabase.co"
    service_key = "your-service-key"
    
    Environment variables:
    - SUPABASE_URL: The project URL for your Supabase instance.
    - SUPABASE_SERVICE_KEY: The service role key for backend access.

    Returns:
        An initialized Supabase client instance, or None if credentials are not set.
    """
    supabase_url = None
    supabase_key = None
    
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'supabase' in st.secrets:
            supabase_url = st.secrets['supabase']['url']
            supabase_key = st.secrets['supabase']['service_key']
            logger.info("Using Supabase credentials from Streamlit secrets")
    except Exception:
        # Silently fall back to environment variables
        pass
    
    # Fallback to environment variables (for local development)
    if not supabase_url or not supabase_key:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        if supabase_url and supabase_key:
            logger.info("Using Supabase credentials from environment variables")

    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found in Streamlit secrets or environment variables.")
        logger.warning("For Streamlit Cloud: Add [supabase] section to secrets with 'url' and 'service_key'")
        logger.warning("For local development: Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        return None

    try:
        client = create_client(supabase_url, supabase_key)
        logger.info(f"Supabase client created successfully for URL: {supabase_url}")
        return client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None
