"""
Supabase Client

Handles the connection and interaction with the Supabase backend.
"""

import os
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

def get_supabase_client() -> Client:
    """
    Initializes and returns the Supabase client.

    Reads connection details from environment variables:
    - SUPABASE_URL: The project URL for your Supabase instance.
    - SUPABASE_SERVICE_KEY: The service role key for backend access.

    Returns:
        An initialized Supabase client instance, or None if credentials are not set.
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials (URL or Service Key) are not set. Database operations will be disabled.")
        return None

    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None
