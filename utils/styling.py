"""Custom CSS styling for the Streamlit app."""

import streamlit as st


def apply_custom_styling():
    """Apply custom CSS for enhanced visual appeal with Streamlit."""
    
    # Modern clean theme for trading platform
    st.markdown("""
    <style>
    /* Modern clean theme for trading platform */
    .stMetric {
        background-color: #FFFFFF;
        border: 2px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stMetric > div > div > div {
        color: #1F2937;
    }
    
    /* Custom status indicators */
    .stStatus {
        border-radius: 12px;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    
    /* Enhanced data editor styling */
    .stDataEditor {
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
    
    /* Custom badge colors */
    .stBadge {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }
    
    /* Trading-themed colors for metrics with better contrast */
    .profit-metric {
        color: #059669 !important;
        font-weight: 600;
    }
    
    .loss-metric {
        color: #DC2626 !important;
        font-weight: 600;
    }
    
    .neutral-metric {
        color: #D97706 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced metric text styling for light theme
    st.markdown("""
    <style>
    /* Enhanced metric text styling for light theme */
    .stMetric {
        color: #1F2937 !important;
    }

    .stMetric label, .stMetric .metric-label, .stMetric .stMetricLabel {
        color: #6B7280 !important;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* Ensure all metric text is visible with proper contrast */
    .stMetric * {
        color: #1F2937 !important;
    }

    .stMetric .stMetricValue {
        color: #111827 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    .stMetric .stMetricDelta {
        color: #059669 !important;
        font-weight: 600;
    }

    /* Clean alert styling */
    .stAlert {
        background-color: #FEFEFE !important;
        color: #1F2937 !important;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stAlert * {
        color: #1F2937 !important;
    }

    /* Success alerts */
    .stAlert[data-baseweb="notification"] {
        background-color: #F0FDF4 !important;
        border-color: #BBF7D0;
    }

    /* Warning alerts */
    .stAlert[data-baseweb="notification"][aria-label="warning"] {
        background-color: #FFFBEB !important;
        border-color: #FED7AA;
    }

    /* Error alerts */
    .stAlert[data-baseweb="notification"][aria-label="error"] {
        background-color: #FEF2F2 !important;
        border-color: #FECACA;
    }

    /* Info alerts */
    .stAlert[data-baseweb="notification"][aria-label="info"] {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE;
    }

    /* Clean expander styling */
    .stExpander .stExpanderHeader {
        color: #1F2937 !important;
        background-color: #F9FAFB !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }

    .stExpander .stExpanderHeader:hover {
        background-color: #F3F4F6 !important;
    }

    .stExpander .stExpanderHeader * {
        color: #1F2937 !important;
    }

    /* Clean container styling */
    .stContainer {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border-radius: 12px;
    }

    .stContainer * {
        color: #1F2937 !important;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #F8FAFC;
    }

    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EBF8FF;
        color: #1E40AF;
    }
    </style>
    """, unsafe_allow_html=True)
