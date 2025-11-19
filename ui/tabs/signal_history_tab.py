"""
Signal History Tab
View historical signals and performance

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("Signal History")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("📜 Signal History")
    
    if st.session_state.signal_history:
        df = pd.DataFrame(st.session_state.signal_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Signals", len(df))
        with col2:
            st.metric("Total Risk", f"${df['estimated_risk'].sum():.2f}")
        with col3:
            st.metric("Avg AI Score", f"{df['llm_score'].mean():.2f}")
        
        # Enhanced interactive data editor with new Streamlit features
        st.subheader("📊 Interactive Signal Management")
        
        # Prepare data for editing (only editable columns)
        editable_df = df[['ticker', 'action', 'strike', 'qty', 'estimated_risk', 'llm_score', 'note']].copy()
        editable_df['timestamp'] = df['timestamp']
        editable_df['status'] = df['status']
        
        # Use st.data_editor for interactive editing
        edited_df = st.data_editor(
            editable_df.sort_values('timestamp', ascending=False),
            width='stretch',
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "action": st.column_config.SelectboxColumn(
                    "Strategy",
                    options=st.session_state.config.allowed_strategies,
                    width="medium"
                ),
                "strike": st.column_config.NumberColumn("Strike", format="$%.2f", width="small"),
                "qty": st.column_config.NumberColumn("Quantity", min_value=1, max_value=10, width="small"),
                "estimated_risk": st.column_config.NumberColumn("Risk", format="$%.2f", width="small"),
                "llm_score": st.column_config.NumberColumn("AI Score", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                "note": st.column_config.TextColumn("Notes", width="large"),
                "timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small")
            },
            disabled=["timestamp", "status"]  # Don't allow editing of these
        )
        
        # Check if data was modified
        if not edited_df.equals(editable_df.sort_values('timestamp', ascending=False)):
            st.success("✅ Signal data updated! Changes will be saved to session state.")
            # Update session state with edited data
            # Note: In a real app, you'd want to save this to a database
            st.session_state.signal_history = edited_df.to_dict('records')
        
        # Enhanced performance analytics with new Streamlit features
        st.subheader("📊 Performance Analytics")
        
        # Strategy performance analysis
        strategy_performance = df.groupby('action').agg({
            'estimated_risk': ['count', 'sum', 'mean'],
            'llm_score': 'mean'
        }).round(2)
        strategy_performance.columns = ['Signal Count', 'Total Risk', 'Avg Risk', 'Avg AI Score']
        strategy_performance = strategy_performance.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strategy Performance by Signal Count**")
            # Enhanced bar chart with sorting
            st.bar_chart(
                strategy_performance.set_index('action')['Signal Count'],
                sort="desc"  # New sorting feature
            )
        
        with col2:
            st.write("**Strategy Performance by Total Risk**")
            st.bar_chart(
                strategy_performance.set_index('action')['Total Risk'],
                sort="desc"  # New sorting feature
            )
        
        # Ticker performance analysis
        ticker_performance = df.groupby('ticker').agg({
            'estimated_risk': ['count', 'sum'],
            'llm_score': 'mean'
        }).round(2)
        ticker_performance.columns = ['Signal Count', 'Total Risk', 'Avg AI Score']
        ticker_performance = ticker_performance.reset_index()
        
        st.write("**Ticker Performance Analysis**")
        st.bar_chart(
            ticker_performance.set_index('ticker')['Signal Count'],
            sort="desc"
        )
        
        # Enhanced export with filtering options
        st.subheader("📥 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"], key='export_format_select')
        with col2:
            filter_status = st.multiselect("Filter by Status", ["Paper", "Live"], default=["Paper", "Live"])
        with col3:
            if st.button("📥 Export Data", type="primary"):
                filtered_df = df[df['status'].isin(filter_status)]
                
                if export_format == "CSV":
                    # Add UTF-8 BOM for Excel compatibility
                    csv = '\ufeff' + filtered_df.to_csv(index=False)
                    st.download_button("Download CSV", csv.encode('utf-8-sig'), "signals.csv", "text/csv")
                elif export_format == "JSON":
                    json_data = filtered_df.to_json(orient='records', indent=2)
                    st.download_button("Download JSON", json_data, "signals.json", "application/json")
                elif export_format == "Excel":
                    # For Excel export, you'd need openpyxl: pip install openpyxl
                    st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
    else:
        st.info("No signals generated yet")

