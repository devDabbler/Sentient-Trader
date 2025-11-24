"""
LLM Usage Dashboard UI
Displays LLM usage statistics, cost tracking, and efficiency metrics
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from services.llm_usage_tracker import get_llm_usage_tracker
from models.llm_models import LLMPriority


def render_llm_usage_dashboard():
    """Render the LLM usage monitoring dashboard"""
    st.title("🤖 LLM Usage & Cost Monitor")
    
    tracker = get_llm_usage_tracker()
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = tracker.get_total_cost()
    efficiency = tracker.get_efficiency_metrics()
    
    with col1:
        st.metric(
            "Total Cost",
            f"${total_cost:.4f}",
            help="Total LLM API cost across all services"
        )
    
    with col2:
        st.metric(
            "Cache Hit Rate",
            f"{efficiency['overall_cache_hit_rate']:.1%}",
            help="Percentage of requests served from cache"
        )
    
    with col3:
        st.metric(
            "Avg Cost/Request",
            f"${efficiency['avg_cost_per_request']:.4f}",
            help="Average cost per LLM request (excluding cached)"
        )
    
    with col4:
        st.metric(
            "Cache Savings",
            f"${efficiency['total_cost_saved_by_cache']:.4f}",
            help="Estimated cost saved by caching"
        )
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "💰 Cost Breakdown",
        "⚙️ Configuration",
        "📈 History"
    ])
    
    with tab1:
        render_overview_tab(tracker)
    
    with tab2:
        render_cost_breakdown_tab(tracker)
    
    with tab3:
        render_configuration_tab(tracker)
    
    with tab4:
        render_history_tab(tracker)


def render_overview_tab(tracker):
    """Render overview statistics"""
    st.subheader("Current Usage Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority breakdown
        st.markdown("**Requests by Priority**")
        priority_data = tracker.get_priority_breakdown()
        
        if sum(priority_data.values()) > 0:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(priority_data.keys()),
                values=list(priority_data.values()),
                marker=dict(colors=['#ff4444', '#ff8844', '#ffcc44', '#44ff44']),
                hole=0.4
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No requests recorded yet")
    
    with col2:
        # Provider breakdown
        st.markdown("**Requests by Provider**")
        provider_data = tracker.get_provider_breakdown()
        provider_data = {k: v for k, v in provider_data.items() if v > 0}
        
        if provider_data:
            # Create bar chart
            fig = go.Figure(data=[go.Bar(
                x=list(provider_data.keys()),
                y=list(provider_data.values()),
                marker=dict(color='#4488ff')
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis_title="Provider",
                yaxis_title="Requests"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No requests recorded yet")
    
    st.divider()
    
    # Efficiency metrics
    st.markdown("**Efficiency Metrics**")
    efficiency = tracker.get_efficiency_metrics()
    
    metrics_df = pd.DataFrame([
        {
            "Metric": "Cache Hit Rate",
            "Value": f"{efficiency['overall_cache_hit_rate']:.1%}",
            "Description": "Requests served from cache"
        },
        {
            "Metric": "Avg Tokens/Request",
            "Value": f"{efficiency['avg_tokens_per_request']:.0f}",
            "Description": "Average tokens per request"
        },
        {
            "Metric": "Cost Saved by Cache",
            "Value": f"${efficiency['total_cost_saved_by_cache']:.4f}",
            "Description": "Estimated savings from caching"
        }
    ])
    
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)


def render_cost_breakdown_tab(tracker):
    """Render cost breakdown by service"""
    st.subheader("Cost Breakdown by Service")
    
    service_breakdown = tracker.get_service_breakdown()
    
    if not service_breakdown:
        st.info("No usage data available yet")
        return
    
    # Create DataFrame
    data = []
    for service, stats in service_breakdown.items():
        data.append({
            "Service": service,
            "Requests": stats["requests"],
            "Cached": stats["cached"],
            "Tokens": stats["tokens"],
            "Cost (USD)": stats["cost"],
            "Cache Hit %": stats["cache_hit_rate"] * 100,
            "Errors": stats["errors"]
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("Cost (USD)", ascending=False)
    
    # Display table
    st.dataframe(
        df.style.format({
            "Cost (USD)": "${:.4f}",
            "Cache Hit %": "{:.1f}%",
            "Tokens": "{:,.0f}",
            "Requests": "{:,.0f}",
            "Cached": "{:,.0f}"
        }).background_gradient(subset=["Cost (USD)"], cmap="Reds"),
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()
    
    # Cost by service chart
    st.markdown("**Cost Distribution**")
    
    if len(df) > 0:
        fig = px.bar(
            df,
            x="Service",
            y="Cost (USD)",
            color="Cost (USD)",
            color_continuous_scale="Reds",
            title="Cost by Service"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Export CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "llm_usage_breakdown.csv",
                "text/csv"
            )


def render_configuration_tab(tracker):
    """Render LLM configuration management"""
    st.subheader("LLM Manager Configuration")
    
    manager = tracker.manager
    config = manager.config
    
    # Current configuration
    st.markdown("**Current Settings**")
    
    config_data = [
        {"Setting": "Primary Provider", "Value": config.primary_provider.value},
        {"Setting": "Default Model", "Value": config.default_model},
        {"Setting": "Caching Enabled", "Value": "Yes" if config.enable_caching else "No"},
        {"Setting": "Default Cache TTL", "Value": f"{config.default_cache_ttl}s"},
        {"Setting": "Max Queue Size", "Value": str(config.max_queue_size)},
        {"Setting": "Cost Tracking", "Value": "Yes" if config.cost_tracking_enabled else "No"}
    ]
    
    st.dataframe(
        pd.DataFrame(config_data),
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()
    
    # Rate limits
    st.markdown("**Rate Limits by Provider**")
    
    rate_limit_data = []
    for provider_name, rate_config in config.rate_limits.items():
        rate_limit_data.append({
            "Provider": provider_name,
            "Max RPM": rate_config.max_requests_per_minute,
            "Max Concurrent": rate_config.max_concurrent_requests,
            "Backoff (s)": rate_config.backoff_seconds,
            "Max Retries": rate_config.max_retries
        })
    
    st.dataframe(
        pd.DataFrame(rate_limit_data),
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()
    
    # Cache management
    st.markdown("**Cache Management**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cache_size = len(manager.cache)
        st.metric("Cache Entries", cache_size)
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Stats"):
            if st.session_state.get("confirm_reset"):
                tracker.reset_all_stats()
                st.success("Statistics reset!")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")


def render_history_tab(tracker):
    """Render historical usage data"""
    st.subheader("Historical Usage & Cost")
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        days = st.slider("Days to Display", 1, 30, 7)
    
    with col2:
        if st.button("💾 Save Snapshot"):
            tracker.save_snapshot()
            st.success("Snapshot saved!")
    
    # Get historical data
    historical_cost = tracker.get_historical_cost(days=days)
    
    if not historical_cost:
        st.info("No historical data available yet. Snapshots are saved automatically.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(historical_cost, columns=["Date", "Cost (USD)"])
    
    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Cost (USD)"],
        mode='lines+markers',
        name='Daily Cost',
        line=dict(color='#4488ff', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"LLM Cost Trend (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Cumulative Cost (USD)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("**Daily Breakdown**")
    st.dataframe(
        df.style.format({"Cost (USD)": "${:.4f}"}),
        hide_index=True,
        use_container_width=True
    )
    
    # Summary stats
    if len(df) > 1:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", f"${df['Cost (USD)'].iloc[-1]:.4f}")
        
        with col2:
            daily_avg = (df['Cost (USD)'].iloc[-1] - df['Cost (USD)'].iloc[0]) / len(df)
            st.metric("Avg Daily Increase", f"${daily_avg:.4f}")
        
        with col3:
            projected_monthly = daily_avg * 30
            st.metric("Projected Monthly", f"${projected_monthly:.2f}")


if __name__ == "__main__":
    render_llm_usage_dashboard()
