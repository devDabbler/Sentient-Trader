"""
LLM Usage & Cost Monitoring Tab
Real-time monitoring of LLM request manager usage, costs, and performance
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px

from services.llm_request_manager import get_llm_manager
from models.llm_models import UsageStats


def render_llm_usage_tab():
    """Render the LLM Usage & Cost Monitoring tab"""
    
    st.title("🤖 LLM Usage & Cost Monitor")
    st.markdown("Real-time monitoring of AI/LLM request manager performance and costs")
    
    # Get manager instance
    manager = get_llm_manager()
    
    # Control buttons in header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("### 📊 Overview")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    with col4:
        if st.button("📊 Reset Stats", use_container_width=True):
            manager.reset_stats()
            st.success("Statistics reset!")
            st.rerun()
    
    st.divider()
    
    # Get usage statistics
    all_stats = manager.get_usage_stats()
    total_cost = manager.get_total_cost()
    
    if not all_stats:
        st.info("No LLM requests have been made yet. Start using the system to see statistics here.")
        return
    
    # Calculate aggregate metrics
    total_requests = sum(s.total_requests for s in all_stats.values())
    total_cached = sum(s.cached_requests for s in all_stats.values())
    total_tokens = sum(s.total_tokens for s in all_stats.values())
    total_errors = sum(s.errors for s in all_stats.values())
    
    cache_hit_rate = (total_cached / total_requests * 100) if total_requests > 0 else 0
    
    # Top metrics
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.metric(
            "Total Requests",
            f"{total_requests:,}",
            delta=f"{total_cached} cached ({cache_hit_rate:.1f}%)",
            delta_color="normal"
        )
    
    with metric_cols[1]:
        st.metric(
            "Total Cost",
            f"${total_cost:.4f}",
            delta="USD",
            delta_color="off"
        )
    
    with metric_cols[2]:
        avg_cost_per_request = (total_cost / total_requests) if total_requests > 0 else 0
        st.metric(
            "Avg Cost/Request",
            f"${avg_cost_per_request:.6f}",
            delta=f"{total_tokens:,} tokens",
            delta_color="off"
        )
    
    with metric_cols[3]:
        st.metric(
            "Cache Hit Rate",
            f"{cache_hit_rate:.1f}%",
            delta=f"{total_cached}/{total_requests}",
            delta_color="normal" if cache_hit_rate > 30 else "inverse"
        )
    
    with metric_cols[4]:
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        st.metric(
            "Error Rate",
            f"{error_rate:.2f}%",
            delta=f"{total_errors} errors",
            delta_color="inverse" if total_errors > 0 else "normal"
        )
    
    st.divider()
    
    # Two-column layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Cost breakdown by service
        st.markdown("### 💰 Cost by Service")
        
        service_costs = []
        for service_name, stats in all_stats.items():
            service_costs.append({
                'Service': service_name,
                'Cost': stats.total_cost_usd,
                'Requests': stats.total_requests,
                'Tokens': stats.total_tokens
            })
        
        if service_costs:
            df_costs = pd.DataFrame(service_costs)
            df_costs = df_costs.sort_values('Cost', ascending=False)
            
            # Pie chart
            fig_pie = px.pie(
                df_costs,
                values='Cost',
                names='Service',
                title='Cost Distribution',
                hover_data=['Requests', 'Tokens']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Data table
            st.dataframe(
                df_costs.style.format({
                    'Cost': '${:.4f}',
                    'Requests': '{:,}',
                    'Tokens': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    with right_col:
        # Request breakdown by priority
        st.markdown("### 🎯 Requests by Priority")
        
        priority_data = {}
        for stats in all_stats.values():
            for priority, count in stats.requests_by_priority.items():
                priority_data[priority] = priority_data.get(priority, 0) + count
        
        if priority_data:
            df_priority = pd.DataFrame([
                {'Priority': k, 'Count': v}
                for k, v in priority_data.items()
            ])
            
            # Sort by priority level
            priority_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            df_priority['Priority'] = pd.Categorical(
                df_priority['Priority'],
                categories=priority_order,
                ordered=True
            )
            df_priority = df_priority.sort_values('Priority')
            
            # Bar chart
            fig_priority = px.bar(
                df_priority,
                x='Priority',
                y='Count',
                title='Request Priority Distribution',
                color='Priority',
                color_discrete_map={
                    'CRITICAL': '#FF4444',
                    'HIGH': '#FF9944',
                    'MEDIUM': '#FFDD44',
                    'LOW': '#44DD44'
                }
            )
            st.plotly_chart(fig_priority, use_container_width=True)
            
            # Data table
            st.dataframe(
                df_priority.style.format({'Count': '{:,}'}),
                use_container_width=True,
                hide_index=True
            )
    
    st.divider()
    
    # Provider breakdown
    st.markdown("### 🌐 Provider Distribution")
    
    provider_cols = st.columns(3)
    
    provider_data = {}
    for stats in all_stats.values():
        for provider, count in stats.requests_by_provider.items():
            provider_data[provider] = provider_data.get(provider, 0) + count
    
    if provider_data:
        # Provider pie chart
        with provider_cols[0]:
            df_providers = pd.DataFrame([
                {'Provider': k, 'Requests': v}
                for k, v in provider_data.items()
            ])
            
            fig_providers = px.pie(
                df_providers,
                values='Requests',
                names='Provider',
                title='Provider Usage'
            )
            st.plotly_chart(fig_providers, use_container_width=True)
        
        # Provider table
        with provider_cols[1]:
            st.dataframe(
                df_providers.style.format({'Requests': '{:,}'}),
                use_container_width=True,
                hide_index=True
            )
        
        # Provider metrics
        with provider_cols[2]:
            for provider, count in provider_data.items():
                pct = (count / total_requests * 100) if total_requests > 0 else 0
                st.metric(
                    provider.title(),
                    f"{count:,}",
                    delta=f"{pct:.1f}%",
                    delta_color="off"
                )
    
    st.divider()
    
    # Detailed service breakdown
    st.markdown("### 📋 Detailed Service Statistics")
    
    detailed_data = []
    for service_name, stats in all_stats.items():
        cache_rate = (stats.cached_requests / stats.total_requests * 100) if stats.total_requests > 0 else 0
        avg_tokens = stats.total_tokens / (stats.total_requests - stats.cached_requests) if (stats.total_requests - stats.cached_requests) > 0 else 0
        
        detailed_data.append({
            'Service': service_name,
            'Total Requests': stats.total_requests,
            'Cached': stats.cached_requests,
            'Cache Hit %': f"{cache_rate:.1f}%",
            'Total Tokens': stats.total_tokens,
            'Avg Tokens': int(avg_tokens),
            'Total Cost': stats.total_cost_usd,
            'Avg Cost': stats.total_cost_usd / stats.total_requests if stats.total_requests > 0 else 0,
            'Errors': stats.errors,
            'Last Request': stats.last_request_time.strftime('%Y-%m-%d %H:%M:%S') if stats.last_request_time else 'N/A'
        })
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed = df_detailed.sort_values('Total Cost', ascending=False)
        
        st.dataframe(
            df_detailed.style.format({
                'Total Requests': '{:,}',
                'Cached': '{:,}',
                'Total Tokens': '{:,}',
                'Avg Tokens': '{:,}',
                'Total Cost': '${:.4f}',
                'Avg Cost': '${:.6f}',
                'Errors': '{:,}'
            }).background_gradient(
                subset=['Cache Hit %'],
                cmap='Greens',
                vmin=0,
                vmax=100
            ),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # Cache statistics
    st.markdown("### 💾 Cache Performance")
    
    cache_cols = st.columns(4)
    
    cache_size = len(manager.cache)
    
    with cache_cols[0]:
        st.metric("Cache Entries", f"{cache_size:,}")
    
    with cache_cols[1]:
        st.metric("Cache Hits", f"{total_cached:,}")
    
    with cache_cols[2]:
        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
    
    with cache_cols[3]:
        cost_saved = total_cached * avg_cost_per_request if total_cached > 0 else 0
        st.metric("Est. Cost Saved", f"${cost_saved:.4f}")
    
    # Cache settings info
    with st.expander("⚙️ Cache Configuration"):
        st.markdown(f"""
        **Current Settings:**
        - Cache Enabled: {manager.config.enable_caching}
        - Max Cache Size: 1,000 entries (auto-cleanup at limit)
        - Default TTL: Varies by service (60s - 900s)
        - Cache Strategy: MD5 hash of prompt + model
        
        **Cache Benefits:**
        - Reduced API costs: ~${cost_saved:.4f} saved
        - Faster response times for repeated queries
        - Lower rate limit usage
        - Improved system reliability
        """)
    
    st.divider()
    
    # Cost projections
    st.markdown("### 📈 Cost Projections")
    
    proj_cols = st.columns(3)
    
    # Calculate hourly rate (based on last request time)
    if total_requests > 0 and all_stats:
        # Get time range of requests
        earliest_time = None
        latest_time = None
        
        for stats in all_stats.values():
            if stats.last_request_time:
                if latest_time is None or stats.last_request_time > latest_time:
                    latest_time = stats.last_request_time
        
        if latest_time:
            # Estimate based on current usage
            with proj_cols[0]:
                hourly_cost = avg_cost_per_request * 60  # Assume 60 req/hour
                st.metric("Projected Hourly Cost", f"${hourly_cost:.4f}")
            
            with proj_cols[1]:
                daily_cost = hourly_cost * 24
                st.metric("Projected Daily Cost", f"${daily_cost:.2f}")
            
            with proj_cols[2]:
                monthly_cost = daily_cost * 30
                st.metric("Projected Monthly Cost", f"${monthly_cost:.2f}")
    
    # Configuration info
    with st.expander("⚙️ LLM Manager Configuration"):
        st.markdown(f"""
        **Provider Settings:**
        - Primary Provider: {manager.config.primary_provider.value}
        - Fallback Providers: {', '.join([p.value for p in manager.config.fallback_providers])}
        - Default Model: {manager.config.default_model}
        
        **Rate Limits:**
        - OpenRouter: {manager.config.rate_limits.get('openrouter', {}).max_requests_per_minute if manager.config.rate_limits.get('openrouter') else 'N/A'} req/min
        - Concurrent Requests: {manager.config.rate_limits.get('openrouter', {}).max_concurrent_requests if manager.config.rate_limits.get('openrouter') else 'N/A'}
        
        **Queue Settings:**
        - Max Queue Size: {manager.config.max_queue_size}
        - Priority Levels: CRITICAL > HIGH > MEDIUM > LOW
        
        **Environment Variables:**
        - `OPENROUTER_API_KEY`: {'✓ Set' if manager.api_keys.get(manager.config.primary_provider) else '✗ Missing'}
        - `LLM_ENABLE_CACHE`: {manager.config.enable_caching}
        - `LLM_DEFAULT_MODEL`: {manager.config.default_model}
        """)
    
    # Export options
    st.divider()
    st.markdown("### 📥 Export Data")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        if st.button("📊 Export Stats (JSON)", use_container_width=True):
            import json
            stats_json = {
                'timestamp': datetime.now().isoformat(),
                'total_cost': total_cost,
                'total_requests': total_requests,
                'cache_hit_rate': cache_hit_rate,
                'services': {
                    name: {
                        'requests': stats.total_requests,
                        'cost': stats.total_cost_usd,
                        'tokens': stats.total_tokens,
                        'cached': stats.cached_requests,
                        'errors': stats.errors
                    }
                    for name, stats in all_stats.items()
                }
            }
            
            st.download_button(
                "Download JSON",
                data=json.dumps(stats_json, indent=2),
                file_name=f"llm_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with export_cols[1]:
        if st.button("📄 Export Stats (CSV)", use_container_width=True):
            if detailed_data:
                df_export = pd.DataFrame(detailed_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"llm_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with export_cols[2]:
        st.markdown("**Auto-refresh:** Coming soon")
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | LLM Request Manager v1.0")


if __name__ == "__main__":
    render_llm_usage_tab()
