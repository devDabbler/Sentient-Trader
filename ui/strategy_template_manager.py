"""
UI component for managing option strategy templates.
Allows users to add, edit, view, and delete custom strategy templates.
"""

import streamlit as st
from models.option_strategy_templates import (
    OptionStrategyTemplate,
    template_manager
)
from datetime import datetime


def render_template_manager():
    """Render the strategy template manager UI"""
    
    st.header("📚 Option Strategy Template Manager")
    st.write("Create and manage your custom option strategy templates")
    
    # Tabs for different actions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 View Templates",
        "➕ Add New Template",
        "✏️ Edit Template",
        "🔍 Search Templates"
    ])
    
    # TAB 1: View Templates
    with tab1:
        render_view_templates()
    
    # TAB 2: Add New Template
    with tab2:
        render_add_template()
    
    # TAB 3: Edit Template
    with tab3:
        render_edit_template()
    
    # TAB 4: Search Templates
    with tab4:
        render_search_templates()


def render_view_templates():
    """Display all existing templates"""
    st.subheader("Your Strategy Templates")
    
    templates = template_manager.get_all_templates()
    
    if not templates:
        st.info("No templates found. Add your first template in the 'Add New Template' tab!")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_experience = st.selectbox(
            "Filter by Experience",
            ["All", "Beginner", "Intermediate", "Advanced", "Professional"],
            key="view_filter_exp"
        )
    with col2:
        filter_direction = st.selectbox(
            "Filter by Direction",
            ["All", "Bullish", "Bearish", "Neutral", "Volatility"],
            key="view_filter_dir"
        )
    with col3:
        filter_oa = st.checkbox("Option Alpha Compatible Only", value=False, key="view_filter_oa")
    
    # Apply filters
    filtered_templates = templates
    if filter_experience != "All":
        filtered_templates = [t for t in filtered_templates if t.experience_level == filter_experience]
    if filter_direction != "All":
        filtered_templates = [t for t in filtered_templates if filter_direction.upper() in t.direction.upper()]
    if filter_oa:
        filtered_templates = [t for t in filtered_templates if t.option_alpha_compatible]
    
    st.write(f"**Showing {len(filtered_templates)} of {len(templates)} templates**")
    
    # Display templates
    for template in filtered_templates:
        with st.expander(f"🎯 {template.name} ({template.experience_level} | {template.risk_level} Risk)"):
            render_template_details(template)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Use in Strategy Advisor", key=f"use_{template.strategy_id}"):
                    st.session_state.selected_template = template.strategy_id
                    st.success(f"✅ Template selected! Go to Strategy Advisor tab.")
            with col2:
                if st.button(f"Delete Template", key=f"delete_{template.strategy_id}", type="secondary"):
                    if template_manager.delete_template(template.strategy_id):
                        st.success(f"Deleted: {template.name}")
                        st.rerun()
                    else:
                        st.error("Failed to delete template")


def render_template_details(template: OptionStrategyTemplate):
    """Render detailed view of a template"""
    
    # Basic Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Experience Level", template.experience_level)
        st.metric("Risk Level", template.risk_level)
    with col2:
        st.metric("Capital Required", template.capital_requirement)
        st.metric("Strategy Type", template.strategy_type)
    with col3:
        st.metric("Direction", template.direction)
        if template.typical_win_rate:
            st.metric("Win Rate", template.typical_win_rate)
    
    st.markdown(f"**Description:** {template.description}")
    
    # Risk/Reward
    st.markdown("### 💰 Risk & Reward")
    st.write(f"**Max Loss:** {template.max_loss}")
    st.write(f"**Max Gain:** {template.max_gain}")
    if template.profit_target:
        st.write(f"**Profit Target:** {template.profit_target}")
    if template.stop_loss:
        st.write(f"**Stop Loss:** {template.stop_loss}")
    
    # Best Conditions
    st.markdown("### ✅ Best For")
    for condition in template.best_for:
        st.write(f"• {condition}")
    
    st.write(f"**Ideal IV Rank:** {template.ideal_iv_rank}")
    st.write(f"**Ideal Outlook:** {', '.join(template.ideal_market_outlook)}")
    st.write(f"**Typical DTE:** {template.typical_dte}")
    
    # Setup Steps
    if template.setup_steps:
        st.markdown("### 📖 Setup Steps")
        for i, step in enumerate(template.setup_steps, 1):
            st.write(f"{i}. {step}")
    
    # Management Rules
    if template.management_rules:
        st.markdown("### 🛠️ Management Rules")
        for rule in template.management_rules:
            st.write(f"• {rule}")
    
    # Examples
    if template.examples:
        st.markdown("### 💡 Examples")
        for example in template.examples:
            st.info(example)
    
    # Warnings
    if template.warnings:
        st.markdown("### ⚠️ Warnings")
        for warning in template.warnings:
            st.warning(warning)
    
    # Notes
    if template.notes:
        st.markdown("### 📝 Notes")
        st.write(template.notes)
    
    # Option Alpha
    if template.option_alpha_compatible:
        st.markdown("### 🔗 Option Alpha")
        st.success(f"✅ Compatible - Action: `{template.option_alpha_action}`")
    
    # Metadata
    with st.expander("ℹ️ Metadata"):
        st.write(f"**Strategy ID:** {template.strategy_id}")
        st.write(f"**Created:** {template.created_date}")
        if template.source:
            st.write(f"**Source:** {template.source}")
        if template.tags:
            st.write(f"**Tags:** {', '.join(template.tags)}")


def render_add_template():
    """Form to add a new template"""
    st.subheader("Add New Strategy Template")
    
    with st.form("add_template_form"):
        # Basic Info
        st.markdown("### Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Strategy Name*", placeholder="e.g., Iron Butterfly")
            strategy_id = st.text_input(
                "Strategy ID*",
                placeholder="e.g., iron_butterfly",
                help="Unique identifier (lowercase, underscores only)"
            )
        with col2:
            strategy_type = st.selectbox(
                "Strategy Type*",
                ["SINGLE_LEG", "SPREAD", "MULTI_LEG", "COMPLEX"]
            )
            direction = st.selectbox(
                "Direction*",
                ["BULLISH", "BEARISH", "NEUTRAL", "VOLATILITY"]
            )
        
        description = st.text_area(
            "Description*",
            placeholder="Brief description of the strategy...",
            height=100
        )
        
        # Risk Profile
        st.markdown("### Risk Profile")
        col1, col2 = st.columns(2)
        with col1:
            risk_level = st.selectbox("Risk Level*", ["Low", "Medium", "High", "Very High"])
            experience_level = st.selectbox(
                "Experience Level*",
                ["Beginner", "Intermediate", "Advanced", "Professional"]
            )
        with col2:
            capital_requirement = st.selectbox(
                "Capital Requirement*",
                ["Low", "Medium", "High", "Very High"]
            )
            typical_win_rate = st.text_input("Typical Win Rate", placeholder="e.g., 60-70%")
        
        max_loss = st.text_input("Max Loss*", placeholder="e.g., Premium paid")
        max_gain = st.text_input("Max Gain*", placeholder="e.g., Unlimited")
        
        # Market Conditions
        st.markdown("### Market Conditions")
        ideal_iv_rank = st.selectbox(
            "Ideal IV Rank*",
            ["Low (<30)", "Medium (30-60)", "High (>60)", "Any"]
        )
        
        ideal_outlook = st.multiselect(
            "Ideal Market Outlook*",
            ["Bullish", "Bearish", "Neutral"],
            default=["Neutral"]
        )
        
        best_for = st.text_area(
            "Best For (one per line)*",
            placeholder="High IV environment\nNeutral outlook\nIncome generation",
            height=100
        )
        
        # Trade Details
        st.markdown("### Trade Details")
        col1, col2 = st.columns(2)
        with col1:
            typical_dte = st.text_input("Typical DTE*", placeholder="e.g., 30-45 days")
            profit_target = st.text_input("Profit Target", placeholder="e.g., 50% of credit")
        with col2:
            stop_loss = st.text_input("Stop Loss", placeholder="e.g., 2x credit received")
        
        # Option Alpha
        st.markdown("### Option Alpha Integration")
        option_alpha_compatible = st.checkbox("Option Alpha Compatible", value=True)
        option_alpha_action = st.text_input(
            "Option Alpha Action",
            placeholder="e.g., IRON_CONDOR, SELL_PUT",
            help="The action code used in Option Alpha"
        )
        
        # Additional Details
        st.markdown("### Additional Details")
        setup_steps = st.text_area(
            "Setup Steps (one per line)",
            placeholder="Step 1: Analyze market conditions\nStep 2: Select strikes\nStep 3: Enter position",
            height=100
        )
        
        management_rules = st.text_area(
            "Management Rules (one per line)",
            placeholder="Close at 50% profit\nRoll if tested\nMonitor daily",
            height=100
        )
        
        examples = st.text_area(
            "Examples (one per line)",
            placeholder="Stock at $100: Sell $95 put, Sell $105 call",
            height=100
        )
        
        warnings = st.text_area(
            "Warnings (one per line)",
            placeholder="⚠️ Requires margin\n⚠️ High risk if stock moves significantly",
            height=100
        )
        
        notes = st.text_area("Additional Notes", height=100)
        source = st.text_input("Source", placeholder="e.g., tastytrade, Option Alpha")
        tags = st.text_input("Tags (comma-separated)", placeholder="income, neutral, high-iv")
        
        # Submit
        submitted = st.form_submit_button("💾 Save Template", type="primary")
        
        if submitted:
            # Validation
            if not all([name, strategy_id, description, max_loss, max_gain, typical_dte]):
                st.error("Please fill in all required fields marked with *")
                return
            
            # Check if ID already exists
            if template_manager.get_template(strategy_id):
                st.error(f"Strategy ID '{strategy_id}' already exists. Please use a unique ID.")
                return
            
            # Create template
            template = OptionStrategyTemplate(
                strategy_id=strategy_id,
                name=name,
                description=description,
                strategy_type=strategy_type,
                direction=direction,
                risk_level=risk_level,
                max_loss=max_loss,
                max_gain=max_gain,
                experience_level=experience_level,
                capital_requirement=capital_requirement,
                best_for=[line.strip() for line in best_for.split('\n') if line.strip()],
                ideal_iv_rank=ideal_iv_rank,
                ideal_market_outlook=ideal_outlook,
                typical_dte=typical_dte,
                typical_win_rate=typical_win_rate if typical_win_rate else None,
                profit_target=profit_target if profit_target else None,
                stop_loss=stop_loss if stop_loss else None,
                option_alpha_compatible=option_alpha_compatible,
                option_alpha_action=option_alpha_action,
                setup_steps=[line.strip() for line in setup_steps.split('\n') if line.strip()],
                management_rules=[line.strip() for line in management_rules.split('\n') if line.strip()],
                examples=[line.strip() for line in examples.split('\n') if line.strip()],
                warnings=[line.strip() for line in warnings.split('\n') if line.strip()],
                notes=notes if notes else None,
                source=source if source else None,
                tags=[tag.strip() for tag in tags.split(',') if tag.strip()]
            )
            
            # Save template
            if template_manager.add_template(template):
                st.success(f"✅ Template '{name}' saved successfully!")
                st.balloons()
            else:
                st.error("Failed to save template. Please try again.")


def render_edit_template():
    """Form to edit an existing template"""
    st.subheader("Edit Existing Template")
    
    templates = template_manager.get_all_templates()
    
    if not templates:
        st.info("No templates available to edit. Add a template first!")
        return
    
    # Select template to edit
    template_names = {t.name: t.strategy_id for t in templates}
    selected_name = st.selectbox("Select Template to Edit", list(template_names.keys()))
    
    if not selected_name:
        return
    
    template = template_manager.get_template(template_names[selected_name])
    
    if not template:
        st.error("Template not found")
        return
    
    st.info(f"Editing: **{template.name}** (ID: {template.strategy_id})")
    
    # Pre-filled form (similar to add form but with existing values)
    with st.form("edit_template_form"):
        # Basic Info
        st.markdown("### Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Strategy Name*", value=template.name)
            strategy_id_display = st.text_input(
                "Strategy ID (cannot be changed)",
                value=template.strategy_id,
                disabled=True
            )
        with col2:
            strategy_type = st.selectbox(
                "Strategy Type*",
                ["SINGLE_LEG", "SPREAD", "MULTI_LEG", "COMPLEX"],
                index=["SINGLE_LEG", "SPREAD", "MULTI_LEG", "COMPLEX"].index(template.strategy_type)
            )
            direction = st.selectbox(
                "Direction*",
                ["BULLISH", "BEARISH", "NEUTRAL", "VOLATILITY"],
                index=["BULLISH", "BEARISH", "NEUTRAL", "VOLATILITY"].index(template.direction)
            )
        
        description = st.text_area("Description*", value=template.description, height=100)
        
        # Risk Profile
        st.markdown("### Risk Profile")
        col1, col2 = st.columns(2)
        with col1:
            risk_level = st.selectbox(
                "Risk Level*",
                ["Low", "Medium", "High", "Very High"],
                index=["Low", "Medium", "High", "Very High"].index(template.risk_level)
            )
            experience_level = st.selectbox(
                "Experience Level*",
                ["Beginner", "Intermediate", "Advanced", "Professional"],
                index=["Beginner", "Intermediate", "Advanced", "Professional"].index(template.experience_level)
            )
        with col2:
            capital_requirement = st.selectbox(
                "Capital Requirement*",
                ["Low", "Medium", "High", "Very High"],
                index=["Low", "Medium", "High", "Very High"].index(template.capital_requirement)
            )
            typical_win_rate = st.text_input(
                "Typical Win Rate",
                value=template.typical_win_rate or ""
            )
        
        max_loss = st.text_input("Max Loss*", value=template.max_loss)
        max_gain = st.text_input("Max Gain*", value=template.max_gain)
        
        # Market Conditions
        st.markdown("### Market Conditions")
        ideal_iv_rank = st.selectbox(
            "Ideal IV Rank*",
            ["Low (<30)", "Medium (30-60)", "High (>60)", "Any"],
            index=["Low (<30)", "Medium (30-60)", "High (>60)", "Any"].index(template.ideal_iv_rank)
        )
        
        ideal_outlook = st.multiselect(
            "Ideal Market Outlook*",
            ["Bullish", "Bearish", "Neutral"],
            default=template.ideal_market_outlook
        )
        
        best_for = st.text_area(
            "Best For (one per line)*",
            value='\n'.join(template.best_for),
            height=100
        )
        
        # Trade Details
        st.markdown("### Trade Details")
        col1, col2 = st.columns(2)
        with col1:
            typical_dte = st.text_input("Typical DTE*", value=template.typical_dte)
            profit_target = st.text_input(
                "Profit Target",
                value=template.profit_target or ""
            )
        with col2:
            stop_loss = st.text_input("Stop Loss", value=template.stop_loss or "")
        
        # Option Alpha
        st.markdown("### Option Alpha Integration")
        option_alpha_compatible = st.checkbox(
            "Option Alpha Compatible",
            value=template.option_alpha_compatible
        )
        option_alpha_action = st.text_input(
            "Option Alpha Action",
            value=template.option_alpha_action
        )
        
        # Additional Details
        st.markdown("### Additional Details")
        setup_steps = st.text_area(
            "Setup Steps (one per line)",
            value='\n'.join(template.setup_steps),
            height=100
        )
        
        management_rules = st.text_area(
            "Management Rules (one per line)",
            value='\n'.join(template.management_rules),
            height=100
        )
        
        examples = st.text_area(
            "Examples (one per line)",
            value='\n'.join(template.examples),
            height=100
        )
        
        warnings = st.text_area(
            "Warnings (one per line)",
            value='\n'.join(template.warnings),
            height=100
        )
        
        notes = st.text_area("Additional Notes", value=template.notes or "", height=100)
        source = st.text_input("Source", value=template.source or "")
        tags = st.text_input("Tags (comma-separated)", value=', '.join(template.tags))
        
        # Submit
        submitted = st.form_submit_button("💾 Update Template", type="primary")
        
        if submitted:
            # Create updated template
            updated_template = OptionStrategyTemplate(
                strategy_id=template.strategy_id,  # Keep original ID
                name=name,
                description=description,
                strategy_type=strategy_type,
                direction=direction,
                risk_level=risk_level,
                max_loss=max_loss,
                max_gain=max_gain,
                experience_level=experience_level,
                capital_requirement=capital_requirement,
                best_for=[line.strip() for line in best_for.split('\n') if line.strip()],
                ideal_iv_rank=ideal_iv_rank,
                ideal_market_outlook=ideal_outlook,
                typical_dte=typical_dte,
                typical_win_rate=typical_win_rate if typical_win_rate else None,
                profit_target=profit_target if profit_target else None,
                stop_loss=stop_loss if stop_loss else None,
                option_alpha_compatible=option_alpha_compatible,
                option_alpha_action=option_alpha_action,
                setup_steps=[line.strip() for line in setup_steps.split('\n') if line.strip()],
                management_rules=[line.strip() for line in management_rules.split('\n') if line.strip()],
                examples=[line.strip() for line in examples.split('\n') if line.strip()],
                warnings=[line.strip() for line in warnings.split('\n') if line.strip()],
                notes=notes if notes else None,
                created_date=template.created_date,  # Keep original date
                source=source if source else None,
                tags=[tag.strip() for tag in tags.split(',') if tag.strip()]
            )
            
            # Update template
            if template_manager.update_template(template.strategy_id, updated_template):
                st.success(f"✅ Template '{name}' updated successfully!")
            else:
                st.error("Failed to update template. Please try again.")


def render_search_templates():
    """Search and filter templates"""
    st.subheader("Search Strategy Templates")
    
    search_query = st.text_input(
        "🔍 Search by name, description, or tags",
        placeholder="e.g., iron condor, high IV, income"
    )
    
    if search_query:
        results = template_manager.search_templates(search_query)
        
        if results:
            st.success(f"Found {len(results))} matching templates")
            
            for template in results:
                with st.expander(f"🎯 {template.name}"):
                    render_template_details(template)
        else:
            st.warning("No templates found matching your search")
    else:
        st.info("Enter a search term to find templates")
