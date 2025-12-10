import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="Analysis Dashboard", layout="wide")

# Basic matplotlib styling (no seaborn needed)
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-whitegrid')  # Use matplotlib's built-in style

st.title("Threadworks 𓄀 Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    st.success("File uploaded successfully! Running analysis...")

    # Load the file
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Data Processing
    st.subheader("Data Processing")
    
    # Convert seconds to days
    time_cols = ['SecInBacklog', 'SecBlocked', 'SecInProgress', 'SecReadyReview',
                 'SecInReview', 'SecReadyForTest', 'SecInTest', 'SecInCI']
    
    for col in time_cols:
        if col in df.columns:
            new_col = col.replace('Sec', 'days')
            df[new_col] = df[col] / 86400
    
    # Calculate derived metrics
    day_cols = [c for c in df.columns if c.startswith('days')]
    if day_cols:
        df['totalCycleTime'] = df[day_cols].sum(axis=1)
    
    # Check for required columns before calculations
    required_cols = ['daysInProgress', 'daysInReview', 'daysInTest', 'daysInCI']
    if all(col in df.columns for col in required_cols):
        df['activeDevelopmentTime'] = (df['daysInProgress'] + df['daysInReview'] +
                                        df['daysInTest'] + df['daysInCI'])
    
    wait_cols = ['daysReadyReview', 'daysReadyForTest', 'daysBlocked']
    if all(col in df.columns for col in wait_cols):
        df['totalWaitTime'] = df[wait_cols].sum(axis=1)
    
    if 'totalCycleTime' in df.columns and 'daysInBacklog' in df.columns:
        df['cycleTimeExcludingBacklog'] = df['totalCycleTime'] - df['daysInBacklog']
    
    # Clean column names
    if 'Team ' in df.columns:
        df = df.rename(columns={'Team ': 'Team'})
    
    if 'Priority' in df.columns:
        df['Priority_clean'] = df['Priority'].str.strip()
    
    # Convert dates
    if 'Created' in df.columns:
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['month'] = df['Created'].dt.to_period('M')
    
    if 'Resolved' in df.columns:
        df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    
    st.success("Data processed successfully!")
    
    # Basic Statistics
    st.header("Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", f"{len(df):,}")
    with col2:
        if 'totalCycleTime' in df.columns:
            st.metric("Avg Cycle Time", f"{df['totalCycleTime'].mean():.1f} days")
    with col3:
        if 'totalCycleTime' in df.columns:
            st.metric("Median Cycle Time", f"{df['totalCycleTime'].median():.1f} days")
    with col4:
        if 'Status' in df.columns:
            st.metric("Resolved Tickets", f"{len(df[df['Status'] == 'Resolved']):,}")
    
    if 'Created' in df.columns:
        st.write(f"**Date Range:** {df['Created'].min().date()} to {df['Created'].max().date()}")
    
    # Stage Duration Analysis
    st.header("Stage Duration Analysis")
    
    stage_cols = {
        'Backlog': 'daysInBacklog',
        'In Progress': 'daysInProgress',
        'Ready for Review': 'daysReadyReview',
        'In Review': 'daysInReview',
        'Ready for Test': 'daysReadyForTest',
        'In Test': 'daysInTest',
        'In CI': 'daysInCI',
        'Blocked': 'daysBlocked'
    }
    
    stages = {}
    for label, col in stage_cols.items():
        if col in df.columns:
            stages[label] = df[col].mean()
    
    if stages:
        stages_sorted = dict(sorted(stages.items(), key=lambda x: x[1], reverse=True))
        total_time = sum(stages_sorted.values())
        
        # Display stage breakdown
        stage_df = pd.DataFrame([
            {'Stage': stage, 'Days': days, 'Percentage': f"{(days/total_time*100):.1f}%"}
            for stage, days in stages_sorted.items()
        ])
        st.dataframe(stage_df, use_container_width=True)
        
        # Waterfall Chart
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(stages_sorted),
            x=list(stages_sorted.keys()),
            y=list(stages_sorted.values()),
            text=[f"{v:.1f}d" for v in stages_sorted.values()],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Cycle Time Breakdown",
            showlegend=False,
            height=500,
            yaxis_title="Days"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie Chart
        fig = px.pie(
            values=list(stages_sorted.values()),
            names=list(stages_sorted.keys()),
            title='Time Distribution by Stage'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Active Development by Type
    if 'Type' in df.columns:
        st.header("Active Development by Ticket Type")
        
        ticket_types = df['Type'].unique()
        colors_by_type = {'Bug': '#D32F2F', 'Story': '#1976D2', 'Task': '#2E7D32'}
        
        fig = make_subplots(
            rows=1, cols=len(ticket_types),
            subplot_titles=[f'{t} (n={len(df[df["Type"]==t]):,})' for t in ticket_types],
            specs=[[{"type": "waterfall"}] * len(ticket_types)]
        )
        
        stage_mapping = {
            'In Progress': 'daysInProgress',
            'Ready for Review': 'daysReadyReview',
            'In Review': 'daysInReview',
            'Ready for Test': 'daysReadyForTest',
            'In Test': 'daysInTest',
            'In CI': 'daysInCI',
            'Blocked': 'daysBlocked'
        }
        
        for idx, ticket_type in enumerate(ticket_types):
            type_df = df[df['Type'] == ticket_type]
            
            stages_type = {}
            for label, col in stage_mapping.items():
                if col in df.columns:
                    stages_type[label] = type_df[col].mean()
            
            if stages_type:
                fig.add_trace(
                    go.Waterfall(
                        orientation="v",
                        measure=["relative"] * len(stages_type),
                        x=list(stages_type.keys()),
                        y=list(stages_type.values()),
                        text=[f"{v:.1f}d" for v in stages_type.values()],
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        increasing={"marker": {"color": colors_by_type.get(ticket_type, '#333')}},
                        showlegend=False
                    ),
                    row=1, col=idx+1
                )
        
        fig.update_layout(
            title="Active Development Cycle Time by Ticket Type<br><sub>(Backlog time excluded)</sub>",
            height=500
        )
        fig.update_yaxes(title_text="Days")
        st.plotly_chart(fig, use_container_width=True)
    
    # Team Performance
    if 'Team' in df.columns:
        st.header("Team Performance Analysis")
        
        team_stats = pd.DataFrame()
        
        for team in df['Team'].unique():
            if team == 'Unassigned' or pd.isna(team):
                continue
            
            team_df = df[df['Team'] == team]
            
            team_stats.loc[team, 'Tickets'] = len(team_df)
            
            if 'totalWaitTime' in df.columns:
                team_stats.loc[team, 'waitTime'] = team_df['totalWaitTime'].mean()
            if 'activeDevelopmentTime' in df.columns:
                team_stats.loc[team, 'activeTime'] = team_df['activeDevelopmentTime'].mean()
            if 'totalCycleTime' in df.columns:
                team_stats.loc[team, 'cycleTime'] = team_df['totalCycleTime'].mean()
        
        team_stats = team_stats[team_stats['Tickets'] >= 10].copy()
        
        if len(team_stats) > 0 and 'waitTime' in team_stats.columns and 'activeTime' in team_stats.columns:
            fig = px.scatter(
                team_stats,
                x='waitTime',
                y='activeTime',
                size='Tickets',
                color='cycleTime',
                hover_name=team_stats.index,
                hover_data={
                    'waitTime': ':.1f',
                    'activeTime': ':.1f',
                    'cycleTime': ':.1f',
                    'Tickets': ':,',
                },
                labels={
                    'waitTime': 'Wait Time (days)',
                    'activeTime': 'Active Development Time (days)',
                    'cycleTime': 'Avg Cycle Time (days)'
                },
                title='Team Performance: Wait Time vs Active Development',
                color_continuous_scale='YlOrRd',
                size_max=60
            )
            
            median_wait = team_stats['waitTime'].median()
            median_active = team_stats['activeTime'].median()
            
            fig.add_hline(y=median_active, line_dash="dash", line_color="gray",
                          annotation_text=f"Median Active: {median_active:.1f}d")
            fig.add_vline(x=median_wait, line_dash="dash", line_color="gray",
                          annotation_text=f"Median Wait: {median_wait:.1f}d")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                team_stats.nlargest(10, 'waitTime')[['waitTime', 'activeTime', 'Tickets']].round(1),
                use_container_width=True
            )
    
    # Priority Analysis
    if 'Priority_clean' in df.columns:
        st.header("Priority Analysis")
        
        agg_dict = {'Key': 'count'}
        if 'activeDevelopmentTime' in df.columns:
            agg_dict['activeDevelopmentTime'] = 'mean'
        if 'totalWaitTime' in df.columns:
            agg_dict['totalWaitTime'] = 'mean'
        if 'cycleTimeExcludingBacklog' in df.columns:
            agg_dict['cycleTimeExcludingBacklog'] = 'mean'
        
        priority_stats = df.groupby('Priority_clean').agg(agg_dict).round(1)
        
        col_names = ['Count']
        if 'activeDevelopmentTime' in agg_dict:
            col_names.append('Active Dev')
        if 'totalWaitTime' in agg_dict:
            col_names.append('Wait Time')
        if 'cycleTimeExcludingBacklog' in agg_dict:
            col_names.append('Cycle (ex. backlog)')
        
        priority_stats.columns = col_names
        st.dataframe(priority_stats, use_container_width=True)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Active Development Time', 'Wait Time', 'Ticket Volume')
        )
        
        if 'Active Dev' in priority_stats.columns:
            fig.add_trace(
                go.Bar(x=priority_stats.index, y=priority_stats['Active Dev'],
                       marker_color='#1976D2', text=priority_stats['Active Dev'],
                       texttemplate='%{text:.1f}d', name='Active Dev'),
                row=1, col=1
            )
        
        if 'Wait Time' in priority_stats.columns:
            fig.add_trace(
                go.Bar(x=priority_stats.index, y=priority_stats['Wait Time'],
                       marker_color='#FFA502', text=priority_stats['Wait Time'],
                       texttemplate='%{text:.1f}d', name='Wait Time'),
                row=1, col=2
            )
        
        fig.add_trace(
            go.Bar(x=priority_stats.index, y=priority_stats['Count'],
                   marker_color='#2E7D32', text=priority_stats['Count'],
                   texttemplate='%{text:,}', name='Count'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Priority Analysis")
        fig.update_yaxes(title_text="Days", row=1, col=1)
        fig.update_yaxes(title_text="Days", row=1, col=2)
        fig.update_yaxes(title_text="Tickets", row=1, col=3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Blocked Tickets
    if 'daysBlocked' in df.columns:
        st.header("Blocked Tickets Analysis")
        
        blocked = df[df['daysBlocked'] > 0].copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Blocked Tickets", f"{len(blocked):,} ({len(blocked)/len(df)*100:.1f}%)")
        with col2:
            st.metric("Avg Blocked Time", f"{blocked['daysBlocked'].mean():.1f} days")
        with col3:
            st.metric("Max Blocked Time", f"{blocked['daysBlocked'].max():.1f} days")
        
        if len(blocked) > 0:
            fig = px.histogram(
                blocked[blocked['daysBlocked'] < 100],
                x='daysBlocked',
                nbins=50,
                title='Distribution of Blocking Time (< 100 days shown)',
                labels={'daysBlocked': 'Days Blocked'},
                color_discrete_sequence=['crimson']
            )
            fig.add_vline(x=blocked['daysBlocked'].median(), line_dash="dash",
                          annotation_text=f"Median: {blocked['daysBlocked'].median():.1f}d")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Trends Over Time
    if 'month' in df.columns and 'totalCycleTime' in df.columns:
        st.header("Trends Over Time")
        
        df_filtered = df[df['month'] != '2025-10'].copy()
        monthly = df_filtered.groupby('month').agg({
            'Key': 'count',
            'totalCycleTime': 'mean'
        }).round(1)
        monthly.columns = ['Tickets', 'Avg Cycle Time']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=monthly.index.astype(str), y=monthly['Tickets'],
                   name='Tickets Created', marker_color='lightblue'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=monthly.index.astype(str), y=monthly['Avg Cycle Time'],
                       name='Avg Cycle Time', mode='lines+markers',
                       line=dict(color='red', width=3)),
            secondary_y=True
        )
        
        fig.update_layout(title_text="Trends Over Time", height=500)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Tickets Created", secondary_y=False)
        fig.update_yaxes(title_text="Cycle Time (days)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.header("Summary")
    
    summary_metrics = []
    
    if 'activeDevelopmentTime' in df.columns:
        avg_active = df['activeDevelopmentTime'].mean()
        summary_metrics.append(("Average Active Development", f"{avg_active:.1f} days"))
    
    if 'totalWaitTime' in df.columns:
        avg_wait = df['totalWaitTime'].mean()
        summary_metrics.append(("Average Wait Time", f"{avg_wait:.1f} days"))
    
    if 'cycleTimeExcludingBacklog' in df.columns:
        summary_metrics.append(("Median Cycle Time (ex. backlog)", 
                               f"{df['cycleTimeExcludingBacklog'].median():.1f} days"))
        summary_metrics.append(("90th Percentile", 
                               f"{df['cycleTimeExcludingBacklog'].quantile(0.9):.1f} days"))
    
    if 'daysBlocked' in df.columns:
        blocked = df[df['daysBlocked'] > 0]
        summary_metrics.append(("Blocked Tickets", 
                               f"{len(blocked):,} ({len(blocked)/len(df)*100:.1f}%)"))
    
    if 'NumSprints' in df.columns:
        multi_sprint = df[df['NumSprints'] > 1]
        summary_metrics.append(("Multi-sprint Tickets", 
                               f"{len(multi_sprint):,} ({len(multi_sprint)/len(df)*100:.1f}%)"))
    
    if summary_metrics:
        summary_data = {
            "Metric": [m[0] for m in summary_metrics],
            "Value": [m[1] for m in summary_metrics]
        }
        st.table(pd.DataFrame(summary_data))

else:
    st.info("Please upload an Excel (.xlsx) file to begin.")