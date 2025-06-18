
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import Energy_ETL functions
try:
    from ETL_Script import load_processed_data, load_report, generate_demo_data
except ImportError:
    st.error("ETL_Script.py not found. Please check teh directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Eliq Energy Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    
    df = load_processed_data()
    if df is None:
        st.warning("No processed data found. Generating demo data...")
        df = generate_demo_data(60) 
    return df

@st.cache_data
def load_quality_report():
    return load_report()

def create_time_series_chart(df, metric="energy_consumption_kwh"):
    
    if metric == "daily_total":
        daily_df = df.groupby('date')[metric.replace('daily_total', 'energy_consumption_kwh')].sum().reset_index()
        daily_df.columns = ['date', 'value']
        title = "Daily Energy Consumption"
        y_label = "Daily Total (kWh)"
    else:
        daily_df = df.groupby('date')[metric].mean().reset_index()
        daily_df.columns = ['date', 'value']
        title = f"Average {metric.replace('_', ' ').title()}"
        y_label = metric.replace('_', ' ').title()
    
    fig = px.line(daily_df, x='date', y='value', title=title,
                  labels={'value': y_label, 'date': 'Date'})
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_hourly_pattern(df):
    
    hourly_avg = df.groupby(['hour_of_day', 'is_weekend'])['energy_consumption_kwh'].mean().reset_index()
    
    fig = px.line(hourly_avg, x='hour_of_day', y='energy_consumption_kwh', 
                  color='is_weekend', title='Hourly Consumption Pattern',
                  labels={'energy_consumption_kwh': 'Average Consumption (kWh)', 
                         'hour_of_day': 'Hour of Day', 'is_weekend': 'Weekend'})
    fig.update_layout(height=400)
    return fig

def create_seasonal_analysis(df):
    
    seasonal = df.groupby(['season', 'time_period'])['energy_consumption_kwh'].mean().reset_index()
    
    fig = px.bar(seasonal, x='season', y='energy_consumption_kwh', color='time_period',
                 title='Seasonal Consumption by Time Period',
                 labels={'energy_consumption_kwh': 'Average Consumption (kWh)'})
    fig.update_layout(height=400)
    return fig

# Heat maps for patterns
def create_heatmap(df):
    
    df['day_name'] = df['timestamp_local'].dt.day_name() if 'timestamp_local' in df.columns else pd.Categorical(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])[df['day_of_week']]
    
    heatmap_data = df.groupby(['hour_of_day', 'day_name'])['energy_consumption_kwh'].mean().unstack(fill_value=0)
    
    #TODO: Order the days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(columns=[day for day in day_order if day in heatmap_data.columns])
    
    fig = px.imshow(heatmap_data.values, 
                    labels=dict(x="Day of Week", y="Hour", color="Avg kWh"),
                    x=heatmap_data.columns, y=heatmap_data.index,
                    title="Energy Consumption Heatmap: Hour vs Day",
                    aspect="auto", color_continuous_scale="Viridis")
    fig.update_layout(height=500)
    return fig

def main():
    # Header
    st.title("Eliq Energy Analytics Dashboard")
    st.markdown("**Interactive analysis of energy consumption data**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/1f77b4/white?text=ELIQ+ENERGY", width=200)
        st.markdown("### Dashboard Controls")
        
        if st.button("Click to Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    

    if not Path("output/energy_data.parquet").exists():
        if Path("home_assignment_raw_data.parquet").exists():
            st.info("🔄 Processing original data... Please wait.")
            with st.spinner("Running ETL pipeline..."):
                try:
                    from ETL_Script import EnergyETL
                    etl = EnergyETL()
                    etl.run()
                    st.success(" Data processed ")
                    st.rerun()
                except Exception as e:
                    st.error(f"ETL failed: {e}")
                    st.stop()

    # Load data
    with st.spinner("Loading energy data..."):
        df = load_data()
        report = load_quality_report()
    
    if df is None or df.empty:
        st.error(" No data available")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([" Overview", "Analytics", " Query", " Report"])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_consumption = df['energy_consumption_kwh'].sum()
            st.metric("Total Consumption", f"{total_consumption:,.0f} kWh", 
                     delta=f"{len(df):,} records")
        
        with col2:
            avg_consumption = df['energy_consumption_kwh'].mean()
            st.metric("Average Hourly", f"{avg_consumption:.2f} kWh",
                     delta=f"±{df['energy_consumption_kwh'].std():.2f}")
        
        with col3:
            peak_consumption = df['energy_consumption_kwh'].max()
            st.metric("Peak Consumption", f"{peak_consumption:.2f} kWh",
                     delta="Max recorded")
        
        with col4:
            valid_pct = (df['data_quality_flag'] == 'valid').mean() * 100 if 'data_quality_flag' in df.columns else 100
            st.metric("Data Quality", f"{valid_pct:.1f}%", delta="Valid records")
        
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_time_series_chart(df, "daily_total"), use_container_width=True)
        with col2:
            st.plotly_chart(create_hourly_pattern(df), use_container_width=True)
        
        
        st.plotly_chart(create_heatmap(df), use_container_width=True)
    
    with tab2:
        st.markdown("### Advanced Analytics")
        
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analysis_type = st.selectbox("Analysis Type", 
                                       ["Seasonal Patterns", "Peak vs Off-Peak", "Weekend vs Weekday"])
            
            if 'season' in df.columns:
                selected_seasons = st.multiselect("Seasons", df['season'].unique(), 
                                                 default=df['season'].unique(), key="analytics_seasons")
            else:
                selected_seasons = None
        
        with col2:
            if analysis_type == "Seasonal Patterns":
                st.plotly_chart(create_seasonal_analysis(df), use_container_width=True)
            
            elif analysis_type == "Peak vs Off-Peak":
                if 'time_period' in df.columns:
                    peak_data = df.groupby('time_period')['energy_consumption_kwh'].agg(['mean', 'sum']).reset_index()
                    fig = px.bar(peak_data, x='time_period', y='mean', 
                               title='Peak vs Off-Peak Average Consumption')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Time period data not available")
            
            elif analysis_type == "Weekend vs Weekday":
                weekend_data = df.groupby('is_weekend')['energy_consumption_kwh'].agg(['mean', 'sum']).reset_index()
                weekend_data['period'] = weekend_data['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
                fig = px.bar(weekend_data, x='period', y='mean',
                           title='Weekend vs Weekday Average Consumption')
                st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Energy Consumption Stats")
            stats = df['energy_consumption_kwh'].describe().round(2)
            st.dataframe(stats)
        
        with col2:
            if 'time_period' in df.columns:
                st.markdown("#### Time Period Analysis")
                time_stats = df.groupby('time_period')['energy_consumption_kwh'].agg(['mean', 'std', 'count']).round(2)
                st.dataframe(time_stats)
    
    with tab3:
        st.markdown("### Custom Query Interface")
        
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Filters")
            
            # filters
            if 'date' in df.columns:
                date_range = st.date_input("Date Range", 
                                         value=(df['date'].min(), df['date'].max()),
                                         min_value=df['date'].min(),
                                         max_value=df['date'].max())
            
           
            hour_range = st.slider("Hour Range", 0, 23, (0, 23))
            
            
            include_weekend = st.checkbox("Include Weekends", True)
            
            
            if 'season' in df.columns:
                seasons = st.multiselect("Seasons", df['season'].unique(), 
                                       default=df['season'].unique(), key="query_seasons")
            
            
            consumption_range = st.slider("Consumption Range (kWh)", 
                                        float(df['energy_consumption_kwh'].min()),
                                        float(df['energy_consumption_kwh'].max()),
                                        (float(df['energy_consumption_kwh'].min()), 
                                         float(df['energy_consumption_kwh'].max())))
        
        with col2:
            filtered_df = df.copy()
            
            if 'date' in df.columns and len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
                    (filtered_df['date'] <= pd.to_datetime(date_range[1]))
                ]
            
            filtered_df = filtered_df[
                (filtered_df['hour_of_day'] >= hour_range[0]) &
                (filtered_df['hour_of_day'] <= hour_range[1])
            ]
            
            if not include_weekend:
                filtered_df = filtered_df[filtered_df['is_weekend'] == False]
            
            if 'season' in filtered_df.columns and seasons:
                filtered_df = filtered_df[filtered_df['season'].isin(seasons)]
            
            filtered_df = filtered_df[
                (filtered_df['energy_consumption_kwh'] >= consumption_range[0]) &
                (filtered_df['energy_consumption_kwh'] <= consumption_range[1])
            ]
            
            
            st.markdown("#### Query Results")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Records Found", f"{len(filtered_df):,}")
            with col_b:
                st.metric("Total Consumption", f"{filtered_df['energy_consumption_kwh'].sum():.1f} kWh")
            with col_c:
                st.metric("Average Consumption", f"{filtered_df['energy_consumption_kwh'].mean():.2f} kWh")
            
            
            st.markdown("#### Filtered Data Preview")
            st.dataframe(filtered_df.head(100), use_container_width=True)
            
            
            if st.button(" Export Results"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"eliq_energy_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.markdown("###  Processing Report")
        
        if report:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Processing Summary")
                st.json({
                    "Records Processed": report.get('records_processed', 'N/A'),
                    "Date Range": report.get('date_range', 'N/A'),
                    "Total Consumption": f"{report.get('total_consumption_kwh', 0):,.1f} kWh",
                    "Average Hourly": f"{report.get('avg_hourly_kwh', 0):.2f} kWh"
                })
            
            with col2:
                st.markdown("#### Data Quality")
                if 'data_quality' in report:
                    quality_df = pd.DataFrame(list(report['data_quality'].items()), 
                                            columns=['Quality Flag', 'Count'])
                    fig = px.pie(quality_df, values='Count', names='Quality Flag',
                               title='Data Quality Distribution')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(" No processing report available. Run the ETL pipeline to generate.")
        
        # Dataset info(Data_describe)
        st.markdown("#### Current Dataset Info")
        info_data = {
            "Total Records": len(df),
            "Date Range": f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A",
            "Columns": len(df.columns),
            "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
        }
        st.json(info_data)
    
    
if __name__ == "__main__":
    main()
