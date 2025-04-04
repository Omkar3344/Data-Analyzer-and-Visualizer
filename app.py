import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO
import time
import os
import sys

# Add the current directory to path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Update imports to include all download link functions
from utils.file_handlers import (
    read_file, 
    infer_column_types, 
    convert_df_to_csv_download_link,
    convert_df_to_excel_download_link,
    convert_df_to_json_download_link
)
from utils.data_analysis import analyze_tabular_data, analyze_text_data, generate_data_summary
from utils.visualizations import create_visualization

# Page configuration
st.set_page_config(
    page_title="Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Base theme colors */
    :root {
        --background-color: #131315;
        --text-color: #ffffff;
        --accent-color: #c4e456;
        --secondary-color: #99b536;
        --dark-accent: #7a9215;
        --light-accent: #e8f7aa;
        --border-color: #2d2d30;
    }

    /* Override Streamlit's base styling */
    .stApp {
        background-color: var(--background-color);
    }
    
    .main .block-container {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown {
        color: var(--text-color) !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: var(--accent-color) !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0px 2px 2px rgba(0, 0, 0, 0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--accent-color) !important;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--secondary-color);
        padding-bottom: 0.5rem;
    }
    
    /* Text styling */
    .info-text {
        font-size: 1rem;
        color: #cccccc !important;
    }
    
    /* Box styling */
    .success-box {
        padding: 1.2rem;
        background-color: #1a1a1c;
        border-radius: 5px;
        border-left: 5px solid var(--accent-color);
        color: var(--text-color) !important;
        font-weight: 400;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .success-box h2 {
        color: var(--accent-color) !important;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .success-box ul li {
        color: #e0e0e0 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Metric styling */
    .css-1wivap2 {
        background-color: #1a1a1c !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .css-1wivap2 label {
        color: var(--accent-color) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-hxt7ib {
        background-color: #1a1a1c;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--accent-color);
        color: #000000;
        font-weight: 600;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        color: #000000;
    }
    
    /* DataFrame styling */
    .dataframe {
        background-color: #1a1a1c !important;
        color: #e0e0e0 !important;
    }
    
    .dataframe th {
        background-color: #2a2a2c !important;
        color: var(--accent-color) !important;
    }
    
    /* SelectBox styling */
    .stSelectbox label {
        color: var(--text-color) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1c;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
        background-color: #1a1a1c;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a2a2c !important;
        color: var(--accent-color) !important;
        border-bottom: 2px solid var(--accent-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'error' not in st.session_state:
    st.session_state.error = None

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>🔄 Data Input</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a file", 
                                     type=['csv', 'xlsx', 'xls', 'json'],
                                     help="Supported file types: CSV, Excel, and JSON")
    
    if uploaded_file is not None:
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_format = uploaded_file.name.split('.')[-1].lower()  # Add this line
        
        with st.spinner("Processing file..."):
            # Read the file
            data, data_type, error = read_file(uploaded_file)
            
            if error:
                st.session_state.error = error
                st.session_state.data = None
                st.session_state.data_type = None
                st.session_state.analysis_results = None
            else:
                st.session_state.error = None
                st.session_state.data = data
                st.session_state.data_type = data_type
                
                # Analyze data based on type
                if data_type == 'tabular':
                    st.session_state.column_types = infer_column_types(data)
                    st.session_state.analysis_results = analyze_tabular_data(data)
                    st.session_state.data_summary = generate_data_summary(data, 'tabular')
                elif data_type == 'text':
                    st.session_state.analysis_results = analyze_text_data(data)
                    st.session_state.data_summary = generate_data_summary(data, 'text')
                elif data_type == 'image':
                    # Image handling will be minimal in this version
                    st.session_state.data_summary = "Image analysis features are limited in this version."
                
                st.success(f"Successfully processed {uploaded_file.name}")
    
    # Replace the existing export section with this enhanced version
    if st.session_state.data is not None and st.session_state.data_type == 'tabular':
        st.markdown("<h2 class='sub-header'>💾 Export</h2>", unsafe_allow_html=True)
        
        # Add CSS for download links
        st.markdown("""
        <style>
        .download-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 16px;
        }
        .download-link {
            display: block;
            padding: 8px 12px;
            background-color: #1a1a1c;
            color: #c4e456 !important;
            text-align: center;
            border-radius: 4px;
            border: 1px solid #2d2d30;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s;
        }
        .download-link:hover {
            background-color: #2a2a2c;
            color: #e8f7aa !important;
            border-color: #c4e456;
        }
        </style>
        <div class="download-container">
        """, unsafe_allow_html=True)
        
        # Original file format (from session state)
        original_format = getattr(st.session_state, 'file_format', '').lower()
        
        # Show download links for formats different from the original
        if original_format != 'csv':
            st.markdown(convert_df_to_csv_download_link(st.session_state.data), unsafe_allow_html=True)
        
        if original_format != 'xlsx' and original_format != 'xls':
            st.markdown(convert_df_to_excel_download_link(st.session_state.data), unsafe_allow_html=True)
        
        if original_format != 'json':
            st.markdown(convert_df_to_json_download_link(st.session_state.data), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>ℹ️ About</h2>", unsafe_allow_html=True)
    
    # About section with proper HTML rendering

    about_html = '<div style="background-color: #1a1a1c; padding: 15px; border-radius: 5px; border-left: 4px solid #c4e456;">'
    about_html += '<h4 style="color: #c4e456;">Data Visualizer & Analyzer</h4>'
    about_html += '<p style="font-size: 1rem; color: #e0e0e0;">This interactive data analysis tool helps you quickly understand and visualize your datasets without coding. Perfect for data exploration, pattern discovery, and generating insights.</p>'
    about_html += '<h5 style="color: #c4e456; margin-top: 15px;">✨ Key Features</h5>'
    about_html += '<ul style="font-size: 1rem; color: #cccccc; margin-bottom: 12px;">'
    about_html += '<li><strong style="color: #e0e0e0;">Automated Analysis:</strong> Instant statistical summaries and data quality assessment</li>'
    about_html += '<li><strong style="color: #e0e0e0;">Interactive Visualizations:</strong> Explore your data through customizable charts</li>'
    about_html += '<li><strong style="color: #e0e0e0;">Smart Insights:</strong> Discover patterns, outliers, and relationships</li>'
    about_html += '<li><strong style="color: #e0e0e0;">Export Capabilities:</strong> Save processed data and visualizations</li>'
    about_html += '</ul>'
    about_html += '<h5 style="color: #c4e456; margin-top: 15px;">📊 Supported Formats</h5>'
    about_html += '<p style="font-size: 1rem; color: #e0e0e0;">Currently supports CSV, Excel (.xlsx, .xls), and JSON files with tabular data structure.</p>'
    about_html += '<h5 style="color: #c4e456; margin-top: 15px;">💡 Tips</h5>'
    about_html += '<p style="font-size: 1rem; color: #e0e0e0;">For best results, ensure your data has clean column names and appropriate data types. Larger datasets (>100k rows) may experience slower performance.</p>'
    about_html += '</div>'

    st.markdown(about_html, unsafe_allow_html=True)

# Main content area
st.markdown("<h1 class='main-header'>Data Visualizer & Analyzer</h1>", unsafe_allow_html=True)

if st.session_state.error:
    st.error(st.session_state.error)
elif st.session_state.data is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Data Insights", "Summary"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>📊 Data Overview</h2>", unsafe_allow_html=True)
        
        if st.session_state.data_type == 'tabular':
            rows, cols = st.session_state.data.shape
            st.markdown(f"<p class='info-text'>Dataset has {rows} rows and {cols} columns</p>", unsafe_allow_html=True)
            
            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Preview")
                st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            with col2:
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': st.session_state.data.dtypes.index,
                    'Data Type': st.session_state.data.dtypes.values,
                    'Inferred Type': [st.session_state.column_types.get(col, 'unknown') for col in st.session_state.data.columns]
                })
                st.dataframe(dtypes_df, use_container_width=True)
                
                # Missing values summary
                st.subheader("Missing Values")
                missing = st.session_state.data.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': np.round((missing.values / len(st.session_state.data)) * 100, 2)
                }).sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
        
        elif st.session_state.data_type == 'text':
            # Display text preview and basic stats
            st.subheader("Text Preview")
            preview_len = min(1000, len(st.session_state.data))
            st.text_area("", st.session_state.data[:preview_len] + ('...' if len(st.session_state.data) > preview_len else ''), height=200)
            
            # Basic text stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Characters", len(st.session_state.data))
            col2.metric("Words", len(st.session_state.data.split()))
            col3.metric("Sentences", len([s for s in st.session_state.data.split('.') if s.strip()]))
            col4.metric("Paragraphs", len(st.session_state.data.split('\n\n')))
        
        elif st.session_state.data_type == 'image':
            # Display image
            st.subheader("Image Preview")
            st.image(st.session_state.data, caption=st.session_state.file_name)
            st.markdown("Image analysis features are limited in this app version.")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>🎨 Visualizations</h2>", unsafe_allow_html=True)
        
        if st.session_state.data_type == 'tabular':
            # Visualization selector
            viz_type = st.selectbox("Select visualization type", 
                                   ["Distribution", "Correlation", "Relationship", "Composition", "Comparison", "Trends"])
            
            if viz_type == "Distribution":
                # Distribution visualizations
                st.subheader("Distribution Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        selected_col = st.selectbox("Select column for histogram", num_cols)
                        
                        fig = px.histogram(st.session_state.data, x=selected_col, 
                                         marginal="box", title=f"Distribution of {selected_col}",
                                         color_discrete_sequence=['#c4e456'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    cat_cols = [col for col, type in st.session_state.column_types.items() 
                               if type in ['categorical', 'text'] and st.session_state.data[col].nunique() < 15]
                    
                    if cat_cols:
                        selected_cat = st.selectbox("Select categorical column", cat_cols)
                        
                        # Create a DataFrame from value counts with explicit column names
                        value_counts_df = st.session_state.data[selected_cat].value_counts().reset_index()
                        value_counts_df.columns = ['Category', 'Count']
                        
                        fig = px.bar(value_counts_df, 
                                   x='Category', y='Count', 
                                   title=f"Counts of {selected_cat}",
                                   labels={'Category': selected_cat, 'Count': 'Count'},
                                   color='Category',
                                   color_discrete_sequence=px.colors.sequential.Viridis)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Correlation":
                # Correlation visualizations
                st.subheader("Correlation Analysis")
                
                num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) > 1:
                    corr_cols = st.multiselect("Select columns for correlation matrix", 
                                             num_cols, default=num_cols[:min(5, len(num_cols))])
                    
                    if corr_cols and len(corr_cols) > 1:
                        corr = st.session_state.data[corr_cols].corr()
                        
                        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                                      title="Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numerical columns for correlation analysis")
            
            elif viz_type == "Relationship":
                # Relationship visualizations
                st.subheader("Relationship Analysis")
                
                num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_col = st.selectbox("Select X axis", num_cols, index=0)
                    
                    with col2:
                        y_col = st.selectbox("Select Y axis", num_cols, index=min(1, len(num_cols)-1))
                    
                    cat_cols = [col for col, type in st.session_state.column_types.items() 
                               if type in ['categorical', 'text'] and st.session_state.data[col].nunique() < 10]
                    
                    color_col = None
                    if cat_cols:
                        color_col = st.selectbox("Color points by (optional)", ["None"] + cat_cols)
                        if color_col == "None":
                            color_col = None
                    
                    # Create scatter plot
                    fig = px.scatter(st.session_state.data, x=x_col, y=y_col, 
                                   color=color_col, title=f"{y_col} vs {x_col}",
                                   opacity=0.8, size_max=10, 
                                   trendline="ols" if color_col is None else None,
                                   color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numerical columns for relationship analysis")
            
            elif viz_type == "Composition":
                # Composition visualizations
                st.subheader("Composition Analysis")
                
                cat_cols = [col for col, type in st.session_state.column_types.items() 
                           if type in ['categorical', 'text'] and st.session_state.data[col].nunique() < 15]
                
                if cat_cols:
                    selected_cat = st.selectbox("Select column for pie chart", cat_cols)
                    
                    pie_data = st.session_state.data[selected_cat].value_counts().reset_index()
                    pie_data.columns = [selected_cat, 'count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig = px.pie(pie_data, values='count', names=selected_cat, 
                                   title=f"Composition of {selected_cat}",
                                   color_discrete_sequence=px.colors.sequential.Viridis)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Treemap
                        fig = px.treemap(pie_data, path=[selected_cat], values='count',
                                        title=f"Treemap of {selected_cat}",
                                        color='count', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No suitable categorical columns for composition analysis")
            
            elif viz_type == "Comparison":
                # Comparison visualizations
                st.subheader("Comparison Analysis")
                
                cat_cols = [col for col, type in st.session_state.column_types.items() 
                           if type in ['categorical', 'text'] and 1 < st.session_state.data[col].nunique() < 15]
                num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                
                if cat_cols and num_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cat_col = st.selectbox("Select categorical column", cat_cols)
                    
                    with col2:
                        num_col = st.selectbox("Select numerical column", num_cols)
                    
                    # Box plot
                    fig = px.box(st.session_state.data, x=cat_col, y=num_col, 
                               title=f"Comparison of {num_col} by {cat_col}",
                               color=cat_col, notched=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart with error bars
                    agg_data = st.session_state.data.groupby(cat_col)[num_col].agg(['mean', 'std']).reset_index()
                    fig = px.bar(agg_data, x=cat_col, y='mean', 
                               error_y='std', title=f"Mean {num_col} by {cat_col} (with std dev)",
                               color=cat_col, labels={'mean': f'Mean {num_col}'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need both categorical and numerical columns for comparison analysis")
            
            elif viz_type == "Trends":
                # Time series and trend visualizations
                st.subheader("Trend Analysis")
                
                # Check for datetime columns
                datetime_cols = st.session_state.data.select_dtypes(include=['datetime']).columns.tolist()
                potential_datetime = [col for col in st.session_state.data.columns 
                                    if col.lower().find('date') >= 0 or col.lower().find('time') >= 0]
                
                date_col = None
                if datetime_cols:
                    date_col = st.selectbox("Select date/time column", datetime_cols)
                elif potential_datetime:
                    date_col = st.selectbox("Select potential date/time column", potential_datetime)
                    # Try to convert to datetime
                    try:
                        st.session_state.data[date_col] = pd.to_datetime(st.session_state.data[date_col])
                    except:
                        st.error(f"Could not convert {date_col} to date/time format")
                        date_col = None
                
                if date_col:
                    num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        value_col = st.selectbox("Select value column", num_cols)
                        
                        # Time series plot
                        fig = px.line(st.session_state.data.sort_values(date_col), 
                                    x=date_col, y=value_col, 
                                    title=f"{value_col} over Time",
                                    markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Rolling average
                        window = st.slider("Rolling average window", min_value=2, max_value=30, value=7)
                        temp_df = st.session_state.data.sort_values(date_col).copy()
                        temp_df[f'{value_col}_rolling'] = temp_df[value_col].rolling(window=window).mean()
                        
                        fig = px.line(temp_df, x=date_col, y=[value_col, f'{value_col}_rolling'],
                                    title=f"{value_col} with {window}-period Rolling Average",
                                    labels={f'{value_col}_rolling': f'{window}-period Rolling Avg'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numerical columns available for trend analysis")
                else:
                    st.info("No date/time columns detected. Trend analysis requires temporal data.")
        
        elif st.session_state.data_type == 'text':
            # Text visualizations
            st.subheader("Text Visualizations")
            
            # Word cloud
            st.markdown("### Word Cloud")
            if st.session_state.analysis_results and 'top_words' in st.session_state.analysis_results:
                word_freq = st.session_state.analysis_results['top_words']
                
                if word_freq:
                    wc = WordCloud(background_color="white", max_words=100, 
                                  max_font_size=40, width=800, height=400).generate_from_frequencies(word_freq)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt.gcf())
                
                # Top words bar chart
                st.markdown("### Top Words")
                top_n = min(20, len(word_freq))
                top_words = dict(list(word_freq.items())[:top_n])
                
                fig = px.bar(x=list(top_words.keys()), y=list(top_words.values()),
                           labels={'x': 'Word', 'y': 'Frequency'},
                           title=f"Top {top_n} Words",
                           color=list(top_words.values()),
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Readability metrics
                st.markdown("### Readability Metrics")
                if 'readability' in st.session_state.analysis_results:
                    readability = st.session_state.analysis_results['readability']
                    
                    metrics_df = pd.DataFrame({
                        'Metric': list(readability.keys()),
                        'Score': list(readability.values())
                    })
                    
                    fig = px.bar(metrics_df, x='Metric', y='Score',
                               title="Readability Scores",
                               color='Score', color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>🔍 Data Insights</h2>", unsafe_allow_html=True)
        
        if st.session_state.data_type == 'tabular' and st.session_state.analysis_results:
            # Display key insights from analysis results
            
            # Potential outliers
            if 'potential_outliers' in st.session_state.analysis_results and st.session_state.analysis_results['potential_outliers']:
                st.subheader("Potential Outliers")
                
                outliers = st.session_state.analysis_results['potential_outliers']
                outlier_df = pd.DataFrame({
                    'Column': list(outliers.keys()),
                    'Outlier Count': [o['count'] for o in outliers.values()],
                    'Percentage': [f"{o['percentage']:.2f}%" for o in outliers.values()]
                })
                
                st.dataframe(outlier_df, use_container_width=True)
                
                # Visualize outliers
                if outlier_df.shape[0] > 0:
                    selected_col = st.selectbox("View outliers for column", outlier_df['Column'].tolist())
                    
                    # Box plot
                    fig = px.box(st.session_state.data, y=selected_col, 
                               title=f"Box Plot of {selected_col} Showing Outliers",
                               points="outliers")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Missing values patterns
            st.subheader("Missing Values Patterns")
            missing = st.session_state.data.isnull().sum()
            cols_with_missing = missing[missing > 0]
            
            if len(cols_with_missing) > 0:
                # Visualize missing values
                fig = px.bar(x=cols_with_missing.index, y=cols_with_missing.values,
                           labels={'x': 'Column', 'y': 'Missing Count'},
                           title="Missing Values by Column",
                           color=cols_with_missing.values,
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
                # Missing value correlation
                if len(cols_with_missing) > 1:
                    st.markdown("#### Missing Value Correlation")
                    st.markdown("This shows if missing values in different columns tend to occur together.")
                    
                    # Create binary missing value indicators
                    missing_df = pd.DataFrame()
                    for col in cols_with_missing.index:
                        missing_df[f"{col}_missing"] = st.session_state.data[col].isnull().astype(int)
                    
                    # Calculate correlation
                    missing_corr = missing_df.corr()
                    
                    # Visualize correlation
                    fig = px.imshow(missing_corr, text_auto=True,
                                   title="Missing Value Correlation",
                                   color_continuous_scale='RdBu_r',
                                   zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in the dataset!")
            
            # Skewness analysis
            st.subheader("Distribution Skewness")
            num_cols = st.session_state.data.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                skewness = st.session_state.data[num_cols].skew().sort_values(ascending=False)
                
                skew_df = pd.DataFrame({
                    'Column': skewness.index,
                    'Skewness': skewness.values
                })
                
                # Add skewness interpretation
                def interpret_skewness(skew_val):
                    if abs(skew_val) < 0.5:
                        return "Approximately Symmetric"
                    elif 0.5 <= skew_val < 1:
                        return "Moderately Positively Skewed"
                    elif skew_val >= 1:
                        return "Highly Positively Skewed"
                    elif -1 < skew_val <= -0.5:
                        return "Moderately Negatively Skewed"
                    else:  # skew_val <= -1
                        return "Highly Negatively Skewed"
                
                skew_df['Interpretation'] = skew_df['Skewness'].apply(interpret_skewness)
                
                # Visualize skewness
                fig = px.bar(skew_df, x='Column', y='Skewness',
                           title="Skewness by Column",
                           color='Skewness',
                           color_continuous_scale='RdBu_r',
                           labels={'Skewness': 'Skewness (>0 = right-tailed, <0 = left-tailed)'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table
                st.dataframe(skew_df, use_container_width=True)
            
            # PCA visualization if available
            if 'pca' in st.session_state.analysis_results:
                st.subheader("Principal Component Analysis")
                
                pca_data = st.session_state.analysis_results['pca']
                explained_var = pca_data['explained_variance_ratio']
                cumulative_var = pca_data['cumulative_variance_ratio']
                
                # Create DataFrame for plotting
                pca_df = pd.DataFrame({
                    'Principal Component': [f"PC{i+1}" for i in range(len(explained_var))],
                    'Explained Variance (%)': [v * 100 for v in explained_var],
                    'Cumulative Variance (%)': [v * 100 for v in cumulative_var]
                })
                
                # Plot
                fig = px.bar(pca_df, x='Principal Component', y='Explained Variance (%)',
                           title="Variance Explained by Principal Components",
                           color='Explained Variance (%)',
                           color_continuous_scale='Viridis')
                
                # Add cumulative variance line
                fig.add_trace(
                    go.Scatter(
                        x=pca_df['Principal Component'],
                        y=pca_df['Cumulative Variance (%)'],
                        mode='lines+markers',
                        name='Cumulative Variance (%)',
                        line=dict(color='red', width=3),
                        yaxis='y2'
                    )
                )
                
                # Update layout for secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title='Cumulative Variance (%)',
                        titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        anchor='x',
                        overlaying='y',
                        side='right'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpreting PCA results
                components_needed = next((i+1 for i, v in enumerate(cumulative_var) if v >= 0.8), len(cumulative_var))
                st.markdown(f"""
                #### PCA Interpretation
                
                - The first principal component explains **{explained_var[0]*100:.2f}%** of the variance.
                - **{components_needed}** principal components are needed to explain at least 80% of the variance.
                - This suggests the data has an **{'intrinsic dimensionality of ' + str(components_needed) if components_needed < len(explained_var) else 'high dimensionality'}**.
                """)
        
        elif st.session_state.data_type == 'text' and st.session_state.analysis_results:
            # Text analysis insights
            st.subheader("Text Insights")
            
            # Display readability interpretation
            if 'readability' in st.session_state.analysis_results:
                flesch_score = st.session_state.analysis_results['readability']['flesch_reading_ease']
                
                # Determine education level based on score
                if flesch_score >= 90:
                    grade_level = "5th grade"
                    difficulty = "Very Easy"
                elif flesch_score >= 80:
                    grade_level = "6th grade"
                    difficulty = "Easy"
                elif flesch_score >= 70:
                    grade_level = "7th grade"
                    difficulty = "Fairly Easy"
                elif flesch_score >= 60:
                    grade_level = "8th-9th grade"
                    difficulty = "Standard"
                elif flesch_score >= 50:
                    grade_level = "10th-12th grade"
                    difficulty = "Fairly Difficult"
                elif flesch_score >= 30:
                    grade_level = "College"
                    difficulty = "Difficult"
                else:
                    grade_level = "College Graduate"
                    difficulty = "Very Difficult"
                
                st.markdown(f"""
                #### Readability Analysis
                
                The text has a Flesch Reading Ease score of **{flesch_score:.2f}**, which is classified as "**{difficulty}**" to read.
                
                This corresponds to approximately a **{grade_level}** education level.
                """)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Flesch Reading Ease", f"{flesch_score:.2f}", f"{difficulty}")
                col2.metric("SMOG Index", f"{st.session_state.analysis_results['readability']['smog_index']:.2f}")
                col3.metric("Coleman-Liau Index", f"{st.session_state.analysis_results['readability']['coleman_liau_index']:.2f}")
            
            # Text statistics insights
            st.subheader("Text Statistics")
            
            word_count = st.session_state.analysis_results.get('word_count', 0)
            char_count = st.session_state.analysis_results.get('char_count', 0)
            sentence_count = st.session_state.analysis_results.get('sentence_count', 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Word Length", f"{char_count / max(1, word_count):.2f} chars")
            col2.metric("Average Sentence Length", f"{word_count / max(1, sentence_count):.2f} words")
            col3.metric("Words per Character", f"{word_count / max(1, char_count):.4f}")
            
            # Word frequency analysis
            if 'top_words' in st.session_state.analysis_results and st.session_state.analysis_results['top_words']:
                st.subheader("Vocabulary Richness")
                
                unique_words = len(st.session_state.analysis_results['top_words'])
                
                col1, col2 = st.columns(2)
                col1.metric("Unique Words", unique_words)
                col2.metric("Vocabulary Richness", f"{unique_words / max(1, word_count):.4f}",
                          help="Ratio of unique words to total words")
    
    with tab4:
        st.markdown("<h2 class='sub-header'>📑 Summary</h2>", unsafe_allow_html=True)
        
        if st.session_state.data_summary:
            st.markdown(st.session_state.data_summary)
            
            # Generate additional insights based on data type
            if st.session_state.data_type == 'tabular':
                st.subheader("Data Quality Assessment")
                
                # Calculate quality score
                missing_pct = sum(item['percentage'] for item in st.session_state.analysis_results.get('missing_values', {}).values()) / len(st.session_state.analysis_results.get('missing_values', {})) if st.session_state.analysis_results.get('missing_values', {}) else 0
                
                outlier_cols = len(st.session_state.analysis_results.get('potential_outliers', {}))
                total_cols = len(st.session_state.data.columns)
                outlier_pct = (outlier_cols / total_cols) * 100 if total_cols > 0 else 0
                
                # Score from 0-100
                completeness_score = 100 - missing_pct
                outlier_score = 100 - outlier_pct
                overall_quality = (completeness_score * 0.6) + (outlier_score * 0.4)
                
                # Display scores
                col1, col2, col3 = st.columns(3)
                col1.metric("Completeness", f"{completeness_score:.2f}%")
                col2.metric("Outlier-Free Score", f"{outlier_score:.2f}%")
                col3.metric("Overall Quality", f"{overall_quality:.2f}%")
                
                # Quality interpretation
                if overall_quality >= 90:
                    quality_text = "Excellent"
                    recommendation = "The data is of very high quality and ready for analysis."
                elif overall_quality >= 80:
                    quality_text = "Good"
                    recommendation = "The data is of good quality with minor issues that might need attention."
                elif overall_quality >= 70:
                    quality_text = "Fair"
                    recommendation = "The data has some quality issues that should be addressed before detailed analysis."
                else:
                    quality_text = "Poor"
                    recommendation = "The data has significant quality issues that require cleaning and preprocessing."
                
                st.markdown(f"""
                #### Data Quality Assessment: {quality_text}
                
                {recommendation}
                
                **Recommendations:**
                """)
                
                if missing_pct > 10:
                    st.markdown("- Consider imputing or removing columns with high missing values")
                if outlier_pct > 20:
                    st.markdown("- Investigate and potentially transform columns with outliers")
                
                # Correlation insights
                if 'correlation' in st.session_state.analysis_results:
                    st.subheader("Correlation Insights")
                    
                    corr_matrix = pd.DataFrame(st.session_state.analysis_results['correlation'])
                    
                    # Find highest correlations
                    corr_pairs = []
                    for col1 in corr_matrix.columns:
                        for col2 in corr_matrix.columns:
                            if col1 != col2 and col1 < col2:  # Avoid duplicates
                                corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
                    
                    # Sort by absolute correlation
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    if corr_pairs:
                        st.markdown("**Strongest Relationships:**")
                        for col1, col2, corr in corr_pairs[:5]:
                            direction = "positive" if corr > 0 else "negative"
                            strength = "very strong" if abs(corr) > 0.8 else "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak"
                            st.markdown(f"- **{col1}** and **{col2}** have a {strength} {direction} correlation ({corr:.3f})")
            
            elif st.session_state.data_type == 'text':
                st.subheader("Text Content Summary")
                
                # Add sentiment analysis message as a placeholder
                st.info("Detailed sentiment analysis and topic modeling features coming in a future update!")
                
                # Simulate topic extraction from most frequent words
                if 'top_words' in st.session_state.analysis_results:
                    top_words = list(st.session_state.analysis_results['top_words'].keys())[:20]
                    
                    st.markdown("**Potentially Important Topics:**")
                    # Show top words in groups of 3-4 as "topics"
                    for i in range(0, min(len(top_words), 15), 4):
                        topic_words = top_words[i:i+4]
                        st.markdown(f"- Topic {i//4 + 1}: {', '.join(['**' + word + '**' for word in topic_words])}")

else:
    # Show instructional content when no file is uploaded
    st.markdown("""
    <div class="success-box">
    <h2>Welcome to Data Visualizer & Analyzer</h2>
    <p>Upload a file using the sidebar to begin analyzing your data.</p>
    <p>This app supports the following file types:</p>
    <ul>
        <li><strong>CSV files</strong> (.csv) - Comma-separated values</li>
        <li><strong>Excel files</strong> (.xlsx, .xls) - Microsoft Excel spreadsheets</li>
        <li><strong>JSON files</strong> (.json) - JavaScript Object Notation</li>
    </ul>
    <p>After uploading, you can:</p>
    <ul>
        <li>View an overview of your data</li>
        <li>Create various visualizations</li>
        <li>Explore data insights</li>
        <li>Get a comprehensive summary</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <h3 style="text-align: center;">📊 Data Overview</h3>
        <p style="text-align: center;">Get quick insights into your dataset structure, columns, and statistics.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h3 style="text-align: center;">📈 Visualizations</h3>
        <p style="text-align: center;">Generate charts and plots to explore relationships and distributions.</p>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <h3 style="text-align: center;">🔍 Data Quality</h3>
        <p style="text-align: center;">Identify missing values, outliers, and potential data issues.</p>
        """, unsafe_allow_html=True)