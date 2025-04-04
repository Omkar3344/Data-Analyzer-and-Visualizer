import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def plot_numerical_distribution(df, column, plot_type='histogram'):
    """
    Plot distribution of numerical data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to visualize
    plot_type : str
        Type of plot ('histogram', 'boxplot', 'violin', 'kde')
    """
    if plot_type == 'histogram':
        fig = px.histogram(df, x=column, 
                          title=f'Distribution of {column}',
                          marginal='box')
        
    elif plot_type == 'boxplot':
        fig = px.box(df, y=column, 
                    title=f'Boxplot of {column}')
        
    elif plot_type == 'violin':
        fig = px.violin(df, y=column, 
                       title=f'Violin Plot of {column}',
                       box=True)
        
    elif plot_type == 'kde':
        fig = px.density_contour(df, x=column,
                               title=f'KDE Plot of {column}')
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    
    return fig

def plot_categorical_distribution(df, column, plot_type='bar'):
    """
    Plot distribution of categorical data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to visualize
    plot_type : str
        Type of plot ('bar', 'pie')
    """
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # Limit to top 15 categories for better visualization
    if len(value_counts) > 15:
        other_count = value_counts.iloc[15:]['count'].sum()
        value_counts = value_counts.iloc[:15]
        value_counts = pd.concat([value_counts, pd.DataFrame({column: ['Other'], 'count': [other_count]})])
    
    if plot_type == 'bar':
        fig = px.bar(value_counts, x=column, y='count',
                    title=f'Distribution of {column}')
        
    elif plot_type == 'pie':
        fig = px.pie(value_counts, names=column, values='count',
                    title=f'Distribution of {column}')
    
    return fig

def plot_correlation_heatmap(df, columns=None):
    """
    Plot correlation heatmap for numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    columns : list
        List of columns to include in the correlation
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    corr = numeric_df.corr()
    
    fig = px.imshow(corr, 
                   text_auto=True, 
                   aspect="auto",
                   title='Correlation Heatmap',
                   color_continuous_scale='RdBu_r')
    
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, size_col=None):
    """
    Create scatter plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    color_col : str
        Column name for color encoding
    size_col : str
        Column name for size encoding
    """
    fig = px.scatter(df, x=x_col, y=y_col, 
                    color=color_col, 
                    size=size_col,
                    opacity=0.7,
                    title=f'Scatter Plot: {y_col} vs {x_col}')
    
    return fig

def plot_line(df, x_col, y_col, color_col=None):
    """
    Create line plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    color_col : str
        Column name for color encoding
    """
    fig = px.line(df, x=x_col, y=y_col, 
                 color=color_col,
                 title=f'Line Plot: {y_col} vs {x_col}')
    
    return fig

def plot_wordcloud(text):
    """
    Create wordcloud from text data
    
    Parameters:
    -----------
    text : str
        Text data to visualize
    """
    # Clean and tokenize text
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Generate and plot wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white', 
                         max_words=200,
                         contour_width=3).generate(' '.join(tokens))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def plot_pca(df, n_components=2):
    """
    Create PCA visualization
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing numerical data
    n_components : int
        Number of components to use
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Extract numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Remove columns with NaN values
    numeric_df = numeric_df.dropna(axis=1)
    
    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numerical columns for PCA")
        return None
    
    # Standardize the data
    X = StandardScaler().fit_transform(numeric_df)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, numeric_df.shape[1]))
    components = pca.fit_transform(X)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    
    # Add explained variance information
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Create scatter plot of first two components
    if components.shape[1] >= 2:
        # Convert NumPy values to Python floats for formatting
        var1 = float(explained_variance[0])
        var2 = float(explained_variance[1])
        
        fig = px.scatter(pca_df, x='PC1', y='PC2',
                        title=f'PCA Visualization (PC1: {var1:.2f}%, PC2: {var2:.2f}%)',
                        opacity=0.7)
        
        # Add loading vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        for i, feature in enumerate(numeric_df.columns):
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=loadings[i, 0],
                y1=loadings[i, 1],
                line=dict(color='red', width=1, dash='dot'),
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                font=dict(size=10)
            )
            
        return fig
    
    return None

def plot_text_stats(text_analysis):
    """
    Create visualizations from text stats
    
    Parameters:
    -----------
    text_analysis : dict
        Dictionary containing text analysis results
    """
    # Word frequency bar chart for top 15 words
    top_words = dict(list(text_analysis['top_words'].items())[:15])
    
    words_df = pd.DataFrame({
        'Word': list(top_words.keys()),
        'Frequency': list(top_words.values())
    }).sort_values('Frequency', ascending=False)
    
    fig = px.bar(words_df, x='Word', y='Frequency',
                title='Top 15 Words Frequency',
                color='Frequency')
    
    return fig

def create_visualization(data, viz_type, **kwargs):
    """
    Create visualization based on data and visualization type
    
    Parameters:
    -----------
    data : pd.DataFrame or str
        Data to visualize
    viz_type : str
        Type of visualization to create
    **kwargs : dict
        Additional parameters for specific visualization types
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        The generated visualization figure
    """
    # For tabular data
    if isinstance(data, pd.DataFrame):
        # Distribution visualizations
        if viz_type == "histogram":
            return plot_numerical_distribution(data, kwargs.get('column'), plot_type='histogram')
        elif viz_type == "boxplot":
            return plot_numerical_distribution(data, kwargs.get('column'), plot_type='boxplot')
        elif viz_type == "bar":
            return plot_categorical_distribution(data, kwargs.get('column'), plot_type='bar')
        elif viz_type == "pie":
            return plot_categorical_distribution(data, kwargs.get('column'), plot_type='pie')
        
        # Relationship visualizations
        elif viz_type == "scatter":
            return plot_scatter(data, kwargs.get('x_col'), kwargs.get('y_col'), 
                              kwargs.get('color_col'), kwargs.get('size_col'))
        elif viz_type == "line":
            return plot_line(data, kwargs.get('x_col'), kwargs.get('y_col'), kwargs.get('color_col'))
        elif viz_type == "correlation":
            return plot_correlation_heatmap(data, kwargs.get('columns'))
        
        # Advanced visualizations
        elif viz_type == "pca":
            return plot_pca(data, kwargs.get('n_components', 2))
    
    # For text data
    elif isinstance(data, str):
        if viz_type == "wordcloud":
            return plot_wordcloud(data)
        elif viz_type == "word_freq":
            analysis_results = kwargs.get('analysis_results', {})
            if 'top_words' in analysis_results:
                return plot_text_stats(analysis_results)
    
    # Return None if no matching visualization
    return None