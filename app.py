import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="MovieMate",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== THEME CONFIGURATION ====================
# genre mapping to single words
GENRE_MAPPING = {
    'Science Fiction': 'SciFi',
    'TV Movie': 'TVMovie',
    'Action': 'Action',
    'Adventure': 'Adventure',
    'Animation': 'Animation',
    'Comedy': 'Comedy',
    'Crime': 'Crime',
    'Documentary': 'Documentary',
    'Drama': 'Drama',
    'Family': 'Family',
    'Fantasy': 'Fantasy',
    'History': 'History',
    'Horror': 'Horror',
    'Music': 'Music',
    'Mystery': 'Mystery',
    'Romance': 'Romance',
    'Thriller': 'Thriller',
    'War': 'War',
    'Western': 'Western'
}

LANGUAGE_MAPPING = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'pl': 'Polish'
}

def get_theme_css():
    """Generate CSS based on theme mode"""
    bg_primary = "#4c3d19"           # background utama
    bg_secondary = "#354024"         # background sekunder
    accent_primary = "#889063"       # accent warna 1
    accent_secondary = "#e5ada8"     # accent warna 2 (PINK)
    text_primary = "#efe8d8"         # text utama
    text_secondary = "#d8d7b2"       # text sekunder
    card_bg = "#e5d7c4"              # background card
    
    return f"""
    <style>
        .main {{
            background-color: {bg_primary};
        }}
        .main-header {{
            font-size: 3rem;
            font-weight: bold;
            color: {accent_secondary};
            text-align: center;
            margin-bottom: 0.5rem;
        }}
        .tagline {{
            font-size: 1.2rem;
            color: {accent_primary};
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }}
        .logo-container {{
            text-align: center;
            margin: 2rem 0;
        }}
        .logo-container img {{
            max-width: 200px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .kpi-card {{
            background: linear-gradient(135deg, {accent_primary} 0%, 
                        {accent_secondary} 100%);
            padding: 20px;
            border-radius: 10px;
            color: {text_primary};
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}
        .kpi-label {{
            font-size: 1rem;
            opacity: 0.9;
        }}
        .movie-card {{
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(150deg, {bg_secondary} 0%, {text_secondary} 50%, {accent_secondary} 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: {text_primary};
        }}
        .movie-card h4 {{
            margin-bottom: 20px;
        }}
        .movie-card p {{
            color: {bg_secondary};
            margin: 0;
        }}
        .movie-card .top {{
            margin-bottom:20px;
        }}
        .section-header {{
            color: {text_secondary};
            font-weight: bold;
        }}
        .stSelectbox label, .stSlider label, .stRadio label, 
        .stMultiSelect label, .stTextInput label, .stNumberInput label {{
            color: {text_secondary} !important;
        }}
        div[data-testid="column"]:has(button[kind="primary"]) {{
            display: flex;
            align-items: flex-end;
        }}
        
        .stButton > button[kind="primary"] {{
            background-color: {accent_secondary} !important;
            border-color: {accent_secondary} !important;
            color: {bg_secondary} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: #d99a95 !important;
            border-color: #d99a95 !important;
        }}
        .stButton > button[kind="primary"]:active {{
            background-color: #cc8882 !important;
            border-color: #cc8882 !important;
        }}
        
        .stSlider [role="slider"] {{
            background-color: {accent_secondary} !important;
            border-color: {accent_secondary} !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background-color: {accent_secondary} !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div > div {{
            background-color: {accent_secondary} !important;
        }}
        .stSlider [role="slider"]:hover {{
            background-color: #d99a95 !important;
        }}
        
        .stSlider * {{
            color: {accent_secondary} !important;
        }}
        .stSlider p {{
            color: {accent_secondary} !important;
        }}
        .stSlider div {{
            color: {accent_secondary} !important;
        }}
        .stSlider span {{
            color: {accent_secondary} !important;
        }}
        /* Tick bar values */
        [data-testid="stTickBar"] {{
            color: {accent_secondary} !important;
        }}
        [data-testid="stTickBar"] * {{
            color: {accent_secondary} !important;
        }}
        [data-testid="stTickBar"] div {{
            color: {accent_secondary} !important;
        }}
        /* Thumb values */
        [data-testid="stThumbValue"] {{
            color: {accent_secondary} !important;
        }}
        .stSlider [data-testid="stTickBar"] + div {{
            color: {accent_secondary} !important;
        }}
        .stSlider [data-testid="stTickBar"] ~ div {{
            color: {accent_secondary} !important;
        }}
        
        .stRadio [role="radiogroup"] label:hover {{
            background-color: rgba(229, 173, 168, 0.1) !important;
        }}
        .stRadio [data-baseweb="radio"] > div:first-child {{
            background-color: transparent !important;
            border-color: {accent_secondary} !important;
        }}
        .stRadio [data-baseweb="radio"][aria-checked="true"] > div:first-child {{
            background-color: {accent_secondary} !important;
            border-color: {accent_secondary} !important;
        }}
        .stRadio [data-baseweb="radio"][aria-checked="true"] > div:first-child::after {{
            background-color: {bg_secondary} !important;
        }}
        
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {accent_secondary} !important;
            color: {bg_secondary} !important;
        }}
        .stMultiSelect [data-baseweb="tag"] svg {{
            fill: {bg_secondary} !important;
        }}
        .stMultiSelect [data-baseweb="select"] > div {{
            border-color: {text_secondary} !important;
        }}
        .stMultiSelect [data-baseweb="select"]:focus-within > div {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 1px {accent_secondary} !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 1px {accent_secondary} !important;
        }}
        
        .stNumberInput > div > div > input:focus {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 1px {accent_secondary} !important;
        }}
        .stNumberInput button {{
            color: {accent_secondary} !important;
        }}
        .stNumberInput button:hover {{
            background-color: rgba(229, 173, 168, 0.1) !important;
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            border-color: {text_secondary} !important;
        }}
        .stSelectbox [data-baseweb="select"]:focus-within > div {{
            border-color: {accent_secondary} !important;
            box-shadow: 0 0 0 1px {accent_secondary} !important;
        }}
        
        .streamlit-expanderHeader {{
            color: {text_secondary} !important;
        }}
        .streamlit-expanderHeader:hover {{
            background-color: rgba(229, 173, 168, 0.05) !important;
        }}
        .streamlit-expanderHeader svg {{
            fill: {accent_secondary} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {text_secondary} !important;
        }}
        [data-testid="stMetricDelta"] svg {{
            fill: {accent_secondary} !important;
        }}
        
        .stAlert {{
            background-color: rgba(229, 173, 168, 0.1) !important;
            border-left-color: {accent_secondary} !important;
        }}
        
        .stSpinner > div {{
            border-top-color: {accent_secondary} !important;
        }}
        
        .stProgress > div > div > div {{
            background-color: {accent_secondary} !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: rgba(229, 173, 168, 0.2) !important;
            border-bottom-color: {accent_secondary} !important;
            color: {accent_secondary} !important;
        }}
        .stTabs [data-baseweb="tab-list"] button:hover {{
            background-color: rgba(229, 173, 168, 0.1) !important;
        }}
        
        .stCheckbox input[type="checkbox"]:checked + div {{
            background-color: {accent_secondary} !important;
            border-color: {accent_secondary} !important;
        }}
    </style>
    """

# ==================== ALGORITHM IMPLEMENTATIONS ====================
def quick_sort(arr, key=lambda x: x, reverse=False):
    """Quick Sort Algorithm - O(n log n) average case"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    pivot_val = key(pivot)
    
    if reverse:
        left = [x for x in arr if key(x) > pivot_val]
        middle = [x for x in arr if key(x) == pivot_val]
        right = [x for x in arr if key(x) < pivot_val]
    else:
        left = [x for x in arr if key(x) < pivot_val]
        middle = [x for x in arr if key(x) == pivot_val]
        right = [x for x in arr if key(x) > pivot_val]
    
    return (quick_sort(left, key, reverse) + middle +
            quick_sort(right, key, reverse))

def binary_search(arr, target, key=lambda x: x):
    """Binary Search Algorithm - O(log n)"""
    left, right = 0, len(arr) - 1
    results = []
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = key(arr[mid])
        
        if mid_val == target:
            i = mid
            while i >= 0 and key(arr[i]) == target:
                results.append(arr[i])
                i -= 1
            i = mid + 1
            while i < len(arr) and key(arr[i]) == target:
                results.append(arr[i])
                i += 1
            return results
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return results

def linear_search(arr, query, key=lambda x: x):
    """Linear Search Algorithm - O(n)"""
    results = []
    query_lower = str(query).lower()
    
    for item in arr:
        if query_lower in str(key(item)).lower():
            results.append(item)
    
    return results

def merge_sort(arr, key=lambda x: x, reverse=False):
    """Merge Sort Algorithm - O(n log n) stable"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key, reverse)
    right = merge_sort(arr[mid:], key, reverse)
    
    return merge(left, right, key, reverse)

def merge(left, right, key, reverse):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        left_val = key(left[i])
        right_val = key(right[j])
        
        if (left_val <= right_val) != reverse:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ==================== DATA PREPROCESSING ====================

@st.cache_data
def load_data():
    """Load and preprocess movie data"""
    df = pd.read_csv('data/tmdb_movies.csv')
    
    df['Release_Date'] = pd.to_datetime(df['Release_Date'],
                                        errors='coerce')
    df['Year'] = df['Release_Date'].dt.year.astype('Int64')
    df['Genres'] = df['Genres'].fillna('Unknown')
    df['Rating_average'] = pd.to_numeric(df['Rating_average'],
                                          errors='coerce')
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
    df['Popularity'] = pd.to_numeric(df['Popularity'],
                                      errors='coerce')
    
    df = df.dropna(subset=['Year', 'Title'])
    df = df[df['Year'] >= 1900]
    df = df[df['Year'] <= 2025]
    
    return df

# ==================== HELPER FUNCTIONS ====================

def map_genre(genre):
    """Map genre to single word"""
    return GENRE_MAPPING.get(genre.strip(), genre.strip())

def map_language(lang_code):
    """Map language code to full name"""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)

def create_genre_list(df):
    """Extract all unique genres from dataset"""
    all_genres = set()
    for genres_str in df['Genres'].dropna():
        separators = ['|', ',']
        genres = str(genres_str)
        for sep in separators:
            if sep in genres:
                genre_parts = [map_genre(g.strip()) for g in
                               genres.split(sep)]
                all_genres.update(genre_parts)
                break
        else:
            all_genres.add(map_genre(genres.strip()))
    
    all_genres = {g for g in all_genres if g and g != 'Unknown'}
    return sorted(list(all_genres))

def filter_by_genre(df, selected_genres):
    """Filter dataframe by multiple genres (AND logic)"""
    if not selected_genres or len(selected_genres) == 0:
        return df
    
    def has_all_genres(genres_str):
        """Check if movie has ALL selected genres"""
        movie_genres = []
        for sep in ['|', ',']:
            if sep in str(genres_str):
                movie_genres = [
                    map_genre(g.strip()) for g in str(genres_str).split(sep)
                ]
                break
        else:
            movie_genres = [map_genre(str(genres_str).strip())]
        
        return all(genre in movie_genres for genre in selected_genres)
    
    mask = df['Genres'].apply(has_all_genres)
    return df[mask]

def create_recommendation_system(df, selected_genres, top_n=10):
    """Hash-based recommendation system with AND logic for multiple genres"""
    if not selected_genres or len(selected_genres) == 0:
        return df.nlargest(top_n, 'Rating_average')
    
    def has_all_genres(genres_str):
        """Check if movie has ALL selected genres"""
        movie_genres = []
        for sep in ['|', ',']:
            if sep in str(genres_str):
                movie_genres = [
                    map_genre(g.strip()) for g in str(genres_str).split(sep)
                ]
                break
        else:
            movie_genres = [map_genre(str(genres_str).strip())]
        
        return all(genre in movie_genres for genre in selected_genres)
    
    genre_movies = df[df['Genres'].apply(has_all_genres)]
    
    if len(genre_movies) > 0:
        return genre_movies.nlargest(top_n, 'Rating_average')
    
    return pd.DataFrame()

def get_similar_movies(df, movie, top_n=5):
    """Get similar movies based on genre matching"""
    movie_genres = []
    genres_str = str(movie['Genres'])
    for sep in ['|', ',']:
        if sep in genres_str:
            movie_genres = [map_genre(g.strip()) for g in
                            genres_str.split(sep)]
            break
    else:
        movie_genres = [map_genre(genres_str.strip())]
    
    similar_movies = df[
        df['Genres'].apply(
            lambda x: any(
                genre in [
                    map_genre(g.strip())
                    for sep in ['|', ',']
                    for g in str(x).split(sep)
                ]
                for genre in movie_genres
            )
        )
    ]
    similar_movies = similar_movies[
        similar_movies['Title'] != movie['Title']
    ]
    
    similar_list = similar_movies.to_dict('records')
    sorted_similar = quick_sort(
        similar_list,
        key=lambda x: x.get('Rating_average', 0),
        reverse=True
    )
    
    return sorted_similar[:top_n]

# ==================== MAIN APPLICATION ====================
def main():
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'tmdb_movies.csv' is in the data directory")
        return
    
    st.markdown(get_theme_css(),
                unsafe_allow_html=True)
    
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("./images/logo.png", use_container_width=True)
    except:  # noqa: E722
        pass
    
    # ==================== SIDEBAR FILTERS ====================
    st.sidebar.title("Narrow things down")
    
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    all_genres = create_genre_list(df)
    selected_genres = st.sidebar.multiselect(
        "Select Genre(s)",
        options=all_genres,
        default=[],
        help="Select multiple genres. Movies must contain ALL selected genres."
    )
    
    unique_languages = df['Original_Language'].dropna().unique()
    mapped_languages = sorted(
        [(lang, map_language(lang)) for lang in unique_languages],
        key=lambda x: x[1]
    )
    language_options = ['All'] + [
        f"{name} ({code})" for code, name in mapped_languages[:10]
    ]
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=language_options
    )
    
    filtered_df = df[
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ]
    
    if selected_language != 'All':
        lang_code = selected_language.split('(')[-1].strip(')')
        filtered_df = filtered_df[
            filtered_df['Original_Language'] == lang_code
        ]
    
    filtered_df = filter_by_genre(filtered_df, selected_genres)
    
    # ==================== PAGE NAVIGATION ====================
    page = st.sidebar.radio(
        "Navigate to:",
        ["Overview Analytics", "Search & Discovery"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Total Movies**: {len(filtered_df):,}")
    
    # ==================== PAGE 1: OVERVIEW ANALYTICS ====================
    if page == "Overview Analytics":
        st.header("At a glance.")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(filtered_df):,}</div>
                <div class="kpi-label">Total Films</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_rating = filtered_df['Rating_average'].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{avg_rating:.2f}</div>
                <div class="kpi-label">Avg Rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_genres = len(create_genre_list(filtered_df))
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{unique_genres}</div>
                <div class="kpi-label">Total Genres</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_duration = filtered_df['Runtime'].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{avg_duration:.0f}m</div>
                <div class="kpi-label">Avg Duration</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("A look at the leading genres")
            
            genre_counts = {}
            for genres_str in filtered_df['Genres'].dropna():
                for sep in ['|', ',']:
                    if sep in str(genres_str):
                        for genre in str(genres_str).split(sep):
                            mapped_genre = map_genre(genre)
                            genre_counts[mapped_genre] = (
                                genre_counts.get(mapped_genre, 0) + 1
                            )
                        break
                else:
                    mapped_genre = map_genre(str(genres_str))
                    genre_counts[mapped_genre] = (
                        genre_counts.get(mapped_genre, 0) + 1
                    )
            
            genre_list = [(genre, count) for genre, count in
                          genre_counts.items()]
            sorted_genres = quick_sort(
                genre_list,
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            genre_df = pd.DataFrame(sorted_genres,
                                    columns=['Genre', 'Count'])
            genre_df = genre_df.sort_values('Count', ascending=True)
            
            fig_bar = px.bar(
                genre_df,
                x='Count',
                y='Genre',
                orientation='h',
                color='Count',
                color_continuous_scale=[
                    [0, '#e5ada8'],
                    [0.5, '#e5d7c4'],
                    [0.75, '#889063'],
                    [1, '#354024']
                ]
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("The spread of languages")
            
            lang_counts = filtered_df['Original_Language'].value_counts().head(10)
            mapped_lang_counts = {
                map_language(lang): count for lang, count in
                lang_counts.items()
            }
            
            sorted_langs = sorted(
                mapped_lang_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_4 = sorted_langs[:4]
            others_count = sum([count for _, count in sorted_langs[4:]])
            
            if others_count > 0:
                top_4.append(('Others', others_count))
            
            pie_data = dict(top_4)
            
            fig_pie = px.pie(
                values=list(pie_data.values()),
                names=list(pie_data.keys()),
                color_discrete_sequence=[
                    '#354024',
                    '#a1a37e',
                    '#e5d2c1',
                    '#e5bbb1',
                    '#e5aea9'
                ]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Line Chart
        st.subheader("Tracing the years of movie releases")
        
        trend_df = filtered_df[filtered_df['Year'] <= 2020]
        
        yearly_data = trend_df.groupby('Year').agg({
            'Title': 'count',
            'Rating_average': 'mean'
        }).reset_index()
        yearly_data.columns = ['Year', 'Count', 'Avg_Rating']
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Count'],
            mode='lines+markers',
            name='Number of Movies',
            line=dict(color='#354024', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(207, 187, 153, 0.2)'
        ))
        
        fig_line.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Avg_Rating'],
            mode='lines+markers',
            name='Average Rating',
            line=dict(color='#e5aea9', width=3),
            marker=dict(size=6),
            yaxis='y2'
        ))
        
        fig_line.update_layout(
            xaxis=dict(
                title='Year',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='Number of Movies',
                side='left',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis2=dict(
                title='Average Rating',
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, 10]
            ),
            hovermode='x unified',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # ==================== PAGE 2: GRANULAR SEARCH ====================
    elif page == "Search & Discovery":
        st.header("The Art of Discovery")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Movies",
                placeholder="Enter movie title...",
                label_visibility="visible"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button(
                "Search", type="primary", use_container_width=True
            )
        
        st.markdown("### Advanced Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rating_range = st.slider(
                "Rating Range",
                min_value=0.0,
                max_value=10.0,
                value=(0.0, 10.0),
                step=0.1
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                [
                    'Rating (High to Low)',
                    'Rating (Low to High)',
                    'Popularity (High to Low)',
                    'Year (Newest First)',
                    'Duration (Longest First)'
                ]
            )
        
        with col3:
            results_limit = st.number_input(
                "Show Results",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
        
        search_df = filtered_df[
            (filtered_df['Rating_average'] >= rating_range[0]) &
            (filtered_df['Rating_average'] <= rating_range[1])
        ]
        
        if search_query:
            st.markdown(
                f"### Search Results for: **{search_query}**"
            )
            
            movies_list = search_df.to_dict('records')
            search_results = linear_search(
                movies_list,
                search_query,
                key=lambda x: x.get('Title', '')
            )
            
            st.info(
                f"**Algorithm Used**: Linear Search | "
                f"**Complexity**: O(n) | "
                f"**Results Found**: {len(search_results)}"
            )
            
            if search_results:
                if sort_by == 'Rating (High to Low)':
                    search_results = merge_sort(
                        search_results,
                        key=lambda x: x.get('Rating_average', 0),
                        reverse=True
                    )
                elif sort_by == 'Rating (Low to High)':
                    search_results = merge_sort(
                        search_results,
                        key=lambda x: x.get('Rating_average', 0),
                        reverse=False
                    )
                elif sort_by == 'Popularity (High to Low)':
                    search_results = merge_sort(
                        search_results,
                        key=lambda x: x.get('Popularity', 0),
                        reverse=True
                    )
                elif sort_by == 'Year (Newest First)':
                    search_results = merge_sort(
                        search_results,
                        key=lambda x: x.get('Year', 0),
                        reverse=True
                    )
                elif sort_by == 'Duration (Longest First)':
                    search_results = merge_sort(
                        search_results,
                        key=lambda x: x.get('Runtime', 0),
                        reverse=True
                    )
                
                for i, movie in enumerate(
                    search_results[:results_limit]
                ):
                    with st.expander(
                        f"{i+1}. {movie['Title']} "
                        f"({movie.get('Year', 'N/A')})"
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Original Title**: {movie.get(
                                'Original_Title', 'N/A'
                            )}  
                            **Genres**: {movie.get('Genres', 'N/A')}  
                            **Overview**: {movie.get(
                                'Overview', 'No overview available'
                            )[:200]}...
                            """)
                        
                        with col2:
                            st.metric(
                                "Rating",
                                f"{movie.get('Rating_average', 0):.1f}/10"
                            )
                            st.metric(
                                "Popularity",
                                f"{movie.get('Popularity', 0):.0f}"
                            )
                            st.metric(
                                "Duration",
                                f"{movie.get('Runtime', 0):.0f} min"
                            )
                        
                        st.markdown("---")
                        st.markdown("**Similar Movies**")
                        
                        similar_movies = get_similar_movies(
                            search_df, movie, top_n=5
                        )
                        
                        if similar_movies:
                            for j, sim_movie in enumerate(
                                similar_movies
                            ):
                                st.markdown(f"""
                                {j+1}. **{sim_movie['Title']}** 
                                ({sim_movie.get('Year', 'N/A')}) - 
                                Rating: {sim_movie.get(
                                    'Rating_average', 0
                                ):.1f}/10
                                """)
                        else:
                            st.info("No similar movies found")
            else:
                st.warning("No movies found matching your search.")
        
        st.markdown("---")
        st.markdown("### Personalized Recommendations")
        
        if selected_genres and len(selected_genres) > 0:
            genres_text = ", ".join(selected_genres)
            
            recommendations = create_recommendation_system(
                search_df,
                selected_genres,
                top_n=10
            )
            
            if len(recommendations) > 0:
                st.success(
                    f"Found {len(recommendations)} recommendations "
                    f"containing ALL genres: **{genres_text}**"
                )
                
                cols = st.columns(2)
                for idx, (_, movie) in enumerate(
                    recommendations.iterrows()
                ):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{movie['Title']}</h4>
                            <p><strong>Year:</strong> {movie.get(
                                'Year', 'N/A'
                            )}</p>
                            <p class="rating"><strong>Rating:</strong> {movie.get(
                                'Rating_average', 0
                            ):.1f}/10</p>
                            <p><strong>Genres:</strong> {movie.get(
                                'Genres', 'N/A'
                            )}</p>
                            <p><strong>Overview:</strong> {str(
                                movie.get('Overview', 'N/A')
                            )[:150]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(
                    f"No movies found containing ALL genres: {genres_text}"
                )
        else:
            st.info(
                "Select one or more genres from the sidebar to get "
                "personalized recommendations!"
            )

    st.markdown("---")

if __name__ == "__main__":
    main()