# MovieMate
A comprehensive movie analytics and discovery application built with Streamlit that provides insights into movie trends, search functionality, and personalized recommendations.

## Team Members

**Kelompok 1**
- Ghisele Valerin Sharent Milano (5052241005)
- Kiranna Indy (5052241014)
- Elena Miska Faradisa (5052241022)
- Inessa Regina Angelica Munda (5052241037)
- Muhammad Kafanal Kafi (5052241039)

## Features

### Overview Analytics
- **Key Performance Indicators**: Total films, average rating, genre count, and average duration
- **Genre Distribution**: Visual representation of movie genres in the dataset
- **Language Distribution**: Pie chart showing the breakdown of movie languages
- **Temporal Trends**: Line chart displaying movie releases and rating trends over time

### Search & Discovery
- **Advanced Search**: Linear search implementation for finding movies by title
- **Filtering Options**: 
  - Year range slider
  - Multi-select genre filter (AND logic)
  - Language selection
  - Rating range filter
- **Sorting Options**: Sort by rating, popularity, year, or duration
- **Personalized Recommendations**: Genre-based recommendation system
- **Similar Movies**: Find movies with similar genres to your selected film

## Technical Implementation

### Algorithms Implemented
- **Quick Sort**: O(n log n) average case sorting algorithm
- **Merge Sort**: O(n log n) stable sorting algorithm
- **Binary Search**: O(log n) search algorithm for sorted data
- **Linear Search**: O(n) search algorithm for unstructured text search

### Data Processing
- Data preprocessing and cleaning
- Genre mapping and standardization
- Language code to full name mapping
- Hash-based recommendation system

## Installation & Setup

### Using Virtual Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Kelompok1_MovieMate_Code
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Deactivate the virtual environment** (when done)
   ```bash
   deactivate
   ```

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t moviemate .
   ```

2. **Run the Docker container**
   ```bash
   docker run -p 8501:8501 moviemate
   ```

## Project Structure
```
Kelompok1_MovieMate_Code/
├── app.py                 # Main application file
├── requirements.txt        # Python dependencies
├── Dockerfile            # Docker configuration
├── .gitignore           # Git ignore file
├── README.md            # This file
├── data/
│   └── tmdb_movies.csv  # Movie dataset
└── images/
    └── logo.png         # Application logo
```

## Dependencies
- streamlit>=1.33.0
- pandas>=2.2.0
- plotly>=5.18.0
- numpy>=2.1.0
- openpyxl>=3.1.3

## Data Source
The application uses the TMDB Movies dataset (`tmdb_movies.csv`) which contains information about movies including:
- Title and original title
- Overview and genres
- Cast and crew information
- Release date and status
- Language and runtime
- Popularity and ratings
- Budget and revenue

## Usage
1. **Launch the application** using one of the methods above
2. **Navigate** between "Overview Analytics" and "Search & Discovery" using the sidebar
3. **Apply filters** in the sidebar to narrow down your movie selection
4. **Explore analytics** to gain insights about movie trends
5. **Search for movies** using the search functionality
6. **Get personalized recommendations** by selecting your preferred genres

## Browser Access

After running the application, open your web browser and navigate to:
- Local URL: `http://localhost:8501`
- Network URL: `http://ip:8501`

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Make sure you've activated the virtual environment before installing dependencies
2. **Port already in use**: Streamlit will automatically try the next available port (8502, 8503, etc.)
3. **Data loading error**: Ensure `tmdb_movies.csv` is in the `data/` directory

### Performance Tips
- The application uses Streamlit's caching (`@st.cache_data`) for efficient data loading
- Large datasets may take longer to load initially
- Use the filters in the sidebar to improve search performance

## License
This project is part of a college assignment for Kelompok 1.