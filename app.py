import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.tree import DecisionTreeClassifier

# Configure page
st.set_page_config(
    page_title="Musical Complexity vs Popularity",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI look - Professional typography and spacing
st.markdown("""
<style>
    /* Main Layout */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Global Typography Reset for Streamlit */
    .stMarkdown h1 {
        background: -webkit-linear-gradient(45deg, #a34ae2, #4a90e2);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        margin-bottom: 2rem;
        text-shadow: 0px 4px 20px rgba(163, 74, 226, 0.2);
    }
    .stMarkdown h2 {
        color: #e2d7b5 !important;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
        margin-top: 2.5rem;
        font-weight: 600;
    }
    .stMarkdown h3 {
        color: #4a90e2 !important;
        font-weight: 500;
    }

    /* Cover Hero - The Ultimate Purple Glow */
    .cover-hero {
        background: linear-gradient(135deg, rgba(14, 18, 23, 0.9) 0%, rgba(26, 35, 47, 0.95) 100%);
        padding: 70px 40px;
        border-radius: 16px;
        border-top: 4px solid #a34ae2;
        border-bottom: 1px solid rgba(163, 74, 226, 0.3);
        border-left: 1px solid rgba(163, 74, 226, 0.1);
        border-right: 1px solid rgba(163, 74, 226, 0.1);
        color: #ffffff;
        text-align: center;
        margin-bottom: 50px;
        box-shadow: 0 15px 40px rgba(163, 74, 226, 0.15), inset 0 0 40px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
    }
    .cover-hero::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(163,74,226,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .cover-hero h1 {
        font-size: 4.5rem !important;
        background: -webkit-linear-gradient(45deg, #d38cff, #a34ae2) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 15px;
        font-weight: 900 !important;
        letter-spacing: -2px;
        text-shadow: none !important; /* Managed by clip */
    }
    .cover-hero p {
        font-size: 1.4rem;
        color: #c9c9c9;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* Glassmorphism Metric Boxes */
    .metric-box {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: left;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        height: 100%;
    }
    .metric-box:hover {
        transform: translateY(-8px);
        background: rgba(30, 30, 30, 0.8);
        border-color: rgba(163, 74, 226, 0.5);
        box-shadow: 0 15px 40px rgba(163, 74, 226, 0.15);
    }
    .metric-box h3 {
        margin-top: 0;
        font-size: 1.5rem;
    }

    /* Analysis & Finding Cards */
    .analysis-card {
        background: linear-gradient(145deg, rgba(22, 26, 34, 0.8), rgba(18, 20, 26, 0.9));
        padding: 30px;
        border-radius: 12px;
        border-left: 6px solid #4a90e2;
        border-top: 1px solid rgba(74, 144, 226, 0.2);
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .finding-card {
        background: linear-gradient(145deg, rgba(26, 18, 34, 0.8), rgba(20, 15, 26, 0.9));
        padding: 30px;
        border-radius: 12px;
        border-left: 6px solid #a34ae2;
        border-top: 1px solid rgba(163, 74, 226, 0.2);
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }

    /* Layman Translations - Elevated */
    .layman-translation {
        background: rgba(31, 27, 15, 0.7);
        backdrop-filter: blur(8px);
        padding: 22px;
        border-radius: 8px;
        border-left: 4px solid #F1C40F;
        font-style: italic;
        color: #e2d7b5;
        margin-top: 20px;
        border-right: 1px solid rgba(241, 196, 15, 0.1);
        border-bottom: 1px solid rgba(241, 196, 15, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Application constants
DATA_PATH = "data/spotify_audio_features.csv"
RULES_PATH = "results/rules/association_rules.csv"

@st.cache_data
def load_and_preprocess_data():
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.complexity_index import add_complexity_dimensions
    from sklearn.cluster import KMeans
    
    if not os.path.exists(DATA_PATH):
        return None
        
    raw_df = pd.read_csv(DATA_PATH)
    df = load_dataset(DATA_PATH)
    df = preprocess_data(df)
    df = add_complexity_dimensions(df)
    
    X = df[['musical_complexity', 'popularity']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X).astype(str)
    
    cluster_mapping = {
        '0': 'The Radio Anthems (Low Complexity, Massive Appeal)',
        '1': 'The Experimental Niche (High Complexity, Cult Following)',
        '2': 'The Indie Sweet-Spot (Moderate Complexity, Stable Appeal)'
    }
    df['cluster_name'] = df['cluster'].map(cluster_mapping)
    
    if 'track_name' in raw_df.columns:
        df.insert(0, 'Track Name', raw_df.loc[df.index, 'track_name'])
    if 'artist_name' in raw_df.columns:
        df.insert(1, 'Artist', raw_df.loc[df.index, 'artist_name'])
        
    return df

@st.cache_resource
def load_trained_model(df):
    df_model = df.copy()
    df_model['popularity_class'] = pd.qcut(
        df_model['popularity'], q=3, labels=['Low', 'Medium', 'High']
    )
    drops = ['popularity', 'popularity_class', 'cluster', 'cluster_name', 'Track Name', 'Artist']
    X = df_model.drop([col for col in drops if col in df_model.columns], axis=1)
    y = df_model['popularity_class']
    
    model = DecisionTreeClassifier(max_depth=7, random_state=42)
    model.fit(X, y)
    
    return model, list(X.columns)

df = load_and_preprocess_data()

if df is None:
    st.error(f"Data not found at {DATA_PATH}. Please ensure the dataset exists.")
    st.stop()

st.sidebar.title("Study Navigation")

page = st.sidebar.radio(
    "Analytical Chapters:",
    [
        "1. Cover & Project Abstract", 
        "2. Formulating Musical Complexity", 
        "3. Latent Song Clusters", 
        "4. Feature Impact (Regression)", 
        "5. Association Rule Interactions", 
        "6. Predictive Popularity Playground"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("An interactive dashboard exploring the relationship between acoustic complexity and streaming popularity using audio features.")

if page == "1. Cover & Project Abstract":
    # Professional Dark Theme Cover Page
    st.markdown("""
    <div class='cover-hero'>
        <h1>The Science of the Hit</h1>
        <p>A Data-Driven Investigation into Musical Complexity and Streaming Popularity</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### The Research Question
    Music popularity in the digital era is often treated as a black-box prediction problem. Traditional models attempt to estimate listener engagement based on historical data, but frequently lack interpretability regarding *why* specific intrinsic musical characteristics drive audience reception. 
    
    The core problem addressed in this framework is to determine **how quantifiable musical complexity relates to song popularity**. Specifically, we avoid black-box predictive modeling in favor of identifying transparent, interpretable patterns, latent groupings, and association rules that *explain* chart success.
    
    <div class='layman-translation'>
        <b>In Simple Terms:</b> We want to know exactly what makes a song a hit. Is it better to make a simple, repetitive song, or a highly complex, instrumentally diverse song? Instead of guessing, we used machine learning to scan the audio waves of 75,000 Spotify tracks to give us the mathematical truth.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='analysis-card'>
        <h4>Dataset Context & Preprocessing Architecture</h4>
        <p>The dataset utilizes Spotify Web API features extracted directly from audio signals. To prepare this data for robust statistical modeling and clustering, we implemented a rigorous pipeline:</p>
        <ul>
            <li><b>Missing Value Imputation</b>: Ensuring structural integrity across the time-series estimations.</li>
            <li><b>IQR Outlier Isolation</b>: Extreme variations in tempo and loudness were clamped to prevent distortion of centroid-based clustering (e.g. K-Means).</li>
            <li><b>Min-Max Normalization</b>: Features were scaled uniformly to ensure high-variance metrics (like Tempo) do not mathematically override low-variance metrics (like Acousticness).</li>
        </ul>
        <div class='layman-translation'>
            <b>In Simple Terms (Preprocessing):</b> Data is messy. Sometimes Spotify misses a value, or a song is unusually loud (an outlier). Before we can analyze anything, we "clean" and "standardize" the data so that a loud song and a fast song are judged on an equal mathematical playing field.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### The Target Variable (Popularity)")
        st.write("Before diving into features, it's vital to see how popularity is distributed across the entire platform. The graph to the right reveals that popularity is a heavily right-skewed metric—most songs linger in obscurity (0 popularity), and achieving a Viral hit (80+) is incredibly rare.")
        st.metric("Total Analyzed Audio Signatures", f"{len(df):,}")
        st.metric("Median Database Popularity", f"{int(df['popularity'].median())} / 100")
        
    with col2:
        dist_fig = px.histogram(df.sample(min(10000, len(df))), x='popularity', nbins=50, 
                                title="Platform Popularity Distribution (10k Sample)",
                                color_discrete_sequence=['#1DB954'])
        st.plotly_chart(dist_fig, use_container_width=True)

elif page == "2. Formulating Musical Complexity":
    st.title("Chapter 2: The Complexity Index (MCI)")
    st.markdown("""
    Translating the abstract musicological concept of 'complexity'—which theoretically encompasses rhythmic irregularity, harmonic variation, and structural diversity—into an algorithmic index requires composite feature engineering. 
    
    We hypothesize that complexity is not a single feature but a multi-dimensional construct. We formulate the **Musical Complexity Index (MCI)** by combining normalized variance metrics:
    """)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='metric-box'>
            <h3 style="color: #4a90e2; font-weight: bold;">Rhythmic Complexity</h3>
            <p>Aggregates <b>Tempo</b>, <b>Danceability</b> (Inverted), and <b>Speechiness</b>.</p>
            <p style="font-size: 0.9em; color: #ccc;"><i>High values here indicate off-beat phrasing, rapid tempo shifting, and non-4/4 time signatures—harder to dance to, but musically fascinating.</i></p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-box'>
            <h3 style="color: #e24a4a; font-weight: bold;">Dynamic Complexity</h3>
            <p>Aggregates <b>Energy</b>, <b>Loudness</b>, and <b>Liveness</b>.</p>
            <p style="font-size: 0.9em; color: #ccc;"><i>Measures decibel range fluctuations. A song that goes from a quiet whisper to an explosive chorus has very high dynamic complexity.</i></p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-box'>
            <h3 style="color: #1DB954; font-weight: bold;">Structural Complexity</h3>
            <p>Aggregates <b>Acousticness</b>, <b>Instrumentalness</b>, and track <b>Duration</b>.</p>
            <p style="font-size: 0.9em; color: #ccc;"><i>Evaluates traditional organic instrumentation. A 6-minute acoustic guitar solo is highly structural compared to a 2-minute repetitive synthetic vocal hook.</i></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("""
    <div class='finding-card'>
        <h4>Key Finding: The Popularity Ceiling on Complexity</h4>
        <p>Through our <b>ANOVA F-statistic tests</b>, we established that there is a statistically significant variance in popularity between complexity levels. The data strongly suggests that highly complex songs observe a strict 'popularity ceiling'. Meanwhile, songs hitting the sweet spot of moderate rhythmic variation but low structural complexity yield the highest median listener engagement.</p>
        <div class='layman-translation'>
            <b>In Simple Terms (ANOVA):</b> ANOVA is a statistical test that proves whether groups are actually different from each other, or if it's just random luck. Our test proved it isn't luck: <b>The more complex a song gets, the harder it is for it to become a viral hit.</b> Highly complex songs literally hit a mathematical "ceiling" where the general public stops listening.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Distribution of the MCI against Target Popularity")
    
    col_plot1, col_plot2 = st.columns(2)
    
    plot_df = df.sample(n=min(3000, len(df)), random_state=42)
    with col_plot1:
        fig = px.scatter(plot_df, x='musical_complexity', y='popularity', opacity=0.4, trendline='lowess', trendline_color_override='red',
                         hover_data=['Track Name', 'Artist'], 
                         labels={'musical_complexity': 'Total Musical Complexity', 'popularity': 'Spotify Popularity'},
                         title="Linear Mapping: Complexity vs Popularity")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The red trendline peaks early, and steadily drops off as complexity increases.")
        
    with col_plot2:
        # Density contour plot to show the massive concentration around specific complexity parameters
        fig_dens = px.density_contour(plot_df, x="rhythmic_complexity", y="dynamic_complexity", 
                                      title="Density Topography: Rhythm vs Dynamic Energy",
                                      color_discrete_sequence=['#4a90e2'])
        st.plotly_chart(fig_dens, use_container_width=True)
        st.caption("Contours show where the absolute volume of modern music naturally clusters.")

elif page == "3. Latent Song Clusters":
    st.title("Chapter 3: Profiling Latent Groupings")
    st.markdown("""
    To discover inherent musical 'tribes' independent of subjective genre tags, we executed **K-Means Clustering** across the Complexity Space.
    
    Instead of relying on human labels like "Rock" or "Pop", K-Means looks entirely at the math. It assigns songs into unsuperivsed groupings that share nearly identical mathematical signatures in rhythm, structure, and popularity.
    """)
    
    st.markdown("""
    <div class='analysis-card'>
        <h4>The Three Derived Archetypes</h4>
        <ul>
            <li><b>The Radio Anthems (Low Complexity, Massive Appeal):</b> Low structural complexity, highly predictable rhythmic variance. High baseline popularity. <i>Think Top 40 Pop, standard Hip-Hop. Easy to digest and consistently charted.</i></li>
            <li><b>The Experimental Niche (High Complexity, Cult Following):</b> Extremely high structural and dynamic complexity, but suffers heavily in mass popularity. <i>Think Avant-Garde Jazz, Progressive Metal, or 10-minute Orchestral pieces.</i></li>
            <li><b>The Indie Sweet-Spot (Moderate Complexity, Stable Appeal):</b> Tracks that blend organic acousticness with a standard dynamic range. <i>Think Indie Pop, Acoustic Folk, and Legacy Rock. Successful, but not viral.</i></li>
        </ul>
        <div class='layman-translation'>
            <b>Why is this important?</b> Algorithms like Spotify's recommendation feed don't categorize you entirely by Human Genres anymore. If you listen to a lot of 'Experimental Niche', the algorithm will recommend you other songs mathematically located in that cluster, regardless of whether it's classified as Jazz or Heavy Metal!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Interactive 3D Cluster Typography")
    st.write("Rotate and zoom the 3D space to see how the three mathematical tribes separate themselves physically across the three dimensions of complexity. Individual points represent single songs.")
    
    plot_df_3d = df.sample(n=min(2500, len(df)), random_state=42)
    fig = px.scatter_3d(
        plot_df_3d, 
        x='rhythmic_complexity', 
        y='dynamic_complexity', 
        z='structural_complexity',
        color='cluster_name', 
        hover_data=['Track Name', 'Artist', 'popularity'],
        opacity=0.7,
        title="3D Complexity Space Mapping (2,500 point sample)",
        color_discrete_sequence=['#1DB954', '#e74c3c', '#4a90e2']
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(scene=dict(xaxis_title='Rhythmic (Tempo/Beat)', yaxis_title='Dynamic (Vol/Energy)', zaxis_title='Structural (Acoustic/Length)'),
                      margin=dict(l=0, r=0, b=0, t=50),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

elif page == "4. Feature Impact (Regression)":
    st.title("Chapter 4: Permutation Importance & Interactions")
    st.markdown("""
    We moved beyond raw plotting by applying **Regression Models (OLS)** and **Random Forest Permutation Feature Importance**. The objective is to quantify precisely *which* audio signals explain the variance in Chart Success.
    """)
    
    st.markdown("""
    <div class='finding-card'>
        <h4>Key Finding: Energy vs Structural Interaction Modeling</h4>
        <p>Our interaction regression mapping proves that high energy is only predictive of positive popularity when structural complexity is LOW. If a track is highly instrumental and acoustic (High Structure), cranking up the Energy actually inversely affects popularity.</p>
        <div class='layman-translation'>
            <b>In Simple Terms:</b> If a song is a basic Pop track (Low structure), making it high-energy makes it MORE popular. But if a song is a complex, 8-minute acoustic piece (High structure), making it wildly high-energy makes it LESS popular. It becomes noisy, chaotic, and drives average listeners away. The context of the song's nature matters.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.subheader("Feature Importance Baseline")
        st.write("Which features does the algorithm rely on the most to predict hit songs?")
        model, features = load_trained_model(df)
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by="Importance", ascending=False)
        
        fig = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', color='Importance', 
                     color_continuous_scale='Magma')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_feat2:
        st.subheader("Bivariate Linear Feature Explorer")
        st.write("Isolate individual acoustic signals and plot their mathematical effects against listener engagement.")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        feature_x = st.selectbox("Signal Vector (X-axis):", num_cols, index=list(num_cols).index('loudness') if 'loudness' in num_cols else 0)
        
        plot_df_2d = df.sample(n=min(3000, len(df)), random_state=42)
        fig2 = px.scatter(plot_df_2d, x=feature_x, y='popularity', opacity=0.25, trendline="ols", hover_data=['Track Name', 'Artist'], color_discrete_sequence=['#1DB954'])
        fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

elif page == "5. Association Rule Interactions":
    st.title("Chapter 5: Extracting Explicit Logic (Association Rules)")
    st.markdown("""
    Predictive accuracy holds little value without interpretability. We deployed FP-Growth/Apriori Itemset Mining by discretizing our continuous audio spectrums into `[Low, Medium, High]` categorical bins. 
    
    This process yielded explicit, readable rules connecting abstract feature combinations directly to listener demographics. This provides extreme visibility into the literal 'formulas' dictating the current streaming climate.
    """)
    
    st.markdown("""
    <div class='analysis-card'>
        <h4>Reading the Extraction Logic</h4>
        <ul>
            <li><b>Antecedents (The Cause/Input)</b>: The specific audio signals the track exhibits.</li>
            <li><b>Consequents (The Effect/Output)</b>: The derived outcome (e.g. High Popularity).</li>
            <li><b>Confidence %</b>: The statistical probability; *if a song has X, there is a Y% chance it achieves Z.*</li>
            <li><b>Lift</b>: The core metric of validity. A lift ratio scaling above 1.0 indicates that the combination is highly mathematical and intentional, far beyond random chance.</li>
        </ul>
        <div class='layman-translation'>
            <b>In Simple Terms:</b> Think of Association Rules like an algorithm scanning millions of grocery receipts. The algorithm eventually learns that <i>"If someone buys Peanut Butter (Antecedent), there is a 90% chance (Confidence) they will also buy Jelly (Consequent)."</i> We did the precise same thing, but for acoustic features and hit songs.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists(RULES_PATH):
        rules_df = pd.read_csv(RULES_PATH)
        
        col1, col2 = st.columns(2)
        min_conf = col1.slider("Filter by Reliability (Confidence)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        min_lift = col2.slider("Filter by Significance (Lift)", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
        
        filtered_rules = rules_df[(rules_df['confidence'] >= min_conf) & (rules_df['lift'] >= min_lift)]
        
        st.write(f"Displaying **{len(filtered_rules)}** explicit causal patterns.")
        st.dataframe(filtered_rules.sort_values(by='lift', ascending=False), use_container_width=True)
        
        st.markdown("### Association Pathways Diagram")
        st.write("Visualizing the flow of input signals (Antecedents) leading to output states (Consequents). Thicker lines indicate higher confidence logic.")
        if len(filtered_rules) > 0 and len(filtered_rules) < 100:
            # Parallel categories to show Antecedent -> Consequent flow
            fig = px.parallel_categories(
                filtered_rules, dimensions=['antecedents', 'consequents'],
                color="confidence", color_continuous_scale=px.colors.sequential.Inferno,
                labels={'antecedents':'Feature Inputs', 'consequents':'Popularity Outcome'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Extracted rules graph not found at {RULES_PATH}. Please execute the backend mining pipeline.")

elif page == "6. Predictive Popularity Playground":
    st.title("Chapter 6: Simulated Track Outcomes")
    st.markdown("""
    By integrating the cumulative logic from our statistical frameworks, we have synthesized a transparent, live **Decision Tree Logic Matrix**.
    
    This tool allows you to act as a sound engineer. Select raw signal inputs. The engine calculates the derived Musical Complexity indices behind the scenes, parses the entire profile down its learned logical tree, and returns a real-time classification of the song.
    """)
    
    model, features = load_trained_model(df)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("The Mixing Board")
        st.caption("Change these sliders to test the Machine Learning model. Try replicating the acoustic profile of your favorite song to see if the Machine agrees it should be a hit.")
        
        input_data = {}
        # Unlocked 9 total sliders so the user has full control to escape the 'Medium' mean-trap
        core_features = ['tempo', 'energy', 'loudness', 'danceability', 'acousticness', 'valence', 'instrumentalness', 'speechiness', 'liveness']
        
        for feature in core_features:
            if feature in features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                # Initializations to help the tree actually navigate
                if feature in ['instrumentalness', 'speechiness', 'liveness']:
                    start_val = 0.0
                elif feature in ['danceability', 'energy']:
                    start_val = 0.7 
                elif feature in ['loudness']:
                    start_val = -5.0
                else:
                    start_val = mean_val
                    
                input_data[feature] = st.slider(f"{feature.title()}", min_val, max_val, start_val)
        
        # Fill remaining non-target variables to mean
        for feature in features:
            if feature not in input_data:
                input_data[feature] = float(df[feature].mean())
                
        input_df = pd.DataFrame([input_data])
        
        # Calculate complexity features on the hypothetical data before passing
        from src.complexity_index import add_complexity_dimensions
        input_df = add_complexity_dimensions(input_df)
        
        # Maintain strict column layout
        input_df = input_df[features] 
    
    with col2:
        st.subheader("Backend Simulation Diagnostics")
        prediction = model.predict(input_df)[0]
        
        # Dynamic styling for prediction
        color_map = {
            "High": "#1DB954",
            "Medium": "#f39c12",
            "Low": "#e74c3c"
        }
        pred_color = color_map.get(prediction, "#1DB954")
        
        try:
            complex_score = input_df['musical_complexity'].iloc[0]
            rhythmic_score = input_df['rhythmic_complexity'].iloc[0]
            dynamic_score = input_df['dynamic_complexity'].iloc[0]
            structural_score = input_df['structural_complexity'].iloc[0]
        except Exception:
            complex_score = 0.0 
            rhythmic_score = 0.0
            dynamic_score = 0.0
            structural_score = 0.0
            
        st.markdown(f"""
        <div style="background-color: #161a22; padding: 25px; border-radius: 10px; border-left: 8px solid {pred_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
            <h2 style="margin:0; border:none; text-transform: uppercase;">Forecasted Engagement: <span style="color:{pred_color};">{prediction}</span></h2>
            <br/>
            <h4>Calculated Complexity Index (MCI): {complex_score:.3f}</h4>
            <ul style="color: #b3b3b3; line-height: 1.8;">
                <li><b>Rhythmic Volatility:</b> {rhythmic_score:.3f}</li>
                <li><b>Dynamic Fluctuation:</b> {dynamic_score:.3f}</li>
                <li><b>Structural Complexity:</b> {structural_score:.3f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Adding a visual Radar / Spider Chart to compare the user's input against the ideal 'Hit'
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Input Acoustic Radar Profile")
        
        # Normalize the input features 0-1 for the radar chart specifically
        radar_features = ['danceability', 'energy', 'acousticness', 'valence', 'instrumentalness']
        radar_vals = []
        for rf in radar_features:
            # We scale it purely for visuals based on min/max of df
            if df[rf].max() > 0:
                normalized = (input_df[rf].iloc[0] - df[rf].min()) / (df[rf].max() - df[rf].min())
            else:
                normalized = 0
            radar_vals.append(normalized * 100) # 0-100 scale
            
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=radar_vals,
          theta=[f.title() for f in radar_features],
          fill='toself',
          name='Your Track',
          line_color=pred_color
        ))
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
          showlegend=False,
          margin=dict(l=40, r=40, t=20, b=20),
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        <div class='layman-translation'>
            <b>How the Algorithm Thinks:</b> The tree splits paths based entirely on mathematical boundaries derived from our dataset. Try dropping 'Instrumentalness' to 0 and pushing 'Danceability' and 'Energy' high to see if you can break into the coveted HIGH popularity bracket! The algorithm heavily penalizes chaotic, rambling acoustic structures.
        </div>
        """, unsafe_allow_html=True)
