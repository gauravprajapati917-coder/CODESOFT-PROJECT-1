import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Dataset create karna (Dummy Data for Movies)
def load_data():
    data = {
        'Movie_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Title': [
            'The Dark Knight', 'Inception', 'Interstellar', 
            'The Hangover', 'Superbad', 'Toy Story', 
            'Finding Nemo', 'The Conjuring', 'Insidious', 'The Matrix'
        ],
        'Genre': [
            'Action Sci-Fi Thriller', 'Action Sci-Fi Thriller', 'Sci-Fi Drama Adventure',
            'Comedy', 'Comedy', 'Animation Comedy Family',
            'Animation Adventure Family', 'Horror Mystery Thriller', 'Horror Mystery', 'Action Sci-Fi'
        ]
    }
    return pd.DataFrame(data)

# 2. Recommendation Engine Function
def get_recommendations(movie_title, df, top_n=3):
    # Movie titles ko lowercase me convert karna taaki case-insensitive search ho sake
    df['Title_Lower'] = df['Title'].str.lower()
    movie_title_lower = movie_title.lower()
    
    # Check karna ki movie dataset me hai ya nahi
    if movie_title_lower not in df['Title_Lower'].values:
        return f"Sorry! '{movie_title}' hamare database me nahi hai. Kripya koi dusri movie try karein."
    
    # TF-IDF Vectorizer se genres ko numbers (vectors) me badalna
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Genre'])
    
    # Cosine Similarity calculate karna (Movies ke beech ka similarity score)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Selected movie ka index nikalna
    idx = df[df['Title_Lower'] == movie_title_lower].index[0]
    
    # Us movie ke similarity scores ki list nikalna aur sort karna
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Top matches nikalna (khud us movie ko chhodkar jo search ki gayi hai)
    sim_scores = [score for score in sim_scores if score[0] != idx]
    top_movies_indices = [score[0] for score in sim_scores[:top_n]]
    
    # Recommended movies ka DataFrame return karna
    return df[['Title', 'Genre']].iloc[top_movies_indices]

# 3. Main Program UI
if __name__ == "__main__":
    df = load_data()
    
    print("="*50)
    print("🎬 WELCOME TO CODEWAY / CODSOFT MOVIE RECOMMENDATION SYSTEM 🎬")
    print("="*50)
    print("\nHamare Database me yeh Movies available hain:")
    for movie in df['Title']:
        print(f" - {movie}")
    print("-" * 50)
    
    # User Input loop
    while True:
        user_choice = input("\nKisi ek movie ka naam likhein (ya 'exit' type karke band karein): ").strip()
        
        if user_choice.lower() == 'exit':
            print("\nThank you for using the Recommendation System! Happy Watching! 🍿")
            break
            
        recommendations = get_recommendations(user_choice, df)
        
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print(f"\n🎯 Agar aapko '{user_choice}' pasand hai, toh aapko yeh bhi dekhna chahiye:")
            print(recommendations.to_string(index=False))
        print("-" * 50)