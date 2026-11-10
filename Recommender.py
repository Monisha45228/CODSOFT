"""
Simple Movie Recommendation System
===================================
Demonstrates two classic recommendation approaches:

1. CONTENT-BASED FILTERING
   Recommends movies similar to a given movie, based on movie
   attributes (genres). Uses TF-IDF vectorization + cosine similarity.

2. COLLABORATIVE FILTERING (User-Based)
   Recommends movies to a user based on ratings from *other users*
   with similar taste. Uses a user-item rating matrix + cosine similarity.

No external dataset needed -- sample data is included below.
Only dependency: pandas, numpy, scikit-learn
    pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. SAMPLE DATA
# ---------------------------------------------------------------------------

movies = pd.DataFrame({
    "movie_id": range(1, 11),
    "title": [
        "The Matrix", "Inception", "The Notebook", "Titanic",
        "John Wick", "Mad Max: Fury Road", "La La Land",
        "Interstellar", "Pride and Prejudice", "The Dark Knight"
    ],
    "genres": [
        "Action Sci-Fi Thriller",
        "Action Sci-Fi Thriller",
        "Romance Drama",
        "Romance Drama",
        "Action Thriller",
        "Action Sci-Fi",
        "Romance Musical Drama",
        "Sci-Fi Drama Adventure",
        "Romance Drama",
        "Action Crime Thriller"
    ]
})

# Rows = users, columns = movie_id, values = rating (0 = not rated), scale 1-5
ratings_data = {
    1: [5, 4, 0, 0, 4, 5, 0, 4, 0, 5],   # Alice - likes action/sci-fi
    2: [0, 0, 5, 4, 0, 0, 5, 0, 4, 0],   # Bob   - likes romance/drama
    3: [4, 5, 0, 0, 5, 4, 0, 5, 0, 4],   # Carol - likes action/sci-fi
    4: [0, 0, 4, 5, 0, 0, 4, 0, 5, 0],   # Dan   - likes romance/drama
    5: [3, 4, 3, 0, 4, 0, 0, 5, 0, 4],   # Eve   - mixed, mostly action/sci-fi
}
user_names = ["Alice", "Bob", "Carol", "Dan", "Eve"]
ratings = pd.DataFrame(ratings_data, index=movies["movie_id"])
ratings.columns = user_names
ratings.index.name = "movie_id"


# ---------------------------------------------------------------------------
# 2. CONTENT-BASED FILTERING
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """Recommend movies similar to a given movie using genre text."""

    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df.reset_index(drop=True)
        vectorizer = TfidfVectorizer()
        genre_matrix = vectorizer.fit_transform(self.movies["genres"])
        self.similarity = cosine_similarity(genre_matrix)

    def recommend(self, title: str, top_n: int = 3):
        if title not in self.movies["title"].values:
            raise ValueError(f"Movie '{title}' not found.")

        idx = self.movies.index[self.movies["title"] == title][0]
        scores = list(enumerate(self.similarity[idx]))
        # sort by similarity, skip the movie itself
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        return [
            (self.movies.iloc[i]["title"], round(score, 3))
            for i, score in scores
        ]


# ---------------------------------------------------------------------------
# 3. COLLABORATIVE FILTERING (User-Based)
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    """Recommend movies based on ratings from similar users."""

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings = ratings_df          # movie_id x user
        self.movies = movies_df.set_index("movie_id")
        # similarity between users, based on their rating vectors
        user_matrix = self.ratings.T.values   # user x movie
        self.user_similarity = pd.DataFrame(
            cosine_similarity(user_matrix),
            index=self.ratings.columns,
            columns=self.ratings.columns,
        )

    def recommend(self, user: str, top_n: int = 3):
        if user not in self.ratings.columns:
            raise ValueError(f"User '{user}' not found.")

        # similarity of this user to everyone else (excluding self)
        sims = self.user_similarity[user].drop(user)

        # weighted sum of other users' ratings, weighted by similarity
        weighted_scores = self.ratings.drop(columns=user).mul(sims, axis=1).sum(axis=1)
        sim_sums = sims.abs().sum()
        predicted = weighted_scores / sim_sums if sim_sums > 0 else weighted_scores

        # only recommend movies the target user hasn't rated yet
        unrated = self.ratings[user] == 0
        predicted = predicted[unrated].sort_values(ascending=False).head(top_n)

        return [
            (self.movies.loc[movie_id, "title"], round(score, 3))
            for movie_id, score in predicted.items()
        ]


# ---------------------------------------------------------------------------
# 4. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CONTENT-BASED FILTERING")
    print("=" * 60)
    cb = ContentBasedRecommender(movies)
    for movie in ["The Matrix", "Titanic"]:
        print(f"\nBecause you watched '{movie}', you might like:")
        for title, score in cb.recommend(movie, top_n=3):
            print(f"  - {title} (similarity: {score})")

    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING (User-Based)")
    print("=" * 60)
    cf = CollaborativeRecommender(ratings, movies)
    for user in ["Bob", "Eve"]:
        print(f"\nRecommended for {user}:")
        for title, score in cf.recommend(user, top_n=3):
            print(f"  - {title} (predicted score: {score})")
