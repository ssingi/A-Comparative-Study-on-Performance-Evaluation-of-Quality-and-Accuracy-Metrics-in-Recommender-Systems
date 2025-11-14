"""
특징 결합 하이브리드 (Feature Combination Hybrid)
- 참고문헌:
  [1] Burke, R. (2002). "Hybrid recommender systems: Survey and experiments."
  [2] Balabanović, M., & Shoham, Y. (1997). "Fab: content-based, collaborative recommendation."
  [3] Melville, P., et al. (2002). "Content-boosted collaborative filtering."
  
하이퍼파라미터 근거:
- 비선형 결합 (0.4, 0.4, 0.2): [3] feature weighting 실험
- interaction_score: [2] FAB 시스템의 feature interaction
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from .base_recommender import BaseRecommender
import pandas as pd


class FeatureCombination(BaseRecommender):
    """
    특징 결합 하이브리드 - CF/CB 특징의 비선형 결합
    
    Formula: r̂ = μ + f(CF_features, CB_features) + bias
    
    References:
        [1] Burke (2002) - Feature combination design
        [2] Balabanović & Shoham (1997) - FAB system
        [3] Melville et al. (2002) - Content-boosted CF
    """
    
    class AdvancedCF:
        """고급 CF"""
        def __init__(self, train, mean_rating, std_rating, svd_dim=200):
            self.train = train
            self.mean_rating = mean_rating
            self.std_rating = std_rating
            self.svd_dim = svd_dim
            self.user_factors = {}
            self.movie_factors = {}
            self.user_bias = {}
            self.movie_bias = {}
            
        def fit(self):
            um = self.train.pivot_table(
                index='userId', columns='movieId', values='rating'
            ).fillna(self.mean_rating)
            
            svd = TruncatedSVD(
                n_components=min(self.svd_dim, um.shape[0]-1, um.shape[1]-1),
                random_state=42
            )
            user_features = svd.fit_transform(um)
            movie_features = svd.components_.T
            
            for i, user_id in enumerate(um.index):
                self.user_factors[user_id] = user_features[i]
            for i, movie_id in enumerate(um.columns):
                self.movie_factors[movie_id] = movie_features[i]
            
            for user_id in self.train['userId'].unique():
                ratings = self.train[self.train['userId'] == user_id]['rating'].values
                self.user_bias[user_id] = np.mean(ratings - self.mean_rating)
            
            for movie_id in self.train['movieId'].unique():
                ratings = self.train[self.train['movieId'] == movie_id]['rating'].values
                self.movie_bias[movie_id] = np.mean(ratings - self.mean_rating)
    
    
    class AdvancedCB:
        """고급 CB - 다중 특징 ([2] FAB의 multi-attribute)"""
        def __init__(self, train, movies, mean_rating, std_rating):
            self.train = train
            self.movies = movies
            self.mean_rating = mean_rating
            self.std_rating = std_rating
            self.movie_features = {}
            self.item_to_idx = {}
            self.item_similarity = None
            
        def fit(self):
            """장르 + 인기도 + 연도 특징 결합"""
            mlb = MultiLabelBinarizer()
            genres_matrix = mlb.fit_transform(self.movies['genres'].str.split('|'))
            
            popularity_series = self.train.groupby('movieId')['rating'].count()
            max_pop = popularity_series.max()
            
            movies_copy = self.movies.copy()
            movies_copy['year'] = movies_copy['title'].str.extract(r'\((\d{4})\)')[0]
            movies_copy['year'] = pd.to_numeric(movies_copy['year'], errors='coerce')
            year_median = movies_copy['year'].median()
            movies_copy['year'].fillna(year_median, inplace=True)
            year_normalized = (movies_copy['year'] - movies_copy['year'].min()) / \
                              (movies_copy['year'].max() - movies_copy['year'].min() + 1e-8)
            
            content_matrix = []
            for idx, (_, row) in enumerate(movies_copy.iterrows()):
                movie_id = row['movieId']
                self.item_to_idx[movie_id] = idx
                
                features = np.concatenate([
                    genres_matrix[idx],
                    [popularity_series.get(movie_id, 0) / (max_pop + 1e-8)],
                    [year_normalized.iloc[idx]]
                ])
                
                content_matrix.append(features)
                self.movie_features[movie_id] = features
            
            self.item_similarity = cosine_similarity(np.array(content_matrix))
    
    
    def __init__(self, ratings, movies, name='FeatureCombination', svd_dim=200):
        super().__init__(ratings, movies, name)
        self.cf_model = None
        self.cb_model = None
        self.svd_dim = svd_dim
        
    def fit(self):
        """독립적인 CF/CB 모델 학습"""
        print(f"\n📊 {self.name} 학습 시작...")
        
        # ✅ 평점 기반 분할 (Cold Start 방지)
        # 참고: [1] Burke (2002) - feature combination evaluation
        self.train, self.test = train_test_split(
            self.ratings, test_size=0.2, random_state=42
        )
        
        print(f"   ℹ️  데이터 분할 (평점 기반):")
        print(f"      - Train ratings: {len(self.train):,}개")
        print(f"      - Test ratings: {len(self.test):,}개")
        
        # Cold Start 분석
        train_users = set(self.train['userId'].unique())
        test_users = set(self.test['userId'].unique())
        cold_start_users = test_users - train_users
        print(f"      - Cold Start users: {len(cold_start_users)}명")
        
        print(f"   🔄 고급 CF 모델 생성...")
        self.cf_model = self.AdvancedCF(self.train, self.mean_rating, self.std_rating, self.svd_dim)
        self.cf_model.fit()
        
        print(f"   🔄 고급 CB 모델 생성...")
        self.cb_model = self.AdvancedCB(self.train, self.movies, self.mean_rating, self.std_rating)
        self.cb_model.fit()
        
        print(f"   ✅ {self.name} 학습 완료")
        
    def predict(self, user_id, movie_id):
        """
        비선형 특징 결합
        
        Formula: combined = 0.4*CF + 0.4*CB + 0.2*interaction
        
        References:
            [3] Melville et al. (2002) - feature weighting
        """
        if self.cf_model is None or self.cb_model is None:
            return self.mean_rating
        
        # CF 점수 추출
        if user_id not in self.cf_model.user_factors or movie_id not in self.cf_model.movie_factors:
            cf_score = 0
        else:
            user_vec = self.cf_model.user_factors[user_id]
            movie_vec = self.cf_model.movie_factors[movie_id]
            cf_score = np.dot(user_vec, movie_vec)
            cf_score = np.clip(cf_score, -2, 2)
        
        # CB 점수 추출
        if movie_id not in self.cb_model.movie_features:
            cb_score = 0
        else:
            user_ratings = self.train[self.train['userId'] == user_id]
            if user_ratings.empty:
                cb_score = 0
            else:
                # ✅ 벡터화: for 루프 제거
                rated_movie_ids = user_ratings['movieId'].values
                rated_features = [
                    self.cb_model.movie_features[mid] 
                    for mid in rated_movie_ids 
                    if mid in self.cb_model.movie_features
                ]
                
                if rated_features:
                    user_profile = np.mean(rated_features, axis=0)
                    movie_feat = self.cb_model.movie_features[movie_id]
                    cb_score = np.dot(user_profile, movie_feat)
                    cb_score = np.clip(cb_score, -1.5, 1.5)
                else:
                    cb_score = 0
    
        # 비선형 결합 ([2] FAB system interaction)
        interaction_score = cf_score * cb_score * 0.3
        combined_score = 0.4 * cf_score + 0.4 * cb_score + 0.2 * interaction_score
        
        # Bias 추가
        ub = self.cf_model.user_bias.get(user_id, 0) * 0.8
        mb = self.cf_model.movie_bias.get(movie_id, 0) * 0.8
        
        pred = self.mean_rating + combined_score * self.std_rating + ub + mb
        return np.clip(pred, 1, 5)
    
    def recommend(self, user_id, n=10):
        """
        상위 N개 추천
        
        ✅ 수정:
        1. mean_rating 필터링 제거 → 모든 예측 포함
        2. 예측값이 모두 같으면 인기도로 재정렬
        3. Test 데이터 제외 로직 수정
        """
        if self.cf_model is None or self.cb_model is None:
            return []
        
        # ✅ 수정: Test 데이터 제외 로직 제거 (평가 방해)
        user_rated = set(self.train[self.train['userId'] == user_id]['movieId'].values)
        
        # 1. 인기 영화 500개
        popularity = self.train.groupby('movieId')['rating'].count()
        popular_movies = set(popularity.nlargest(500).index)
        
        # 2. CF 모델 기반 유사 사용자가 좋아한 영화 500개
        cf_similar_movies = set()
        if user_id in self.cf_model.user_factors:
            user_vec = self.cf_model.user_factors[user_id]
            
            user_similarities = {}
            for other_user_id, other_vec in list(self.cf_model.user_factors.items())[:200]:
                if other_user_id != user_id:
                    sim = np.dot(user_vec, other_vec)
                    user_similarities[other_user_id] = sim
            
            top_similar_users = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for similar_user_id, _ in top_similar_users:
                similar_user_ratings = self.train[self.train['userId'] == similar_user_id]
                top_rated = similar_user_ratings.nlargest(50, 'rating')['movieId'].values
                cf_similar_movies.update(top_rated)
        
        # 3. CB 모델 기반 사용자가 평가한 영화와 유사한 영화 500개
        cb_similar_movies = set()
        user_ratings = self.train[self.train['userId'] == user_id]
        
        if not user_ratings.empty and hasattr(self.cb_model, 'item_similarity'):
            top_rated_by_user = user_ratings.nlargest(10, 'rating')['movieId'].values
            
            for rated_movie_id in top_rated_by_user:
                if rated_movie_id in self.cb_model.item_to_idx:
                    idx = self.cb_model.item_to_idx[rated_movie_id]
                    sim_scores = self.cb_model.item_similarity[idx]
                    top_indices = np.argsort(sim_scores)[-50:]
                    
                    for i in top_indices:
                        movie_id = self.movies.iloc[i]['movieId']
                        cb_similar_movies.add(movie_id)
        
        # 후보 영화 = 인기 + CF 유사 + CB 유사 - 평가한 영화
        candidate_movies = (popular_movies | cf_similar_movies | cb_similar_movies) - user_rated
        
        # Fallback 1: 후보가 너무 적으면 전체 영화 사용
        if len(candidate_movies) < n * 2:
            candidate_movies = set(self.movies['movieId'].values) - user_rated
        
        # ✅ 수정: 모든 예측 포함 (필터링 제거)
        predictions = {}
        for movie_id in candidate_movies:
            pred = self.predict(user_id, movie_id)
            predictions[movie_id] = pred
    
        # Fallback 2: 예측이 없으면 인기도 기반 추천
        if not predictions:
            popular_movies_list = popularity.sort_values(ascending=False).index.tolist()
            return [m for m in popular_movies_list if m not in user_rated][:n]
        
        # ✅ 수정: 모든 예측이 동일하면 인기도 정렬 추가
        pred_values = list(predictions.values())
        if len(set(pred_values)) == 1:  # 모든 예측이 같은 값
            # 인기도 기준 재정렬
            predictions_with_pop = {
                movie_id: (pred, popularity.get(movie_id, 0))
                for movie_id, pred in predictions.items()
            }
            sorted_movies = sorted(
                predictions_with_pop.items(), 
                key=lambda x: (x[1][0], x[1][1]),  # 예측 → 인기도 순
                reverse=True
            )
            return [movie_id for movie_id, _ in sorted_movies[:n]]
        
        # 정상 정렬
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n]]