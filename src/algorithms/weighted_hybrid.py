"""
가중 평균 하이브리드 (Weighted Hybrid Recommender)
- 참고문헌:
  [1] Burke, R. (2002). "Hybrid recommender systems: Survey and experiments."
  [2] Claypool, M., et al. (1999). "Combining content-based and collaborative filters."
  [3] Koren, Y. (2009). "Matrix Factorization Techniques."
  
하이퍼파라미터 근거:
- svd_dim=200: [3] Netflix Prize 100-300 차원 사용
- alpha/beta 가중치: [2] 사용자 활동성 기반 적응적 가중치
- activity_weight: [1] cold-start 완화 전략
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from .base_recommender import BaseRecommender
import pandas as pd


class WeightedHybrid(BaseRecommender):
    """
    가중 평균 하이브리드 - 독립적인 고급 CF/CB 결합
    
    Formula: r̂ = α * r̂_CF + β * r̂_CB
    
    References:
        [1] Burke (2002) - Weighted hybrid design
        [2] Claypool et al. (1999) - Adaptive weighting
        [3] Koren (2009) - Advanced CF with bias
    """
    
    class AdvancedCF:
        """고급 CF - Bias 포함 행렬 분해 ([3] Koren 2009)"""
        
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
            """SVD + Bias 학습"""
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
            
            # Bias 계산 ([3] Koren 2009)
            for user_id in self.train['userId'].unique():
                ratings = self.train[self.train['userId'] == user_id]['rating'].values
                self.user_bias[user_id] = np.mean(ratings - self.mean_rating)
            
            for movie_id in self.train['movieId'].unique():
                ratings = self.train[self.train['movieId'] == movie_id]['rating'].values
                self.movie_bias[movie_id] = np.mean(ratings - self.mean_rating)
        
        def predict(self, user_id, movie_id):
            """r̂_ui = μ + b_u + b_i + <p_u, q_i> * activity_weight"""
            if user_id not in self.user_factors or movie_id not in self.movie_factors:
                return self.mean_rating
            
            latent_score = np.dot(self.user_factors[user_id], self.movie_factors[movie_id])
            latent_score = np.clip(latent_score, -2, 2)
            
            # 활동성 기반 가중치 ([1] Burke 2002 - cold-start 완화)
            user_count = len(self.train[self.train['userId'] == user_id])
            activity_weight = min(user_count / 20.0, 1.5)
            
            scaled_score = latent_score * self.std_rating * activity_weight
            
            # Bias 적용
            ub = self.user_bias.get(user_id, 0) * 1.2
            mb = self.movie_bias.get(movie_id, 0) * 1.2
            
            pred = self.mean_rating + scaled_score + ub + mb
            return np.clip(pred, 1, 5)
    
    
    class AdvancedCB:
        """고급 CB - 다중 특징 결합"""
        
        def __init__(self, train, movies, mean_rating, std_rating):
            self.train = train
            self.movies = movies
            self.mean_rating = mean_rating
            self.std_rating = std_rating
            self.movie_features = {}
            self.item_to_idx = {}
            self.item_similarity = None
            
        def fit(self):
            """5가지 특징 결합: 장르, 인기도, 참신성, 연도, 평균평점"""
            mlb = MultiLabelBinarizer()
            genres_matrix = mlb.fit_transform(self.movies['genres'].str.split('|'))
            
            # 인기도/참신성 특징
            popularity_series = self.train.groupby('movieId')['rating'].count()
            max_pop = popularity_series.max()
            min_pop = popularity_series.min()
            novelty_feature = 1 - ((popularity_series - min_pop) / (max_pop - min_pop + 1e-8))
            
            # 연도 특징
            movies_copy = self.movies.copy()
            movies_copy['year'] = movies_copy['title'].str.extract(r'\((\d{4})\)')[0]
            movies_copy['year'] = pd.to_numeric(movies_copy['year'], errors='coerce')
            year_median = movies_copy['year'].median()
            movies_copy['year'].fillna(year_median, inplace=True)
            year_normalized = (movies_copy['year'] - movies_copy['year'].min()) / \
                              (movies_copy['year'].max() - movies_copy['year'].min() + 1e-8)
            
            # 평균 평점 특징
            movie_avg_rating = self.train.groupby('movieId')['rating'].mean()
            rating_normalized = (movie_avg_rating - movie_avg_rating.min()) / \
                                (movie_avg_rating.max() - movie_avg_rating.min() + 1e-8)
            
            # 특징 결합
            content_matrix = []
            for idx, (_, row) in enumerate(movies_copy.iterrows()):
                movie_id = row['movieId']
                self.item_to_idx[movie_id] = idx
                
                features = np.concatenate([
                    genres_matrix[idx],
                    [popularity_series.get(movie_id, 0) / (max_pop + 1e-8)],
                    [novelty_feature.get(movie_id, 0.5)],
                    [year_normalized.iloc[idx]],
                    [rating_normalized.get(movie_id, 0.5)]
                ])
                
                content_matrix.append(features)
                self.movie_features[movie_id] = features
            
            self.item_similarity = cosine_similarity(np.array(content_matrix))
        
        def predict(self, user_id, movie_id):
            """신뢰도 가중 예측 ([1] Burke 2002 - confidence weighting)"""
            if movie_id not in self.item_to_idx:
                return self.mean_rating
            
            user_ratings = self.train[self.train['userId'] == user_id]
            if user_ratings.empty:
                return self.mean_rating
            
            num_ratings = len(user_ratings)
            
            # 활동성/일관성 신뢰도
            activity_confidence = min(num_ratings / 10.0, 1.2)
            
            if num_ratings > 1:
                rating_std = user_ratings['rating'].std()
                consistency_confidence = 1.0 / (1.0 + rating_std / 2.5 * 0.3)
            else:
                consistency_confidence = 0.7
            
            overall_confidence = (activity_confidence + 2*consistency_confidence) / 3.0
            
            # ✅ 벡터화: iterrows() 제거
            target_idx = self.item_to_idx[movie_id]
            
            rated_movie_ids = user_ratings['movieId'].values
            valid_mask = np.array([mid in self.item_to_idx for mid in rated_movie_ids])
            
            if not valid_mask.any():
                return self.mean_rating
            
            valid_movie_ids = rated_movie_ids[valid_mask]
            valid_ratings = user_ratings['rating'].values[valid_mask]
            rated_indices = np.array([self.item_to_idx[mid] for mid in valid_movie_ids])
            
            # 유사도 벡터 추출
            similarities = self.item_similarity[target_idx, rated_indices]
            
            # 유사도 임계값 필터링
            sim_mask = similarities > 0.05
            if not sim_mask.any():
                return self.mean_rating
            
            filtered_similarities = similarities[sim_mask]
            filtered_ratings = valid_ratings[sim_mask]
            
            # 가중 평균 계산
            weighted_ratings = (filtered_ratings - self.mean_rating) * filtered_similarities * overall_confidence
            cb_score = np.mean(weighted_ratings)
            cb_score = np.clip(cb_score, -1.5, 1.5)
            
            pred = self.mean_rating + cb_score * self.std_rating * 1.3
            return np.clip(pred, 1, 5)
    
    
    def __init__(self, ratings, movies, name='WeightedHybrid', svd_dim=200):
        super().__init__(ratings, movies, name)
        self.cf_model = None
        self.cb_model = None
        self.svd_dim = svd_dim
        
    def fit(self):
        """독립적인 CF/CB 모델 학습"""
        print(f"\n📊 {self.name} 학습 시작...")
        
        # ✅ 평점 기반 분할 (Cold Start 방지)
        # 참고: [1] Burke (2002) - hybrid system evaluation
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
        
        print(f"      - Cold Start users: {len(cold_start_users)}명 ({len(cold_start_users)/len(test_users)*100:.1f}%)")
        
        print(f"   🔄 고급 CF 모델 생성...")
        self.cf_model = self.AdvancedCF(self.train, self.mean_rating, self.std_rating, self.svd_dim)
        self.cf_model.fit()
        
        print(f"   🔄 고급 CB 모델 생성...")
        self.cb_model = self.AdvancedCB(self.train, self.movies, self.mean_rating, self.std_rating)
        self.cb_model.fit()
        
        print(f"   ✅ {self.name} 학습 완료")
        
    def predict(self, user_id, movie_id):
        """
        적응적 가중 평균: r̂ = α * r̂_CF + β * r̂_CB
        
        가중치 전략 ([2] Claypool et al. 1999):
            - High activity (≥50): α=0.65, β=0.35 (CF 우선)
            - Medium (≥20): α=0.50, β=0.50 (균형)
            - Low (<20): α=0.35, β=0.65 (CB 우선, cold-start)
        """
        if self.cf_model is None or self.cb_model is None:
            return self.mean_rating
        
        cf_pred = self.cf_model.predict(user_id, movie_id)
        cb_pred = self.cb_model.predict(user_id, movie_id)
        
        # 활동성 기반 적응적 가중치
        user_ratings_count = len(self.train[self.train['userId'] == user_id])
        
        if user_ratings_count >= 50:
            alpha, beta = 0.65, 0.35  # CF 우선
        elif user_ratings_count >= 20:
            alpha, beta = 0.50, 0.50  # 균형
        else:
            alpha, beta = 0.35, 0.65  # CB 우선 (cold-start)
        
        pred = alpha * cf_pred + beta * cb_pred
        return np.clip(pred, 1, 5)
    
    def recommend(self, user_id, n=10):
        """
        상위 N개 추천
        
        ✅ 최적화: 후보 영화 제한 (인기 영화 + 유사 영화)
        """
        if self.cf_model is None or self.cb_model is None:
            return []
        
        user_rated = set(self.train[self.train['userId'] == user_id]['movieId'].values)
        
        # ✅ 1. 인기 영화 500개
        popularity = self.train.groupby('movieId')['rating'].count()
        popular_movies = set(popularity.nlargest(500).index)
        
        # ✅ 2. CF 모델 기반 유사 사용자가 좋아한 영화 500개
        similar_movies = set()
        if user_id in self.cf_model.user_factors:
            user_vec = self.cf_model.user_factors[user_id]
            
            # 유사 사용자 찾기
            user_similarities = {}
            for other_user_id, other_vec in list(self.cf_model.user_factors.items())[:200]:
                if other_user_id != user_id:
                    sim = np.dot(user_vec, other_vec)
                    user_similarities[other_user_id] = sim
            
            # 상위 10명의 유사 사용자
            top_similar_users = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for similar_user_id, _ in top_similar_users:
                similar_user_ratings = self.train[self.train['userId'] == similar_user_id]
                top_rated = similar_user_ratings.nlargest(50, 'rating')['movieId'].values
                similar_movies.update(top_rated)
        
        # 후보 영화 = 인기 + 유사 - 평가한 영화
        candidate_movies = (popular_movies | similar_movies) - user_rated
        
        # Fallback
        if len(candidate_movies) < n * 2:
            candidate_movies = set(self.movies['movieId'].values) - user_rated
        
        # 예측
        predictions = {}
        for movie_id in candidate_movies:
            predictions[movie_id] = self.predict(user_id, movie_id)
        
        if not predictions:
            return []
        
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n]]