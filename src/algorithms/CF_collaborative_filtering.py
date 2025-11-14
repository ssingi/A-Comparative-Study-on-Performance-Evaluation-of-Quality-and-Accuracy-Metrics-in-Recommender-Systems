"""
기초 협력 필터링 (Basic Collaborative Filtering)
- 알고리즘: Matrix Factorization via SVD
- 참고문헌:
  [1] Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques 
      for Recommender Systems." Computer, 42(8), 30-37.
  [2] Funk, S. (2006). "Netflix Update: Try This at Home." Simon Funk Blog.
  
하이퍼파라미터 근거:
- svd_dim=50: [1]에서 Netflix Prize 데이터셋에서 50-200 차원이 적절하다고 제안
- global_mean 사용: [1]의 bias 모델링 기법
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from .base_recommender import BaseRecommender


class CollaborativeFiltering(BaseRecommender):
    """
    기초 협력 필터링 (Basic CF) - SVD 기반 행렬 분해
    
    References:
        [1] Koren et al. (2009) - Matrix Factorization Techniques
        [2] Funk (2006) - SVD-based CF
    """
    
    def __init__(self, ratings, movies, name='CF', svd_dim=50):
        """
        Args:
            svd_dim (int): SVD 잠재 요인 차원 (기본값 50)
                - 근거: [1] Koren et al. (2009)에서 50-200 권장
        """
        super().__init__(ratings, movies, name)
        self.svd_dim = svd_dim
        
        self.user_factors = {}
        self.movie_factors = {}
        self.global_mean = None
        
    def fit(self):
        """
        SVD 기반 협력 필터링 학습
        
        알고리즘:
            1. User-Movie 행렬 생성
            2. TruncatedSVD로 저차원 분해
            3. 사용자/영화 잠재 요인 저장
    
        References:
            [1] Koren et al. (2009) - SVD 기반 분해 방법론
        """
        print(f"\n📊 {self.name} 학습 시작...")
        
        # ✅ 평점 기반 분할 (사용자 기반 X)
        # 이유: Cold Start 방지 - 모든 사용자가 train에 포함되어야 예측 가능
        # 참고: [1] Koren et al. (2009) - random rating split
        self.train, self.test = train_test_split(
            self.ratings, test_size=0.2, random_state=42
        )
        print(f"   ✅ Train: {len(self.train):,}, Test: {len(self.test):,}")
        
        # 전역 평균 계산 ([1] Baseline predictor)
        self.global_mean = self.train['rating'].mean()
        
        # User-Movie 평점 행렬 생성 (결측치는 전역 평균으로 대체)
        um = self.train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(self.global_mean)
        print(f"   ✅ 행렬 크기: {um.shape}")
        
        # SVD 분해 ([2] Funk SVD)
        print(f"   🔄 SVD 분해 ({self.svd_dim}차원)...")
        svd = TruncatedSVD(
            n_components=min(self.svd_dim, um.shape[0]-1, um.shape[1]-1),
            random_state=42
        )
        user_features = svd.fit_transform(um)
        movie_features = svd.components_.T
        
        # 잠재 요인 저장
        for i, user_id in enumerate(um.index):
            self.user_factors[user_id] = user_features[i]
        
        for i, movie_id in enumerate(um.columns):
            self.movie_factors[movie_id] = movie_features[i]
        
        print(f"   ✅ {self.name} 학습 완료 ({len(self.user_factors)}명, {len(self.movie_factors)}개)")
        
    def predict(self, user_id, movie_id):
        """
        평점 예측: r̂_ui = <p_u, q_i>
        
        References:
            [1] Koren et al. (2009) - 내적 기반 예측
        """
        if user_id not in self.user_factors or movie_id not in self.movie_factors:
            return self.global_mean
        
        # 사용자-아이템 잠재 벡터 내적 ([1] 기본 MF 예측)
        pred = np.dot(self.user_factors[user_id], self.movie_factors[movie_id])
        return np.clip(pred, 1, 5)
    
    def recommend(self, user_id, n=10):
        """상위 N개 추천"""
        user_rated = set(self.train[self.train['userId'] == user_id]['movieId'].values)
        
        predictions = {}
        for movie_id in self.movies['movieId'].values:
            if movie_id not in user_rated:
                predictions[movie_id] = self.predict(user_id, movie_id)
        
        if not predictions:
            return []
        
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n]]