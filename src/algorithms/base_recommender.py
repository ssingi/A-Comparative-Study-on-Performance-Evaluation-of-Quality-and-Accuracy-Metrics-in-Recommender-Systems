import numpy as np
from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """추천 시스템 베이스 클래스"""
    
    def __init__(self, ratings, movies, name='BaseRecommender', share_data=False):
        # share_data=True면 원본 참조, False면 복사본 사용
        if share_data:
            self.ratings = ratings
            self.movies = movies
        else:
            self.ratings = ratings.copy()
            self.movies = movies.copy()
            
        self.name = name
        
        self.mean_rating = self.ratings['rating'].mean()
        self.std_rating = self.ratings['rating'].std()
        
        self.train = None
        self.test = None
        
    @abstractmethod
    def fit(self):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, user_id, movie_id):
        """평점 예측"""
        pass
    
    @abstractmethod
    def recommend(self, user_id, n=10):
        """상위 N개 추천"""
        pass