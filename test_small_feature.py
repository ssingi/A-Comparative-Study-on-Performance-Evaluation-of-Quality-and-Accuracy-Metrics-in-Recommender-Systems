"""
Small 데이터셋 FeatureCombination 테스트
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_loader import load_movielens
from src.algorithms.feature_combination import FeatureCombination
from src.evaluator import RecommenderEvaluator
import pandas as pd

print("=" * 100)
print("🧪 Small 데이터셋 FeatureCombination 테스트 (디버깅)")
print("=" * 100)

# 데이터 로드
ratings, movies = load_movielens(dataset_type='Small')

# 모델 생성 및 학습
model = FeatureCombination(ratings, movies, name='FeatureCombination', svd_dim=200)
model.fit()

# 샘플 사용자 추천 테스트
test_user_id = model.test['userId'].iloc[0]
print(f"\n📌 사용자 {test_user_id} 디버깅:")

# 1. CF/CB 점수 확인
print(f"\n   🔍 모델 상태:")
print(f"      - CF user_factors 존재: {test_user_id in model.cf_model.user_factors}")
print(f"      - 사용자 평가 영화 수: {len(model.train[model.train['userId'] == test_user_id])}")

# 2. 샘플 영화에 대한 예측
sample_movie_id = model.movies.iloc[0]['movieId']
pred = model.predict(test_user_id, sample_movie_id)
print(f"\n   🎬 샘플 예측:")
print(f"      - 영화 ID: {sample_movie_id}")
print(f"      - 예측 평점: {pred:.4f}")
print(f"      - 평균 평점: {model.mean_rating:.4f}")
print(f"      - 차이: {abs(pred - model.mean_rating):.4f}")

# 3. 추천 생성
recommendations = model.recommend(test_user_id, n=10)
print(f"\n   ⭐ 추천 결과:")
print(f"      - 추천 영화 수: {len(recommendations)}")

if len(recommendations) == 0:
    print("      ❌ 추천 실패!")
    print("\n   🔍 문제 진단:")
    
    # 후보 영화 확인
    user_rated = set(model.train[model.train['userId'] == test_user_id]['movieId'].values)
    popularity = model.train.groupby('movieId')['rating'].count()
    popular_movies = set(popularity.nlargest(500).index)
    candidate_movies = popular_movies - user_rated
    
    print(f"      - 사용자 평가 영화: {len(user_rated)}")
    print(f"      - 인기 영화: {len(popular_movies)}")
    print(f"      - 후보 영화: {len(candidate_movies)}")
    
    # 샘플 예측 확인
    if len(candidate_movies) > 0:
        sample_candidates = list(candidate_movies)[:10]
        print(f"\n      - 샘플 후보 예측:")
        for cand_movie_id in sample_candidates:
            cand_pred = model.predict(test_user_id, cand_movie_id)
            print(f"         Movie {cand_movie_id}: {cand_pred:.4f}")
else:
    for i, movie_id in enumerate(recommendations, 1):
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        pred_rating = model.predict(test_user_id, movie_id)
        print(f"      {i}. {movie_title}: {pred_rating:.2f}점")

# 평가
print("\n" + "=" * 100)
print("📊 모델 평가")
print("=" * 100)

evaluator = RecommenderEvaluator(model)
result = evaluator.evaluate()

print("\n📊 평가 결과:")
print(f"   순위:")
print(f"      - Precision@10: {result['Precision@10']:.4f}")
print(f"      - Recall@10: {result['Recall@10']:.4f}")
print(f"      - NDCG@10: {result['NDCG@10']:.4f}")

if result['Precision@10'] == 0.0:
    print("\n   ⚠️  순위 지표가 0.0입니다!")
    print("   🔍 원인:")
    print("      1. recommend()가 빈 리스트 반환")
    print("      2. 예측값이 모두 mean_rating과 동일")
    print("      3. CF/CB 점수가 모두 0")

print("\n✅ 테스트 완료")