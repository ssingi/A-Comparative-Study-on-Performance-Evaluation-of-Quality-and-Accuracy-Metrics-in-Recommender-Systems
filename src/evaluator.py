import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metrics import AdvancedMetrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class RecommenderEvaluator:
    """추천 시스템 평가기"""
    
    def __init__(self, model, metrics_class=AdvancedMetrics):
        self.model = model
        self.metrics = metrics_class
        
    def evaluate(self):
        """모델 평가 - 정확도, 순위, 다양성"""
        print(f"\n📊 {self.model.name} 평가 중...")
        
        # ═══════════════════════════════════════════════════════════════
        # 1. 정확도 평가
        # ═══════════════════════════════════════════════════════════════
        actuals = []
        predictions = []
        
        print("   ⏳ 정확도 지표 계산 중...")
        for _, row in tqdm(self.model.test.iterrows(), total=len(self.model.test), desc="   Accuracy"):
            user_id = row['userId']
            movie_id = row['movieId']
            actual = row['rating']
            
            pred = self.model.predict(user_id, movie_id)
            actuals.append(actual)
            predictions.append(pred)
        
        if len(actuals) == 0:
            return self._empty_result()
        
        rmse = self.metrics.rmse(actuals, predictions)
        mae = self.metrics.mae(actuals, predictions)
        
        # 전체 데이터셋 기준 (train + test)
        num_users = self.model.ratings['userId'].nunique()
        num_items = self.model.ratings['movieId'].nunique()
        
        print(f"   📊 희소성 계산:")
        print(f"      - 전체 사용자: {num_users:,}명")
        print(f"      - 전체 아이템: {num_items:,}개")
        print(f"      - 테스트 평점: {len(actuals):,}개")
        
        adjusted_rmse = self.metrics.sparsity_aware_rmse(actuals, predictions, num_users, num_items)
        adjusted_mae = self.metrics.sparsity_aware_mae(actuals, predictions, num_users, num_items)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. 순위 평가
        # ═══════════════════════════════════════════════════════════════
        print("   ⏳ 순위 지표 계산 중...")
        
        # 랜덤 샘플링
        all_test_users = self.model.test['userId'].unique()
        num_users_total = len(all_test_users)
        
        if num_users_total < 1000:
            num_eval_users = min(100, num_users_total)
        else:
            num_eval_users = min(300, num_users_total)
        
        np.random.seed(42)
        test_users = np.random.choice(all_test_users, size=num_eval_users, replace=False)
        
        print(f"   ℹ️  평가 대상: {num_eval_users}명 (전체 {num_users_total:,}명 중, 랜덤 샘플)")
        
        # ✅ 수정: 순위 지표 리스트 초기화
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        map_scores = []
        mrr_scores = []
        
        all_recommendations = []
        
        valid_ranking_count = 0
        recommendation_failures = 0
        no_relevant_items = 0
        
        for user_id in tqdm(test_users, desc="   Ranking"):
            # 추천 생성
            recommended = self.model.recommend(user_id, n=10)
            
            if len(recommended) == 0:
                recommendation_failures += 1
                continue
            
            # 관련 항목 (평점 4 이상)
            user_test = self.model.test[self.model.test['userId'] == user_id]
            relevant = user_test[user_test['rating'] >= 4]['movieId'].values.tolist()
            
            if len(relevant) == 0:
                no_relevant_items += 1
                continue
            
            # 유효한 평가
            valid_ranking_count += 1
            
            # ✅ 수정: 관련도 벡터 계산
            relevance = []
            for movie_id in recommended:
                if movie_id in user_test['movieId'].values:
                    actual_rating = user_test[user_test['movieId'] == movie_id]['rating'].values[0]
                    relevance.append(1 if actual_rating >= 4 else 0)
                else:
                    relevance.append(0)
            
            # ✅ 수정: 순위 지표 계산 및 저장
            precision_scores.append(self.metrics.precision_at_k(recommended, relevant, k=10))
            recall_scores.append(self.metrics.recall_at_k(recommended, relevant, k=10))
            f1_scores.append(self.metrics.f1_at_k(recommended, relevant, k=10))
            ndcg_scores.append(self.metrics.ndcg_at_k(relevance, len(relevant), k=10))
            map_scores.append(self.metrics.map_at_k(recommended, relevant, k=10))
            mrr_scores.append(self.metrics.mrr_at_k(recommended, relevant, k=10))
            
            # 다양성 계산용
            all_recommendations.extend(recommended)
        
        # 통계 출력
        print(f"   ℹ️  순위 평가:")
        print(f"      - 총 테스트 사용자: {len(test_users)}명")
        print(f"      - 유효한 평가: {valid_ranking_count}명")
        print(f"      - 추천 실패: {recommendation_failures}명")
        print(f"      - Relevant 항목 없음: {no_relevant_items}명 (평점 4 이상 없음)")
        
        # ═══════════════════════════════════════════════════════════════
        # 3. 다양성 평가
        # ═══════════════════════════════════════════════════════════════
        print("   ⏳ 다양성 지표 계산 중...")
        
        diversity_users = 0
        for user_id in tqdm(test_users, desc="   Diversity"):
            recommended = self.model.recommend(user_id, n=10)
            if len(recommended) > 0:
                diversity_users += 1
        
        print(f"   ℹ️  다양성 평가: {diversity_users}/{len(test_users)} 사용자")
        
        if len(all_recommendations) == 0:
            return self._empty_result()
        
        # 영화 특징 추출
        movie_features = self._extract_movie_features()
        
        diversity = self.metrics.diversity(all_recommendations, movie_features)
        
        total_items_count = len(self.model.movies)
        coverage = self.metrics.coverage(all_recommendations, total_items_count)
        
        popularity = self.model.train.groupby('movieId')['rating'].count().to_dict()
        novelty = self.metrics.novelty(all_recommendations, popularity)
        popularity_bias = self.metrics.popularity_bias(all_recommendations, popularity)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. 결과 반환
        # ═══════════════════════════════════════════════════════════════
        result = {
            'Model': self.model.name,
            'RMSE': rmse,
            'MAE': mae,
            'Adjusted_RMSE': adjusted_rmse,
            'Adjusted_MAE': adjusted_mae,
            'Precision@10': np.mean(precision_scores) if precision_scores else 0.0,
            'Recall@10': np.mean(recall_scores) if recall_scores else 0.0,
            'F1@10': np.mean(f1_scores) if f1_scores else 0.0,
            'NDCG@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'MAP@10': np.mean(map_scores) if map_scores else 0.0,
            'MRR@10': np.mean(mrr_scores) if mrr_scores else 0.0,
            'Diversity': diversity,
            'Coverage': coverage,
            'Novelty': novelty,
            'PopularityBias': popularity_bias
        }
        
        print(f"   ✅ {self.model.name} 평가 완료")
        return result
    
    def _extract_movie_features(self):
        """영화 특징 추출"""
        from sklearn.preprocessing import MultiLabelBinarizer
        
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(self.model.movies['genres'].str.split('|'))
        
        popularity_series = self.model.train.groupby('movieId')['rating'].count()
        max_pop = popularity_series.max() if len(popularity_series) > 0 else 1.0
        
        movie_features = {}
        for idx, (_, row) in enumerate(self.model.movies.iterrows()):
            movie_id = row['movieId']
            
            genre_vec = genres_matrix[idx]
            pop_score = popularity_series.get(movie_id, 0) / (max_pop + 1e-8)
            
            feature_vec = np.concatenate([genre_vec, [pop_score]])
            movie_features[movie_id] = feature_vec
        
        return movie_features
    
    def _empty_result(self):
        """빈 결과 딕셔너리 반환"""
        return {
            'Model': self.model.name,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Adjusted_RMSE': 0.0,
            'Adjusted_MAE': 0.0,
            'Precision@10': 0.0,
            'Recall@10': 0.0,
            'F1@10': 0.0,
            'NDCG@10': 0.0,
            'MAP@10': 0.0,
            'MRR@10': 0.0,
            'Diversity': 0.0,
            'Coverage': 0.0,
            'Novelty': 0.0,
            'PopularityBias': 0.0
        }