import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


class AdvancedMetrics:
    """추천 시스템을 정확도, 순위, 다양성 측면에서 평가"""
    
    # ═══════════════════════════════════════════════════════════════════════
    # 정확도 지표 (Accuracy Metrics)
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def rmse(actuals, predictions):
        """정확도 지표: RMSE - 평균제곱근오차"""
        if len(actuals) == 0:
            return 0.0
        return np.sqrt(mean_squared_error(actuals, predictions))

    @staticmethod
    def mae(actuals, predictions):
        """정확도 지표: MAE - 평균절대오차"""
        if len(actuals) == 0:
            return 0.0
        return mean_absolute_error(actuals, predictions)

    @staticmethod
    def sparsity_aware_rmse(actuals, predictions, num_users, num_items):
        """
        정확도 지표: 희소성 조정 RMSE
        
        Formula: Adjusted_RMSE = RMSE / (1 + Sparsity)
        
        희소성이 높을수록 조정된 RMSE는 낮아짐 (공정한 평가)
        """
        if len(actuals) == 0:
            return 0.0
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        total_possible = num_users * num_items
        actual_ratings = len(actuals)
        sparsity = 1 - (actual_ratings / (total_possible + 1e-8))
        # ✅ 수정: 곱셈 → 나눗셈
        return rmse / (1 + sparsity)

    @staticmethod
    def sparsity_aware_mae(actuals, predictions, num_users, num_items):
        """
        정확도 지표: 희소성 조정 MAE
        
        Formula: Adjusted_MAE = MAE / (1 + Sparsity)
        
        희소성이 높을수록 조정된 MAE는 낮아짐 (공정한 평가)
        """
        if len(actuals) == 0:
            return 0.0
        mae = mean_absolute_error(actuals, predictions)
        total_possible = num_users * num_items
        actual_ratings = len(actuals)
        sparsity = 1 - (actual_ratings / (total_possible + 1e-8))
        return mae / (1 + sparsity)

    # ═══════════════════════════════════════════════════════════════════════
    # 순위 지표 (Ranking Metrics)
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """순위 지표: Precision@K"""
        if k == 0:
            return 0.0
        rec_set = set(recommended[:k])
        rel_set = set(relevant)
        if len(rec_set) == 0:
            return 0.0
        return len(rec_set & rel_set) / k

    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """순위 지표: Recall@K"""
        rec_set = set(recommended[:k])
        rel_set = set(relevant)
        if len(rel_set) == 0:
            return 0.0
        return len(rec_set & rel_set) / len(rel_set)

    @staticmethod
    def f1_at_k(recommended, relevant, k=10):
        """순위 지표: F1@K"""
        precision = AdvancedMetrics.precision_at_k(recommended, relevant, k)
        recall = AdvancedMetrics.recall_at_k(recommended, relevant, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(relevance, k=10):
        """순위 지표: NDCG@K"""
        rel_array = np.array(relevance[:k])
        if len(rel_array) == 0:
            return 0.0
        gains = rel_array / np.log2(np.arange(2, len(rel_array) + 2))
        dcg = np.sum(gains)
        ideal_rel = np.sort(rel_array)[::-1]
        ideal_gains = ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2))
        idcg = np.sum(ideal_gains)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def map_at_k(recommended, relevant, k=10):
        """순위 지표: MAP@K"""
        rec_set = set(recommended[:k])
        rel_set = set(relevant)
        if len(rel_set) == 0:
            return 0.0
        score = 0.0
        num_hits = 0
        for i, rec in enumerate(recommended[:k]):
            if rec in rel_set:
                num_hits += 1
                score += num_hits / (i + 1)
        return score / min(k, len(rel_set))

    @staticmethod
    def mrr_at_k(recommended, relevant, k=10):
        """순위 지표: MRR@K"""
        rel_set = set(relevant)
        for i, rec in enumerate(recommended[:k]):
            if rec in rel_set:
                return 1.0 / (i + 1)
        return 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # 다양성 지표 (Diversity Metrics)
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def intra_list_diversity(recs, sim_matrix, item_to_idx):
        """다양성 지표: Intra-list Diversity"""
        if len(recs) < 2:
            return 0.0
        valid_recs = [r for r in recs if r in item_to_idx]
        if len(valid_recs) < 2:
            return 0.0
        indices = [item_to_idx[r] for r in valid_recs]
        if not indices:
            return 0.0
        max_idx = max(indices)
        if max_idx >= sim_matrix.shape[0]:
            return 0.0
        
        total_sim = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_sim += sim_matrix[indices[i], indices[j]]
                count += 1
        if count == 0:
            return 0.0
        avg_sim = total_sim / count
        return 1 - avg_sim

    @staticmethod
    def coverage(all_recs, total_items):
        """
        다양성 지표: Catalog Coverage
        
        Args:
            all_recs (list): 추천된 영화 ID 리스트
            total_items (int or DataFrame or Series): 전체 아이템 개수 또는 영화 데이터
        
        Returns:
            float: Coverage 점수 (0~1)
        """
        if len(all_recs) == 0:
            return 0.0
        
        unique_recs = len(set(all_recs))
        
        # ✅ 수정: total_items가 DataFrame/Series인 경우 처리
        if isinstance(total_items, (int, float)):
            num_items = total_items
        elif hasattr(total_items, '__len__'):  # DataFrame, Series, list 등
            num_items = len(total_items)
        else:
            return 0.0
        
        if num_items == 0:
            return 0.0
        
        return unique_recs / num_items

    @staticmethod
    def novelty(recs, popularity):
        """다양성 지표: Novelty"""
        if len(recs) == 0:
            return 0.0
        novelty_score = 0.0
        total_pop = sum(popularity.values()) if popularity else 1e-8
        total_pop = max(total_pop, 1e-8)
        for rec in recs:
            pop = popularity.get(rec, 1e-6)
            pop_ratio = pop / total_pop
            pop_ratio = max(min(pop_ratio, 1.0), 1e-8)
            novelty_score += -np.log2(pop_ratio)
        return novelty_score / len(recs)

    @staticmethod
    def popularity_bias(recs, popularity):
        """다양성 지표: Popularity Bias"""
        if len(recs) == 0:
            return 0.0
        if not popularity or len(popularity) == 0:
            return 0.0
        
        max_pop = max(popularity.values())
        min_pop = min(popularity.values())
        pop_range = max_pop - min_pop if max_pop > min_pop else 1.0
        
        bias_score = 0.0
        for rec in recs:
            pop = popularity.get(rec, 0)
            normalized_pop = (pop - min_pop) / pop_range if pop_range > 0 else 0.0
            bias_score += normalized_pop
        return bias_score / len(recs)

    @staticmethod
    def diversity(all_recs, movie_features):
        """
        다양성 지표: Intra-list Diversity (평균 비유사도)
        
        Args:
            all_recs (list): 추천된 영화 ID 리스트
            movie_features (dict): {movie_id: feature_vector}
        
        Returns:
            float: 다양성 점수 (0~1, 높을수록 다양함)
        """
        if len(all_recs) < 2:
            return 0.0
        
        # 유효한 추천만 필터링
        valid_recs = [r for r in all_recs if r in movie_features]
        if len(valid_recs) < 2:
            return 0.0
        
        # 특징 벡터 추출
        feature_vectors = np.array([movie_features[r] for r in valid_recs])
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(feature_vectors)
        
        # 대각선 제외 (자기 자신과의 유사도 제외)
        n = similarity_matrix.shape[0]
        total_sim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarity_matrix[i, j]
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_sim = total_sim / count
        
        # 다양성 = 1 - 평균 유사도
        return 1 - avg_sim