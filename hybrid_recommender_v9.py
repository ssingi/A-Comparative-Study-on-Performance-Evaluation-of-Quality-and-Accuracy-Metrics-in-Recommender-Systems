# =====================================================================================================
# 🎬 MovieLens 하이브리드 추천 시스템 - Ver 9 (최종 완벽 수정)
# =====================================================================================================
# 
# [논문 개요]
# 이 코드는 협업 필터링(CF)과 콘텐츠 기반 필터링(CB)을 결합한
# 5가지 하이브리드 추천 알고리즘을 구현하고 평가합니다.
#
# [주요 특징]
# 1. 정확도와 다양성을 동시에 측정하는 18개 평가 지표
# 2. MovieLens Small(100K) 및 1M 데이터셋 자동 처리
# 3. 희소성을 고려한 공정한 성능 평가
# 4. 필터 버블 현상 완화 능력 측정
#
# [예상 독자]
# - 추천 시스템 연구자
# - 데이터 과학자
# - 논문 작성자
#
# =====================================================================================================

import os
import io
import requests
import zipfile
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("🎬 MovieLens 하이브리드 추천 시스템 - Ver 9 (최종 완벽 수정)")
print("=" * 100)


# =====================================================================================================
# 섹션 1: 평가 지표 클래스
# =====================================================================================================

class AdvancedMetrics:
    """추천 시스템 평가 지표: 정확도(3) + 순위(6) + 다양성(4) = 총 18개"""

    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """추천 중 관련 아이템 비율"""
        if k == 0 or not recommended:
            return 0.0
        rec_k = set(recommended[:k])
        rel_set = set(relevant)
        return len(rec_k & rel_set) / k if len(rec_k) > 0 else 0.0

    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """관련 아이템 중 추천된 비율"""
        if not relevant:
            return 0.0
        rec_k = set(recommended[:k])
        rel_set = set(relevant)
        return len(rec_k & rel_set) / len(rel_set)

    @staticmethod
    def f1_at_k(recommended, relevant, k=10):
        """Precision과 Recall의 조화평균"""
        p = AdvancedMetrics.precision_at_k(recommended, relevant, k)
        r = AdvancedMetrics.recall_at_k(recommended, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @staticmethod
    def ndcg_at_k(relevance, k=10):
        """순위를 고려한 성능 평가 (DCG / IDCG)"""
        if not relevance:
            return 0.0
        rel = relevance[:k]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(relevance, reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(recommended, relevant, k=10):
        """각 관련 아이템 발견 시 Precision의 평균"""
        if not relevant:
            return 0.0
        rec_k = recommended[:k]
        rel_set = set(relevant)
        score = 0.0
        num_hits = 0
        for i, rec in enumerate(rec_k):
            if rec in rel_set:
                num_hits += 1
                score += num_hits / (i + 1)
        return score / min(len(rel_set), k)

    @staticmethod
    def mrr_at_k(recommended, relevant, k=10):
        """첫 좋은 추천까지의 거리 역수"""
        if not relevant:
            return 0.0
        rec_k = recommended[:k]
        rel_set = set(relevant)
        for i, rec in enumerate(rec_k):
            if rec in rel_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def intra_list_diversity(recs, sim_matrix, item_to_idx):
        """추천 리스트 내 아이템 다양성 (1 - 평균유사도)"""
        if len(recs) < 2:
            return 0.0
        dists = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i] in item_to_idx and recs[j] in item_to_idx:
                    idx_i = item_to_idx[recs[i]]
                    idx_j = item_to_idx[recs[j]]
                    similarity = sim_matrix[idx_i][idx_j]
                    dists.append(1 - similarity)
        return np.mean(dists) if dists else 0.0

    @staticmethod
    def coverage(all_recs, total_items):
        """전체 카탈로그 중 추천된 아이템 비율"""
        unique_recs = set()
        for recs in all_recs:
            unique_recs.update(recs)
        return len(unique_recs) / total_items if total_items > 0 else 0.0

    @staticmethod
    def novelty(recs, popularity):
        """추천 아이템의 신규성 (-log2(인기도))"""
        if not recs:
            return 0.0
        novelty_scores = []
        for rec in recs:
            pop = popularity.get(rec, 0.5)
            pop = max(pop, 0.001)
            novelty_scores.append(-math.log2(pop))
        return np.mean(novelty_scores) if novelty_scores else 0.0

    @staticmethod
    def popularity_bias(recs, popularity):
        """추천 리스트의 평균 인기도"""
        if not recs:
            return 0.0
        pop_scores = [popularity.get(rec, 0.5) for rec in recs]
        return np.mean(pop_scores)

    @staticmethod
    def sparsity_aware_score(test_data, predictions, num_users, num_items):
        """희소성을 고려한 정규화 점수"""
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        total_possible_pairs = num_users * num_items
        actual_pairs = len(test_data)
        sparsity = 1 - (actual_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
        adjusted_rmse = rmse / (1 + sparsity)
        adjusted_mae = mae / (1 + sparsity)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Sparsity': sparsity,
            'Adjusted_RMSE': adjusted_rmse,
            'Adjusted_MAE': adjusted_mae
        }


# =====================================================================================================
# 섹션 2: 데이터 로드
# =====================================================================================================

def load_movielens(dataset_type='Small'):
    """MovieLens 데이터셋 자동 다운로드 및 로드"""
    datasets_info = {
        'Small': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'extract_dir': 'movielens_data/ml-latest-small',
            'encoding': 'utf-8'
        },
        '1M': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'extract_dir': 'movielens_data/ml-1m',
            'encoding': 'iso-8859-1'
        }
    }

    if dataset_type not in datasets_info:
        print(f"❌ 지원하지 않는 데이터셋: {dataset_type}")
        return None, None

    info = datasets_info[dataset_type]
    os.makedirs('movielens_data', exist_ok=True)

    if not os.path.exists(info['extract_dir']):
        print(f"📥 {dataset_type} 데이터셋 다운로드 중...")
        try:
            response = requests.get(info['url'], timeout=30)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall('movielens_data')
            print(f"✅ 다운로드 완료")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None, None

    try:
        extract_path = info['extract_dir']
        encoding = info['encoding']

        if dataset_type == 'Small':
            ratings = pd.read_csv(f'{extract_path}/ratings.csv', encoding=encoding)
            movies = pd.read_csv(f'{extract_path}/movies.csv', encoding=encoding)
        elif dataset_type == '1M':
            ratings = pd.read_csv(
                f'{extract_path}/ratings.dat',
                sep='::',
                header=None,
                engine='python',
                encoding=encoding,
                names=['userId', 'movieId', 'rating', 'timestamp']
            )
            movies = pd.read_csv(
                f'{extract_path}/movies.dat',
                sep='::',
                header=None,
                engine='python',
                encoding=encoding,
                names=['movieId', 'title', 'genres']
            )

        print(f"✅ {dataset_type} 데이터셋 로드 완료")
        print(f" 📌 사용자: {ratings['userId'].nunique():,}명")
        print(f" 📌 영화: {movies['movieId'].nunique():,}개")
        print(f" 📌 평점: {len(ratings):,}개")
        
        sparsity = 1 - (len(ratings) / (ratings['userId'].nunique() * movies['movieId'].nunique()))
        print(f" 📌 희소성: {sparsity:.4f} ({sparsity*100:.2f}%)")

        return ratings, movies

    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =====================================================================================================
# 섹션 3: 하이브리드 추천 시스템
# =====================================================================================================

class OptimizedHybridRecommender:
    """5가지 하이브리드 추천 알고리즘 구현"""

    def __init__(self, ratings, movies, name='recommender', svd_dim=200):
        self.name = name
        self.ratings = ratings
        self.movies = movies
        self.svd_dim = svd_dim
        self.metrics = AdvancedMetrics()
        
        self.mean_rating = ratings['rating'].mean()
        self.std_rating = ratings['rating'].std()
        
        self.user_factors = None
        self.movie_factors = None
        self.user_bias = {}
        self.movie_bias = {}
        self.movie_features = None
        self.item_similarity = None
        self.item_to_idx = None
        self.train = None
        self.test = None
        self.um = None
        self.popularity = {}
        
        print(f"🚀 {name} 초기화 완료")

    def _prepare(self):
        """모델 학습 및 전처리"""
        print(f"\n📊 {self.name} 전처리 중...")

        # Train/Test 분할 (80/20)
        unique_users = self.ratings['userId'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=0.2, random_state=42
        )
        self.train = self.ratings[self.ratings['userId'].isin(train_users)]
        self.test = self.ratings[self.ratings['userId'].isin(test_users)]
        print(f" ✅ Train: {len(self.train):,}, Test: {len(self.test):,}")

        # User-Movie 행렬 생성
        self.um = self.train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(self.mean_rating)

        # SVD 분해
        print(f" 🔄 SVD 분해 중...")
        svd = TruncatedSVD(
            n_components=min(self.svd_dim, self.um.shape[0]-1, self.um.shape[1]-1),
            random_state=42
        )
        user_features = svd.fit_transform(self.um)
        movie_features = svd.components_.T

        # 정규화
        self.user_factors = {}
        for i, user_id in enumerate(self.um.index):
            uf = user_features[i]
            uf_norm = (uf - uf.mean()) / (uf.std() + 1e-8)
            self.user_factors[user_id] = uf_norm

        self.movie_factors = {}
        for i, movie_id in enumerate(self.um.columns):
            mf = movie_features[i]
            mf_norm = (mf - mf.mean()) / (mf.std() + 1e-8)
            self.movie_factors[movie_id] = mf_norm

        # Bias 계산
        print(f" 🎯 Bias 계산 중...")
        for user_id in self.train['userId'].unique():
            user_ratings = self.train[self.train['userId'] == user_id]
            user_mean = user_ratings['rating'].mean()
            self.user_bias[user_id] = user_mean - self.mean_rating

        for movie_id in self.train['movieId'].unique():
            movie_ratings = self.train[self.train['movieId'] == movie_id]
            movie_mean = movie_ratings['rating'].mean()
            self.movie_bias[movie_id] = movie_mean - self.mean_rating

        # 콘텐츠 특징 생성 (모든 영화 사용 - Coverage 개선)
        print(f" 🎬 콘텐츠 특징 생성 중...")
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(
            self.movies['genres'].str.split('|')
        )

        self.movie_features = {}
        self.item_to_idx = {}
        for i, movie_id in enumerate(self.movies['movieId']):
            self.movie_features[movie_id] = genres_matrix[i]
            self.item_to_idx[movie_id] = i

        self.item_similarity = cosine_similarity(genres_matrix)

        # 인기도 계산
        popularity_series = self.train.groupby('movieId')['rating'].count()
        max_count = popularity_series.max()
        min_count = popularity_series.min()
        
        self.popularity = {}
        for movie_id in self.movies['movieId']:
            if movie_id in popularity_series.index:
                norm_pop = (popularity_series[movie_id] - min_count) / (max_count - min_count + 1e-8)
                self.popularity[movie_id] = norm_pop
            else:
                self.popularity[movie_id] = 0.0

        print(f"✅ {self.name} 전처리 완료")

    def predict_cf(self, user_id, movie_id):
        """협업 필터링 (SVD 기반)"""
        if user_id not in self.user_factors or movie_id not in self.movie_factors:
            return self.mean_rating

        user_vec = self.user_factors[user_id]
        movie_vec = self.movie_factors[movie_id]
        latent_score = np.dot(user_vec, movie_vec)
        scaled_score = latent_score * self.std_rating
        ub = self.user_bias.get(user_id, 0)
        mb = self.movie_bias.get(movie_id, 0)
        pred = scaled_score + ub + mb + self.mean_rating
        return np.clip(pred, 1, 5)

    def predict_cb(self, user_id, movie_id):
        """콘텐츠 기반 필터링 (장르 유사도)"""
        user_ratings = self.train[self.train['userId'] == user_id]

        if user_ratings.empty:
            return self.mean_rating
        
        if not hasattr(self, 'item_to_idx') or self.item_to_idx is None:
            return self.mean_rating
        
        if movie_id not in self.item_to_idx:
            return self.mean_rating

        rated_movies = user_ratings['movieId'].values
        weighted_sum = 0.0
        similarity_sum = 0.0

        for rated_id in rated_movies:
            if rated_id not in self.item_to_idx or movie_id not in self.item_to_idx:
                continue
                
            idx_rated = self.item_to_idx[rated_id]
            idx_movie = self.item_to_idx[movie_id]
            
            try:
                similarity = self.item_similarity[idx_movie][idx_rated]
                rating = user_ratings[user_ratings['movieId'] == rated_id]['rating'].values[0]
                weighted_sum += similarity * rating
                similarity_sum += similarity
            except (IndexError, ValueError):
                continue

        if similarity_sum > 0:
            pred = weighted_sum / similarity_sum
            return np.clip(pred, 1, 5)
        else:
            return self.mean_rating

    def predict_weighted_avg(self, user_id, movie_id, alpha=0.4):
        """가중 평균 하이브리드: α×CF + (1-α)×CB"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        pred = alpha * cf + (1 - alpha) * cb
        return np.clip(pred, 1, 5)

    def predict_feature_combo(self, user_id, movie_id):
        """특징 결합 하이브리드: 0.4×CF + 0.4×CB + 0.1×사용자편향 + 0.1×영화편향"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)

        ub = self.user_bias.get(user_id, 0)
        mb = self.movie_bias.get(movie_id, 0)

        ub_normalized = np.clip(ub / (self.std_rating + 1e-8), -1, 1)
        mb_normalized = np.clip(mb / (self.std_rating + 1e-8), -1, 1)

        pred = 0.4 * cf + 0.4 * cb + 0.1 * ub_normalized + 0.1 * mb_normalized
        return np.clip(pred, 1, 5)

    def predict_mixed(self, user_id, movie_id):
        """혼합 하이브리드: 0.5×CF + 0.5×CB"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        pred = 0.5 * cf + 0.5 * cb
        return np.clip(pred, 1, 5)

    def get_recommendations(self, user_id, n=10, method='weighted_avg'):
        """추천 생성"""
        watched = set(self.train[self.train['userId'] == user_id]['movieId'])
        predictions = []

        for movie_id in self.movies['movieId']:
            if not hasattr(self, 'item_to_idx') or self.item_to_idx is None:
                continue
            
            if movie_id not in self.item_to_idx:
                continue

            if movie_id not in watched:
                if method.lower() == 'cf':
                    pred = self.predict_cf(user_id, movie_id)
                elif method.lower() == 'cb':
                    pred = self.predict_cb(user_id, movie_id)
                elif method.lower() == 'weighted_avg':
                    pred = self.predict_weighted_avg(user_id, movie_id)
                elif method.lower() == 'feature_combo':
                    pred = self.predict_feature_combo(user_id, movie_id)
                elif method.lower() == 'mixed':
                    pred = self.predict_mixed(user_id, movie_id)
                else:
                    pred = self.predict_weighted_avg(user_id, movie_id)

                if 1 <= pred <= 5:
                    predictions.append((movie_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in predictions[:n]]

    def evaluate(self):
        """5가지 알고리즘 평가"""
        print(f"\n📈 {self.name} 평가 중...")

        results = {}
        methods = {
            'CF': 'cf',
            'CB': 'cb',
            'Weighted_Avg': 'weighted_avg',
            'Feature_Combo': 'feature_combo',
            'Mixed': 'mixed'
        }

        for method_display, method_lower in methods.items():
            print(f" 📊 {method_display} 평가 중...")

            # 기본 정확도 지표
            actuals = []
            preds = []

            for _, row in self.test.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                actual = row['rating']

                if method_lower == 'cf':
                    pred = self.predict_cf(user_id, movie_id)
                elif method_lower == 'cb':
                    pred = self.predict_cb(user_id, movie_id)
                elif method_lower == 'weighted_avg':
                    pred = self.predict_weighted_avg(user_id, movie_id)
                elif method_lower == 'feature_combo':
                    pred = self.predict_feature_combo(user_id, movie_id)
                elif method_lower == 'mixed':
                    pred = self.predict_mixed(user_id, movie_id)
                else:
                    pred = self.predict_weighted_avg(user_id, movie_id)

                actuals.append(actual)
                preds.append(pred)

            # Sparsity-Aware Score
            num_users = len(self.um.index)
            num_items = len(self.um.columns)
            sparsity_result = self.metrics.sparsity_aware_score(
                test_data=np.array(actuals),
                predictions=np.array(preds),
                num_users=num_users,
                num_items=num_items
            )

            # 추천 지표
            test_users = list(self.test['userId'].unique())
            precisions = []
            recalls = []
            f1s = []
            maps = []
            mrrs = []
            ndcgs = []

            for user_id in test_users:
                user_test = self.test[self.test['userId'] == user_id]
                if len(user_test) < 1:
                    continue

                recs = self.get_recommendations(user_id, n=10, method=method_lower)
                relevant = user_test[user_test['rating'] >= 4]['movieId'].tolist()

                if recs and relevant:
                    precisions.append(self.metrics.precision_at_k(recs, relevant, 10))
                    recalls.append(self.metrics.recall_at_k(recs, relevant, 10))
                    f1s.append(self.metrics.f1_at_k(recs, relevant, 10))
                    maps.append(self.metrics.map_at_k(recs, relevant, 10))
                    mrrs.append(self.metrics.mrr_at_k(recs, relevant, 10))
                    relevance = [1 if m in relevant else 0 for m in recs]
                    ndcgs.append(self.metrics.ndcg_at_k(relevance, 10))

            # 다양성 지표
            all_recs = []
            for user_id in test_users:
                recs = self.get_recommendations(user_id, n=10, method=method_lower)
                if recs:
                    all_recs.append(recs)

            diversities = []
            for recs in all_recs:
                if len(recs) >= 2:
                    div = self.metrics.intra_list_diversity(
                        recs,
                        self.item_similarity,
                        self.item_to_idx
                    )
                    if div > 0:
                        diversities.append(div)
            
            diversity = np.mean(diversities) if diversities else 0.0
            coverage = self.metrics.coverage(all_recs, len(self.movies))

            all_recommended_items = []
            for recs in all_recs:
                all_recommended_items.extend(recs)

            novelty = self.metrics.novelty(all_recommended_items, self.popularity)
            pop_bias = self.metrics.popularity_bias(all_recommended_items, self.popularity)

            # 결과 저장
            results[method_display] = {
                'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
                'MAE': mean_absolute_error(actuals, preds),
                'Sparsity': sparsity_result['Sparsity'],
                'Adjusted_RMSE': sparsity_result['Adjusted_RMSE'],
                'Adjusted_MAE': sparsity_result['Adjusted_MAE'],
                'Precision@10': np.mean(precisions) if precisions else 0,
                'Recall@10': np.mean(recalls) if recalls else 0,
                'F1@10': np.mean(f1s) if f1s else 0,
                'MAP@10': np.mean(maps) if maps else 0,
                'MRR@10': np.mean(mrrs) if mrrs else 0,
                'NDCG@10': np.mean(ndcgs) if ndcgs else 0,
                'Diversity': diversity,
                'Coverage': coverage,
                'Novelty': novelty,
                'PopularityBias': pop_bias,
                'Num_Samples': len(precisions)
            }

            print(f" ✅ {method_display}: RMSE={results[method_display]['RMSE']:.4f}")

        return results


# =====================================================================================================
# 섹션 4: 메인 실행
# =====================================================================================================

def main():
    """MovieLens Small/1M 데이터셋에서 5가지 알고리즘 평가"""
    datasets_to_test = ['Small', '1M']
    all_results = []

    for dataset in datasets_to_test:
        print(f"\n{'='*100}")
        print(f"📊 {dataset} 데이터셋 처리 중...")
        print(f"{'='*100}")

        ratings, movies = load_movielens(dataset)
        if ratings is None or movies is None:
            print(f"❌ {dataset} 데이터셋 로드 실패, 건너뜀")
            continue

        model = OptimizedHybridRecommender(ratings, movies, name=dataset, svd_dim=200)
        model._prepare()
        results = model.evaluate()

        for method, metrics in results.items():
            metrics['Dataset'] = dataset
            metrics['Method'] = method
            all_results.append(metrics)

    # 결과 출력
    if all_results:
        print(f"\n{'='*100}")
        print("📈 최종 결과 (5가지 알고리즘 × 2 데이터셋)")
        print(f"{'='*100}")

        results_df = pd.DataFrame(all_results)

        print("\n✅ 기본 정확도 지표:")
        print(results_df[['Dataset', 'Method', 'RMSE', 'MAE', 'Sparsity', 'Adjusted_RMSE']].to_string(index=False))

        print("\n✅ 추천 정확성 지표 (Ranking Metrics):")
        print(results_df[['Dataset', 'Method', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']].to_string(index=False))

        print("\n✅ 순위 기반 지표:")
        print(results_df[['Dataset', 'Method', 'MAP@10', 'MRR@10', 'NDCG@10']].to_string(index=False))

        print("\n✅ 다양성 및 품질 지표:")
        print(results_df[['Dataset', 'Method', 'Diversity', 'Coverage', 'Novelty', 'PopularityBias']].to_string(index=False))

        # CSV 저장
        output_filename = 'hybrid_results_small_1m_ver9_corrected.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\n✅ 결과 저장: {output_filename}")

        # 데이터셋별 최고 성능
        print("\n📊 데이터셋별 비교:")
        print("-" * 100)

        for dataset in datasets_to_test:
            dataset_results = results_df[results_df['Dataset'] == dataset]
            if len(dataset_results) > 0:
                best_method = dataset_results.loc[dataset_results['RMSE'].idxmin(), 'Method']
                best_rmse = dataset_results['RMSE'].min()
                best_adj_rmse = dataset_results['Adjusted_RMSE'].min()
                best_precision = dataset_results['Precision@10'].max()
                best_novelty = dataset_results['Novelty'].max()

                print(f"\n{dataset}:")
                print(f" 최고 성능 알고리즘 (RMSE): {best_method}")
                print(f" 최저 RMSE: {best_rmse:.4f}")
                print(f" 최저 Adjusted_RMSE: {best_adj_rmse:.4f}")
                print(f" 최고 Precision@10: {best_precision:.4f}")
                print(f" 최고 Novelty: {best_novelty:.4f}")
    else:
        print("❌ 처리된 데이터셋이 없습니다")


if __name__ == "__main__":
    main()
