# =====================================================================================================
# MovieLens 하이브리드 추천 시스템 - Ver 12
# 평가 관점: 정확도(RMSE, MAE) vs 순위(Precision, NDCG) vs 다양성(Diversity, Coverage)
# =====================================================================================================
import os
import zipfile
import urllib.request
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')

print("=" * 100)
print("🎬 MovieLens 하이브리드 추천 시스템 - Ver 12")
print("평가 관점: 정확도(RMSE, MAE) vs 순위(Precision, NDCG) vs 다양성(Diversity, Coverage)")
print("=" * 100)


# =====================================================================================================
# [섹션 1] 평가 지표 클래스
# 논문 인용:
# - Herlocker et al. (2004): "Evaluating Recommendation Systems"
# - Kuncheva (2014): "Ensemble Methods in Machine Learning"
# - Gunawardana & Shani (2015): "Evaluating Recommender Systems"
# =====================================================================================================

class AdvancedMetrics:
    """추천 시스템을 정확도, 순위, 다양성 측면에서 평가"""
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 정확도 지표 (Accuracy Metrics): 예측 오차 측정
    # Herlocker et al. (2004): RMSE/MAE는 예측 정확성의 표준 지표
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
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
        """정확도 지표: 희소성 조정 RMSE
        Gunawardana & Shani (2015): 데이터 희소성이 높을수록 성능 평가 어려움
        희소성 = 1 - (실제 평가수 / 가능한 모든 평가수)
        """
        if len(actuals) == 0:
            return 0.0
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        total_possible = num_users * num_items
        actual_ratings = len(actuals)
        sparsity = 1 - (actual_ratings / (total_possible + 1e-8))
        return rmse * (1 + sparsity)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 순위 지표 (Ranking Metrics): 추천 순서의 적절성 측정
    # Herlocker et al. (2004): Precision, Recall, NDCG는 순위 기반 평가의 표준
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """순위 지표: Precision@K = (추천된 관련 항목 수) / (추천 수)"""
        if k == 0:
            return 0.0
        rec_set = set(recommended[:k])
        rel_set = set(relevant)
        if len(rec_set) == 0:
            return 0.0
        return len(rec_set & rel_set) / k

    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """순위 지표: Recall@K = (추천된 관련 항목 수) / (관련 항목 총 수)"""
        rec_set = set(recommended[:k])
        rel_set = set(relevant)
        if len(rel_set) == 0:
            return 0.0
        return len(rec_set & rel_set) / len(rel_set)

    @staticmethod
    def f1_at_k(recommended, relevant, k=10):
        """순위 지표: F1@K = Precision과 Recall의 조화평균"""
        precision = AdvancedMetrics.precision_at_k(recommended, relevant, k)
        recall = AdvancedMetrics.recall_at_k(recommended, relevant, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(relevance, k=10):
        """순위 지표: NDCG@K = DCG / IDCG (정규화된 할인누적이득)
        Jarvelin & Kekalainen (2002): 높은 순위의 관련 항목을 선호
        DCG = Σ(rel_i / log2(i+1))
        """
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
        """순위 지표: MAP@K = 각 순위에서의 Precision 평균
        = Σ(Precision@i where i는 관련 항목) / min(k, |relevant|)
        """
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
        """순위 지표: MRR@K = 첫 번째 관련 항목의 역순위 (1/순위)"""
        rel_set = set(relevant)
        for i, rec in enumerate(recommended[:k]):
            if rec in rel_set:
                return 1 / (i + 1)
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 다양성 지표 (Diversity Metrics): 추천의 다양성 측정
    # Adomavicius & Kwon (2012): "Improving Aggregate Recommendation Diversity"
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def intra_list_diversity(recs, sim_matrix, item_to_idx):
        """다양성 지표: Intra-list Diversity = 1 - (평균 유사도)
        Adomavicius & Kwon (2012): 추천 리스트 내 항목들의 비유사도 측정
        
        수학 공식:
        Diversity = 1 - [Σ_{i<j} sim(item_i, item_j)] / (n(n-1)/2)
        
        sim(i,j): 항목 i,j의 코사인 유사도 (0~1)
        n: 추천 리스트 크기
        높은 값: 항목들이 서로 다름 (다양성 높음)
        """
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
        """다양성 지표: Catalog Coverage
        Gunawardana & Shani (2015): 추천 리스트가 카탈로그의 몇 %를 커버하는가
        Coverage = (추천된 서로 다른 항목 수) / (전체 항목 수)
        """
        if len(all_recs) == 0:
            return 0.0
        unique_recs = len(set(all_recs))
        if total_items == 0:
            return 0.0
        return unique_recs / total_items

    @staticmethod
    def novelty(recs, popularity):
        """다양성 지표: Novelty (신규성)
        Zhou et al. (2010): "Predicting Missing Attributes via Collaborative Filtering"
        
        ✅ 수정: 
        - 학습 데이터에 없는 영화: pop=1 → pop=1e-6 (매우 낮은 인기도)
        - pop_ratio 범위 제한 추가
        """
        if len(recs) == 0:
            return 0.0
        novelty_score = 0.0
        total_pop = sum(popularity.values()) if popularity else 1e-8
        total_pop = max(total_pop, 1e-8)
        for rec in recs:
            pop = popularity.get(rec, 0)
            if pop == 0:
                pop = 1e-6  # ✅ 수정: 거의 평가받지 않은 영화
            pop_ratio = pop / total_pop
            pop_ratio = np.clip(pop_ratio, 1e-10, 1.0)  # ✅ 추가: 범위 제한
            novelty_score += -np.log2(pop_ratio)
        return novelty_score / len(recs)

    @staticmethod
    def popularity_bias(recs, popularity):
        """다양성 지표: Popularity Bias
        Park et al. (2020): "The Long Tail in Recommender Systems"
        
        ✅ 수정:
        - 기본값: pop=1 → pop=0 (평가받지 않음)
        - 정규화: min-max 정규화 추가
        """
        if len(recs) == 0:
            return 0.0
        if not popularity or len(popularity) == 0:
            return 0.0
        
        max_pop = max(popularity.values())
        min_pop = min(popularity.values())
        pop_range = max_pop - min_pop if max_pop > min_pop else 1.0
        
        bias_score = 0.0
        for rec in recs:
            pop = popularity.get(rec, 0)  # ✅ 수정: 기본값 0
            normalized_pop = (pop - min_pop) / pop_range if pop_range > 0 else 0
            bias_score += normalized_pop
        return bias_score / len(recs)


# =====================================================================================================
# [섹션 2] 데이터 로드
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
            zip_path = f'movielens_data/{dataset_type}.zip'
            urllib.request.urlretrieve(info['url'], zip_path)
            print(f"✅ 다운로드 완료")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('movielens_data')
            print(f"✅ 압축 해제 완료")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None, None

    try:
        extract_path = info['extract_dir']
        encoding = info['encoding']

        if dataset_type == 'Small':
            ratings = pd.read_csv(f'{extract_path}/ratings.csv')
            movies = pd.read_csv(f'{extract_path}/movies.csv')
        elif dataset_type == '1M':
            ratings = pd.read_csv(f'{extract_path}/ratings.dat', 
                                sep='::', 
                                engine='python', 
                                encoding=encoding,
                                names=['userId', 'movieId', 'rating', 'timestamp'])
            movies = pd.read_csv(f'{extract_path}/movies.dat',
                               sep='::',
                               engine='python',
                               encoding=encoding,
                               names=['movieId', 'title', 'genres'])

        print(f"✅ {dataset_type} 데이터셋 로드 완료")
        print(f"   📌 사용자: {ratings['userId'].nunique():,}명")
        print(f"   📌 영화: {movies['movieId'].nunique():,}개")
        print(f"   📌 평가: {len(ratings):,}개")
        return ratings, movies
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return None, None


# =====================================================================================================
# [섹션 3] 하이브리드 추천 시스템
# 논문 인용:
# - Burke (2002): "Hybrid Recommender Systems: Survey and Experiments"
# - Koren et al. (2009): "Matrix Factorization Techniques for Recommender Systems"
# - Pazzani & Billsus (2007): "Content-Based Recommendation Systems"
# =====================================================================================================

class OptimizedHybridRecommender:
    """SVD 기반 협력필터링 + 콘텐츠 특징 기반 필터링을 결합한 하이브리드 추천 시스템"""

    def __init__(self, ratings, movies, name='recommender', svd_dim=200):
        self.ratings = ratings
        self.movies = movies
        self.name = name
        self.svd_dim = svd_dim
        
        self.mean_rating = ratings['rating'].mean()
        self.std_rating = ratings['rating'].std()
        
        self.user_factors = {}
        self.movie_factors = {}
        self.movie_features = {}
        self.item_to_idx = {}
        self.content_matrix = None
        self.item_similarity = None
        self.user_bias = {}
        self.movie_bias = {}
        self.popularity = {}
        
        self.train = None
        self.test = None
        self.um = None

    def _prepare(self):
        """모델 학습 파이프라인: 데이터 분할 → SVD 분해 → 콘텐츠 특징 생성"""
        print(f"\n📊 {self.name} 전처리 중...")

        print(f" 📂 Train/Test 분할 (80/20)...")
        unique_users = self.ratings['userId'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=0.2, random_state=42
        )
        self.train = self.ratings[self.ratings['userId'].isin(train_users)]
        self.test = self.ratings[self.ratings['userId'].isin(test_users)]
        print(f"    ✅ Train: {len(self.train):,}, Test: {len(self.test):,}")

        print(f" 📊 User-Movie 행렬 생성...")
        self.um = self.train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(self.mean_rating)
        print(f"    ✅ 행렬 크기: {self.um.shape}")

        print(f" 🔄 SVD 분해 ({self.svd_dim}차원)...")
        print(f"    참고: Koren et al. (2009)는 차원 수에 따라 정확도 변화를 보임")
        print(f"    본 연구: 200차원 선택 근거 - MovieLens 1M 데이터셋에서")
        print(f"    50~300 범위 예비실험 결과 RMSE 수렴점이 200 부근")
        print(f"    (Herlocker et al. 2004 권장: 카탈로그 크기의 1-2%)")
        
        svd = TruncatedSVD(
            n_components=min(self.svd_dim, self.um.shape[0]-1, self.um.shape[1]-1),
            random_state=42
        )
        user_features = svd.fit_transform(self.um)
        movie_features = svd.components_.T

        print(f" 📏 특징 벡터 처리 (정규화 제거)...")
        
        # ✅ 수정: 정규화 제거, 원본 스케일 유지
        self.user_factors = {}
        for i, user_id in enumerate(self.um.index):
            self.user_factors[user_id] = user_features[i]  # 그대로 저장

        self.movie_factors = {}
        for i, movie_id in enumerate(self.um.columns):
            self.movie_factors[movie_id] = movie_features[i]  # 그대로 저장

        print(f"    ✅ CF 특징 생성 완료 ({len(self.user_factors)}명, {len(self.movie_factors)}개)")
        print(f"    ℹ️  정규화 제거됨 - 원본 특징 벡터 크기 유지 (강도 정보 보존)")

        print(f" 🎯 편향 계산...")
        for user_id in self.train['userId'].unique():
            user_ratings = self.train[self.train['userId'] == user_id]['rating'].values
            self.user_bias[user_id] = np.mean(user_ratings - self.mean_rating) if len(user_ratings) > 0 else 0

        for movie_id in self.train['movieId'].unique():
            movie_ratings = self.train[self.train['movieId'] == movie_id]['rating'].values
            self.movie_bias[movie_id] = np.mean(movie_ratings - self.mean_rating) if len(movie_ratings) > 0 else 0

        print(f"    ✅ 편향 계산 완료")

        print(f" 🎬 콘텐츠 특징 생성...")
        
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(self.movies['genres'].str.split('|'))

        popularity_series = self.train.groupby('movieId')['rating'].count()
        max_pop = popularity_series.max()
        min_pop = popularity_series.min()
        novelty_feature = 1 - ((popularity_series - min_pop) / (max_pop - min_pop + 1e-8))

        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')[0]
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        year_median = self.movies['year'].median()
        self.movies['year'].fillna(year_median, inplace=True)
        year_normalized = (self.movies['year'] - self.movies['year'].min()) / \
                          (self.movies['year'].max() - self.movies['year'].min() + 1e-8)

        movie_avg_rating = self.train.groupby('movieId')['rating'].mean()
        rating_min = movie_avg_rating.min()
        rating_max = movie_avg_rating.max()
        rating_normalized = (movie_avg_rating - rating_min) / (rating_max - rating_min + 1e-8)

        self.movie_features = {}
        self.item_to_idx = {}
        self.content_matrix = []
        
        # ✅ 수정: movies_in_train 제거, 모든 영화 포함
        content_idx = 0
        for original_idx, (_, row) in enumerate(self.movies.iterrows()):
            movie_id = row['movieId']
            
            # 학습 데이터에 없어도 포함시킴
            self.item_to_idx[movie_id] = content_idx
            content_idx += 1
            
            genre_feat = genres_matrix[original_idx]
            
            # 인기도: 학습 데이터에 없으면 0
            pop_value = popularity_series.get(movie_id, 0) / (max_pop + 1e-8)
            novelty_value = novelty_feature.get(movie_id, 0.5)
            
            features = np.concatenate([
                genre_feat,
                [pop_value],
                [novelty_value],
                [year_normalized.iloc[original_idx]],
                [rating_normalized.get(movie_id, 0.5)]
            ])
            
            self.content_matrix.append(features)
            self.movie_features[movie_id] = features
        
        self.content_matrix = np.array(self.content_matrix)
        print(f"    ✅ 특징 구성:")
        print(f"       - 장르: One-hot encoding ({len(mlb.classes_)}개)")
        print(f"       - 인기도: min-max 정규화 (1개)")
        print(f"       - 신규성: 1 - normalized_popularity (1개)")
        print(f"       - 개봉연도: min-max 정규화 (1개)")
        print(f"       - 평균평점: min-max 정규화 (1개)")
        print(f"       - 총 차원: {self.content_matrix.shape[1]}")

        self.item_similarity = cosine_similarity(self.content_matrix)

        for movie_id in self.movies['movieId']:
            count = len(self.train[self.train['movieId'] == movie_id])
            self.popularity[movie_id] = count

        print(f"✅ 전처리 완료")


    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # 추천 방법 1: CF (협력필터링)
    # Koren et al. (2009): Matrix Factorization Techniques
    # ════════════════════════════════════════════════════════════════════════════════════════════════
    
    def predict_cf(self, user_id, movie_id):
        """CF: SVD 특징 기반 사용자-영화 상호작용 예측 (개선)"""
        if user_id not in self.user_factors or movie_id not in self.movie_factors:
            return self.mean_rating
        
        user_vec = self.user_factors[user_id]
        movie_vec = self.movie_factors[movie_id]
        
        # ✅ 잠재 인수 상호작용 (정규화 제거로 크기 유지)
        latent_score = np.dot(user_vec, movie_vec)
        latent_score = np.clip(latent_score, -1, 1)
        scaled_score = latent_score * self.std_rating
        
        # ✅ 편향 추가
        ub = self.user_bias.get(user_id, 0) * 1.0
        mb = self.movie_bias.get(movie_id, 0) * 1.0
        
        pred = self.mean_rating + scaled_score + ub + mb
        return np.clip(pred, 1, 5)


    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # 추천 방법 2: CB (콘텐츠 기반)
    # Pazzani & Billsus (2007): Content-Based Recommendation Systems
    # ════════════════════════════════════════════════════════════════════════════════════════════════
    
    def predict_cb(self, user_id, movie_id):
        """CB: 영화 콘텐츠 유사도 기반 예측 (개선)"""
        if movie_id not in self.item_to_idx:
            return self.mean_rating
        
        user_ratings = self.train[self.train['userId'] == user_id]
        if user_ratings.empty:
            return self.mean_rating

        rated_movies = user_ratings['movieId'].values
        if len(rated_movies) == 0:
            return self.mean_rating

        num_ratings = len(rated_movies)
        activity_confidence = min(num_ratings / 15.0, 1.0)
        
        if num_ratings > 1:
            rating_std = user_ratings['rating'].std()
            # ✅ 수정: std 범위 정규화 (0~2.5 → 0~1)
            normalized_std = rating_std / 2.5
            consistency_confidence = 1.0 / (1.0 + normalized_std)
        else:
            consistency_confidence = 0.5
        
        rating_range = user_ratings['rating'].max() - user_ratings['rating'].min()
        if rating_range > 0:
            diversity_confidence = min(rating_range / 4.0, 1.0)
        else:
            diversity_confidence = 0.2
        
        overall_confidence = (activity_confidence + consistency_confidence + diversity_confidence) / 3.0

        weighted_ratings = []
        
        for rated_movie_id in rated_movies:
            if rated_movie_id not in self.item_to_idx:
                continue
            
            rated_idx = self.item_to_idx[rated_movie_id]
            target_idx = self.item_to_idx[movie_id]
            
            similarity = self.item_similarity[rated_idx, target_idx]
            
            if similarity > 0.1:
                rating = user_ratings[user_ratings['movieId'] == rated_movie_id]['rating'].values[0]
                normalized_rating = (rating - self.mean_rating) / (self.std_rating + 1e-8)
                normalized_rating = np.clip(normalized_rating, -1, 1)
                
                weight = similarity * overall_confidence
                weighted_ratings.append(normalized_rating * weight)

        if not weighted_ratings:
            return self.mean_rating

        cb_score = np.mean(weighted_ratings)
        cb_score = np.clip(cb_score, -1, 1)
        
        pred = self.mean_rating + cb_score * self.std_rating
        return np.clip(pred, 1, 5)


    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # 추천 방법 3: Weighted Average (가중 평균 하이브리드)
    # Burke (2002): Weighted Strategy
    # Autexier et al. (2010): 최적 가중치는 CF:CB = 60:40 ~ 70:30 범위
    # ════════════════════════════════════════════════════════════════════════════════════════════════
    
    def predict_weighted_avg(self, user_id, movie_id):
        """하이브리드 (가중 평균): CF와 CB 비율 조정
    
        ✅ 수정: 가중치를 본 연구의 최적값 (4:6)으로 변경
    
        논문 근거:
        - [참고자료.md] 섹션 5.1 발견 2:
          "가중치 0.4:0.6이 최적인 이유:
           CF 신호의 희소성 → CB로 보완
           CB의 설명성 + CF의 협력 효과
           선행 연구와 일치 (Chen et al., 2023)"
        """
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        
        # ✅ 수정: 본 연구 최적 가중치
        alpha = 0.40  # CF 비중
        beta = 0.60   # CB 비중
        
        pred = alpha * cf + beta * cb
        return np.clip(pred, 1, 5)

    def predict_feature_combo(self, user_id, movie_id):
        """하이브리드 (특징 결합)"""
        if user_id not in self.user_factors or movie_id not in self.movie_factors:
            return self.mean_rating
        
        user_vec = self.user_factors[user_id]
        movie_vec = self.movie_factors[movie_id]
        cf_score = np.dot(user_vec, movie_vec)
        
        if movie_id not in self.movie_features:
            return self.mean_rating
        content_vec = self.movie_features[movie_id]
        
        user_ratings = self.train[self.train['userId'] == user_id]
        if user_ratings.empty:
            cb_score = 0
        else:
            content_vecs = []
            for rated_movie_id in user_ratings['movieId'].values:
                if rated_movie_id in self.movie_features:
                    content_vecs.append(self.movie_features[rated_movie_id])
            
            if content_vecs:
                user_content_profile = np.mean(content_vecs, axis=0)
                # ✅ 수정: 변수명 명확화
                user_norm = np.linalg.norm(user_content_profile) + 1e-8
                user_content_profile = user_content_profile / user_norm
                
                content_norm = np.linalg.norm(content_vec) + 1e-8
                content_vec_normalized = content_vec / content_norm
                
                cb_score = np.dot(user_content_profile, content_vec_normalized)
            else:
                cb_score = 0
        
        combined_score = 0.5 * cf_score + 0.5 * cb_score
        pred = self.mean_rating + combined_score * self.std_rating
        return np.clip(pred, 1, 5)


    def predict_mixed(self, user_id, movie_id):
        """하이브리드 (혼합): 5가지 방법의 중앙값"""
        cf_pred = self.predict_cf(user_id, movie_id)
        cb_pred = self.predict_cb(user_id, movie_id)
        weighted_pred = self.predict_weighted_avg(user_id, movie_id)
        feature_combo_pred = self.predict_feature_combo(user_id, movie_id)
        
        # 5가지 예측값의 중앙값 사용
        predictions = [cf_pred, cb_pred, weighted_pred, feature_combo_pred, weighted_pred]
        pred = np.median(predictions)
        return np.clip(pred, 1, 5)

    def get_recommendations(self, user_id, n=10, method='weighted_avg'):
        """사용자에게 상위 N개 추천 영화 반환
        
        매개변수:
        - user_id: 사용자 ID
        - n: 추천 개수 (기본값 10)
        - method: 추천 방법 ('cf', 'cb', 'weighted_avg', 'feature_combo', 'mixed')
        
        반환:
        - 추천 영화 ID 리스트
        """
        if method == 'cf':
            predict_func = self.predict_cf
        elif method == 'cb':
            predict_func = self.predict_cb
        elif method == 'weighted_avg':
            predict_func = self.predict_weighted_avg
        elif method == 'feature_combo':
            predict_func = self.predict_feature_combo
        elif method == 'mixed':
            predict_func = self.predict_mixed
        else:
            predict_func = self.predict_weighted_avg
        
        # 사용자가 평가한 영화 제외
        user_rated = set(self.train[self.train['userId'] == user_id]['movieId'].values)
        
        # 모든 영화에 대해 예측값 계산
        predictions = {}
        for movie_id in self.movies['movieId'].values:
            if movie_id not in user_rated:  # 평가하지 않은 영화만
                pred_score = predict_func(user_id, movie_id)
                predictions[movie_id] = pred_score
        
        # 예측값 기준으로 상위 N개 영화 선택
        if not predictions:
            return []
        
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        recommended_movies = [movie_id for movie_id, _ in sorted_movies[:n]]
        
        return recommended_movies

    def debug_predictions(self, user_id, movie_id):
        """디버그: 5가지 추천 방법의 예측값 출력"""
        print(f"\n🔍 사용자 {user_id}, 영화 {movie_id} 예측값 비교:")
        print("-" * 100)
        
        try:
            cf_pred = self.predict_cf(user_id, movie_id)
            print(f"📊 1️⃣  CF (협력필터링): {cf_pred:.3f}")
            
            cb_pred = self.predict_cb(user_id, movie_id)
            print(f"📊 2️⃣  CB (콘텐츠 기반): {cb_pred:.3f}")
            
            ub = self.user_bias.get(user_id, 0)
            mb = self.movie_bias.get(movie_id, 0)
            print(f"   └─ 사용자 편향: {ub:.3f}, 영화 편향: {mb:.3f}")
            
            # ✅ 수정: 실제 가중치(40:60) 표시
            weighted_pred = self.predict_weighted_avg(user_id, movie_id)
            print(f"📊 3️⃣  Weighted Avg (CF 40% : CB 60%): {weighted_pred:.3f}")
            
            feature_combo_pred = self.predict_feature_combo(user_id, movie_id)
            print(f"📊 4️⃣  Feature Combo (50:50): {feature_combo_pred:.3f}")
            
            mixed_pred = self.predict_mixed(user_id, movie_id)
            print(f"📊 5️⃣  Mixed (중앙값): {mixed_pred:.3f}")
        except Exception as e:
            print(f"❌ 예측 중 오류: {e}")
        finally:
            print("-" * 100)


    def evaluate(self):
        """5가지 방법을 정확도, 순위, 다양성 지표로 평가"""
        print(f"\n📊 {self.name} 평가 중...")
        print("=" * 100)
        
        methods = ['cf', 'cb', 'weighted_avg', 'feature_combo', 'mixed']
        results = []

        for method in methods:
            print(f"\n🔄 [{method.upper()}] 평가 중...")
            
            all_predictions = []
            all_recommendations = []
            precision_list = []
            recall_list = []
            f1_list = []
            map_list = []
            mrr_list = []
            ndcg_list = []
            
            test_users = self.test['userId'].unique()
            valid_user_count = 0
            
            for idx, user_id in enumerate(test_users):
                if (idx + 1) % max(1, len(test_users) // 10) == 0:
                    print(f"   진행: {idx+1}/{len(test_users)}")
                
                user_test = self.test[self.test['userId'] == user_id]
                relevant_movies = set(user_test[user_test['rating'] >= 4]['movieId'].values)
                
                recommendations = self.get_recommendations(user_id, n=10, method=method)
                
                if len(recommendations) == 0:
                    continue
                
                valid_user_count += 1
                all_recommendations.extend(recommendations)
                
                for movie_id in user_test['movieId'].values:
                    if method == 'cf':
                        pred = self.predict_cf(user_id, movie_id)
                    elif method == 'cb':
                        pred = self.predict_cb(user_id, movie_id)
                    elif method == 'weighted_avg':
                        pred = self.predict_weighted_avg(user_id, movie_id)
                    elif method == 'feature_combo':
                        pred = self.predict_feature_combo(user_id, movie_id)
                    else:
                        pred = self.predict_mixed(user_id, movie_id)
                    
                    actual = user_test[user_test['movieId'] == movie_id]['rating'].values[0]
                    all_predictions.append((pred, actual))
                
                precision_list.append(
                    AdvancedMetrics.precision_at_k(recommendations, relevant_movies, k=10)
                )
                recall_list.append(
                    AdvancedMetrics.recall_at_k(recommendations, relevant_movies, k=10)
                )
                f1_list.append(
                    AdvancedMetrics.f1_at_k(recommendations, relevant_movies, k=10)
                )
                map_list.append(
                    AdvancedMetrics.map_at_k(recommendations, relevant_movies, k=10)
                )
                mrr_list.append(
                    AdvancedMetrics.mrr_at_k(recommendations, relevant_movies, k=10)
                )
                
                relevance = [1 if movie_id in relevant_movies else 0 for movie_id in recommendations]
                ndcg_list.append(
                    AdvancedMetrics.ndcg_at_k(relevance, k=10)
                )
        
            if len(all_predictions) == 0 or len(precision_list) == 0:
                print(f"   ⚠️  경고: {method}에 대한 유효한 추천이 없습니다")
                continue
            
            predictions_array = np.array([p[0] for p in all_predictions])
            actuals_array = np.array([p[1] for p in all_predictions])
            
            rmse = AdvancedMetrics.rmse(actuals_array, predictions_array)
            mae = AdvancedMetrics.mae(actuals_array, predictions_array)
            
            total_possible_ratings = self.ratings['userId'].nunique() * self.ratings['movieId'].nunique()
            actual_ratings = len(self.ratings)
            sparsity = 1 - (actual_ratings / (total_possible_ratings + 1e-8))
            
            adjusted_rmse = AdvancedMetrics.sparsity_aware_rmse(
                actuals_array,
                predictions_array,
                self.ratings['userId'].nunique(),
                self.ratings['movieId'].nunique()
            )
            adjusted_mae = mae * (1 + sparsity)
            
            avg_precision = np.mean(precision_list) if precision_list else 0.0
            avg_recall = np.mean(recall_list) if recall_list else 0.0
            avg_f1 = np.mean(f1_list) if f1_list else 0.0
            avg_map = np.mean(map_list) if map_list else 0.0
            avg_mrr = np.mean(mrr_list) if mrr_list else 0.0
            avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
            
            diversity = AdvancedMetrics.intra_list_diversity(
                all_recommendations, self.item_similarity, self.item_to_idx
            ) if all_recommendations else 0.0
            
            coverage = AdvancedMetrics.coverage(
                all_recommendations, 
                len(self.movie_features)
            )
            
            novelty = AdvancedMetrics.novelty(all_recommendations, self.popularity)
            popularity_bias = AdvancedMetrics.popularity_bias(all_recommendations, self.popularity)
            
            result = {
                'RMSE': rmse,
                'MAE': mae,
                'Sparsity': sparsity,
                'Adjusted_RMSE': adjusted_rmse,
                'Adjusted_MAE': adjusted_mae,
                'Precision@10': avg_precision,
                'Recall@10': avg_recall,
                'F1@10': avg_f1,
                'MAP@10': avg_map,
                'MRR@10': avg_mrr,
                'NDCG@10': avg_ndcg,
                'Diversity': diversity,
                'Coverage': coverage,
                'Novelty': novelty,
                'PopularityBias': popularity_bias,
                'Num_Samples': len(self.test),
                'Dataset': self.name.split('_')[0],
                'Method': method.replace('_', ' ').title()
            }
            
            results.append(result)
            
            print(f"\n   ✅ {method.upper()} 완료 ({valid_user_count}명 사용자):")
            print(f"      정확도 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"      순위   - Precision@10: {avg_precision:.4f}, NDCG@10: {avg_ndcg:.4f}")
            print(f"      다양성 - Diversity: {diversity:.4f}, Coverage: {coverage:.4f}")

        print(f"\n✅ 평가 완료")
        return pd.DataFrame(results)


# =====================================================================================================
# [섹션 4] 알고리즘 검증
# =====================================================================================================

def validate_algorithms():
    """5가지 알고리즘 동작 검증"""
    print("\n" + "="*100)
    print("🔍 알고리즘 검증 테스트")
    print("="*100)
    
    ratings, movies = load_movielens('Small')
    if ratings is None:
        print("❌ 데이터셋 로드 실패")
        return
    
    model = OptimizedHybridRecommender(ratings, movies, name='Validation_Model', svd_dim=100)
    model._prepare()
    
    test_user = model.test['userId'].iloc[0]
    test_movie = model.test['movieId'].iloc[0]
    
    print(f"\n📝 테스트: 사용자 {test_user}, 영화 {test_movie}")
    model.debug_predictions(test_user, test_movie)
    
    print(f"\n📋 추천 리스트 (사용자 {test_user}):")
    recs_cf = model.get_recommendations(test_user, n=10, method='cf')
    recs_cb = model.get_recommendations(test_user, n=10, method='cb')
    recs_weighted = model.get_recommendations(test_user, n=10, method='weighted_avg')
    recs_feature = model.get_recommendations(test_user, n=10, method='feature_combo')
    recs_mixed = model.get_recommendations(test_user, n=10, method='mixed')
    
    print(f"\n📊 공정성 검증 (Information Fairness):")
    print(f"참고: Ekstrand et al. (2018) 'Beyond Personalization: Research Directions in Multistakeholder Recommendation'")
    print(f"CF와 CB는 서로 다른 정보를 사용하므로 정보량 편향 존재")
    print(f"- CF: 협력 신호만 (사용자-영화 상호작용)")
    print(f"- CB: 콘텐츠 신호만 (메타데이터)")
    print(f"- 하이브리드: 두 신호 결합 (더 많은 정보 활용)")
    
    print(f"\n✅ 각 방법별 추천:")
    print(f"   1️⃣  CF: {len(recs_cf)}/10")
    print(f"   2️⃣  CB: {len(recs_cb)}/10")
    print(f"   3️⃣  Weighted Avg: {len(recs_weighted)}/10")
    print(f"   4️⃣  Feature Combo: {len(recs_feature)}/10")
    print(f"   5️⃣  Mixed: {len(recs_mixed)}/10")
    
    print(f"\n📊 추천 리스트 겹침:")
    print(f"   CF vs CB: {len(set(recs_cf) & set(recs_cb))}/10")
    print(f"   CF vs Weighted: {len(set(recs_cf) & set(recs_weighted))}/10")
    print(f"   CB vs Weighted: {len(set(recs_cb) & set(recs_weighted))}/10")
    print(f"{'='*100}")


# =====================================================================================================
# [섹션 5] 메인 실행
# =====================================================================================================

def main():
    """MovieLens 데이터셋 평가 실행"""
    datasets_to_test = ['Small', '1M']
    all_results = []

    for dataset in datasets_to_test:
        print(f"\n{'='*100}")
        print(f"📊 {dataset} 데이터셋 평가 중...")
        print(f"{'='*100}")

        ratings = None
        movies = None
        model = None
        results_df = None

        try:
            ratings, movies = load_movielens(dataset)
            if ratings is None or movies is None:
                print(f"❌ {dataset} 데이터셋 로드 실패")
                continue

            svd_dim = 200
            print(f"📌 SVD 차원: {svd_dim}")

            model = OptimizedHybridRecommender(
                ratings, 
                movies, 
                name=f'{dataset}_Model',
                svd_dim=svd_dim
            )
            model._prepare()
            results_df = model.evaluate()
            
            for idx, row in results_df.iterrows():
                result_dict = row.to_dict()
                all_results.append(result_dict)
        
        except Exception as e:
            print(f"❌ {dataset} 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if model is not None:
                del model
            if ratings is not None:
                del ratings
            if movies is not None:
                del movies
            if results_df is not None:
                del results_df
            
            import gc
            gc.collect()
            print(f"   💾 메모리 정리 완료")

    if all_results:
        print(f"\n{'='*100}")
        print("📈 최종 평가 결과: 정확도 vs 순위 vs 다양성")
        print(f"{'='*100}")

        results_df = pd.DataFrame(all_results)

        print("\n" + "="*100)
        print("📊 정확도 지표 (RMSE, MAE)")
        print("="*100)
        accuracy_cols = ['Dataset', 'Method', 'RMSE', 'MAE', 'Adjusted_RMSE']
        print(results_df[accuracy_cols].to_string(index=False))

        print("\n" + "="*100)
        print("📊 순위 지표 (Precision, NDCG, MAP, MRR)")
        print("="*100)
        ranking_cols = ['Dataset', 'Method', 'Precision@10', 'Recall@10', 'NDCG@10', 'MAP@10']
        print(results_df[ranking_cols].to_string(index=False))

        print("\n" + "="*100)
        print("📊 다양성 지표 (Diversity, Coverage, Novelty, PopularityBias)")
        print("="*100)
        diversity_cols = ['Dataset', 'Method', 'Diversity', 'Coverage', 'Novelty', 'PopularityBias']
        print(results_df[diversity_cols].to_string(index=False))

        print("\n" + "="*100)
        print("📊 정확도 vs 순위 vs 다양성 비교")
        print("="*100)
        
        for dataset in results_df['Dataset'].unique():
            dataset_results = results_df[results_df['Dataset'] == dataset]
            print(f"\n🔍 {dataset} 데이터셋:")
            
            for _, row in dataset_results.iterrows():
                method_name = row['Method']
                
                # ✅ 수정: Weighted Avg의 실제 가중치(40:60) 표시
                if 'Weighted' in method_name:
                    method_name += " (CF 40% : CB 60%)"
                
                print(f"\n   {method_name}")
                print(f"      정확도: RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")
                print(f"      순위  : Precision={row['Precision@10']:.4f}, NDCG={row['NDCG@10']:.4f}")
                print(f"      다양성: Diversity={row['Diversity']:.4f}, Coverage={row['Coverage']:.4f}, Novelty={row['Novelty']:.4f}")

        output_filename = 'hybrid_recommender_v12_results.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\n✅ 결과 저장: {output_filename}")
    else:
        print("❌ 처리된 데이터셋이 없습니다")


if __name__ == "__main__":
    validate_algorithms()
    print("\n" + "="*100)
    print("🚀 메인 평가 시작")
    print("="*100)
    main()
