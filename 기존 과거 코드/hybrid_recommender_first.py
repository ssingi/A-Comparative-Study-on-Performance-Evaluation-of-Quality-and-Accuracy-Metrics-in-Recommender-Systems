# 수정된 하이브리드 추천 시스템 - Ver 7 (UTF-8 인코딩 오류 수정)
# Small (100K) + 1M 데이터셋 모두 지원
# ⭐ 핵심 수정: encoding='iso-8859-1' 또는 'latin-1' 추가

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
print("🎬 MovieLens 하이브리드 추천 시스템 - Small & 1M 데이터셋 지원 (Ver 7 - 인코딩 수정)")
print("=" * 100)

# ================================
# 고급 평가 지표 클래스
# ================================

class AdvancedMetrics:
    """12개 평가 지표 + Sparsity-Aware Score"""
    
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """추천 중 맞은 비율"""
        if k == 0 or not recommended:
            return 0.0
        rec_k = set(recommended[:k])
        rel_set = set(relevant)
        return len(rec_k & rel_set) / k if len(rec_k) > 0 else 0.0
    
    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """찾은 좋은 영화의 비율"""
        if not relevant:
            return 0.0
        rec_k = set(recommended[:k])
        rel_set = set(relevant)
        return len(rec_k & rel_set) / len(rel_set)
    
    @staticmethod
    def f1_at_k(recommended, relevant, k=10):
        """F1 스코어"""
        p = AdvancedMetrics.precision_at_k(recommended, relevant, k)
        r = AdvancedMetrics.recall_at_k(recommended, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @staticmethod
    def ndcg_at_k(relevance, k=10):
        """순위별 할인 누적 이득"""
        if not relevance:
            return 0.0
        rel = relevance[:k]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(relevance, reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(recommended, relevant, k=10):
        """평균 정확도"""
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
        """평균 역순위"""
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
        """리스트 내 다양성"""
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
        """카탈로그 커버리지"""
        unique_recs = set()
        for recs in all_recs:
            unique_recs.update(recs)
        return len(unique_recs) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def novelty(recs, popularity):
        """새로움 정도"""
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
        """인기도 편향"""
        if not recs:
            return 0.0
        pop_scores = [popularity.get(rec, 0.5) for rec in recs]
        return np.mean(pop_scores)
    
    @staticmethod
    def sparsity_aware_score(test_data, predictions, num_users, num_items):
        """희소성을 고려한 성능 점수"""
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

# ================================
# 데이터 로드 함수 (Small & 1M 지원)
# ================================

def load_movielens(dataset_type='Small'):
    """MovieLens 데이터셋 다운로드 및 로드 (Small 또는 1M)"""
    datasets_info = {
        'Small': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'extract_dir': 'movielens_data/ml-latest-small',
            'folder_name': 'ml-latest-small',
            'encoding': 'utf-8'  # Small은 UTF-8
        },
        '1M': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'extract_dir': 'movielens_data/ml-1m',
            'folder_name': 'ml-1m',
            'encoding': 'iso-8859-1'  # 1M은 Latin-1 인코딩
        }
    }
    
    if dataset_type not in datasets_info:
        print(f"❌ 지원하지 않는 데이터셋: {dataset_type}")
        print(f"   지원 데이터셋: {list(datasets_info.keys())}")
        return None, None
    
    info = datasets_info[dataset_type]
    os.makedirs('movielens_data', exist_ok=True)
    
    # 데이터 다운로드 (처음만)
    if not os.path.exists(info['extract_dir']):
        print(f"📥 {dataset_type} 데이터셋 다운로드 중...")
        try:
            response = requests.get(info['url'], timeout=30)
            response.raise_for_status()
            
            # io.BytesIO 사용
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall('movielens_data')
            print(f"✅ 다운로드 완료")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            print(f"   이유: {type(e).__name__}")
            return None, None
    
    # 데이터 로드
    try:
        extract_path = info['extract_dir']
        encoding = info['encoding']
        
        # ⭐ Small과 1M의 다른 포맷 + 인코딩 처리
        if dataset_type == 'Small':
            ratings = pd.read_csv(f'{extract_path}/ratings.csv', encoding=encoding)
            movies = pd.read_csv(f'{extract_path}/movies.csv', encoding=encoding)
            
        elif dataset_type == '1M':
            # 1M 데이터는 구분자가 다름 (::) + 인코딩 지정
            ratings = pd.read_csv(
                f'{extract_path}/ratings.dat',
                sep='::',
                header=None,
                engine='python',
                encoding=encoding,  # ⭐ 인코딩 지정
                names=['userId', 'movieId', 'rating', 'timestamp']
            )
            movies = pd.read_csv(
                f'{extract_path}/movies.dat',
                sep='::',
                header=None,
                engine='python',
                encoding=encoding,  # ⭐ 인코딩 지정
                names=['movieId', 'title', 'genres']
            )
        
        print(f"✅ {dataset_type} 데이터셋 로드 완료")
        print(f"   사용자: {ratings['userId'].nunique():,}명")
        print(f"   영화: {movies['movieId'].nunique():,}개")
        print(f"   평점: {len(ratings):,}개")
        
        # 통계 정보
        sparsity = 1 - (len(ratings) / (ratings['userId'].nunique() * movies['movieId'].nunique()))
        print(f"   희소성: {sparsity:.4f} ({sparsity*100:.2f}%)")
        
        return ratings, movies
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        print(f"   이유: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None

# ================================
# 하이브리드 추천 시스템 클래스
# ================================

class OptimizedHybridRecommender:
    """모든 알고리즘 (CF/CB/Hybrid) 성능 측정"""
    
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
        """전처리 및 모델 준비"""
        print(f"\n📊 {self.name} 전처리 중...")
        
        # Train/Test 분할
        unique_users = self.ratings['userId'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=0.2, random_state=42
        )
        
        self.train = self.ratings[self.ratings['userId'].isin(train_users)]
        self.test = self.ratings[self.ratings['userId'].isin(test_users)]
        
        print(f"  ✅ Train: {len(self.train):,}, Test: {len(self.test):,}")
        
        # 사용자-영화 행렬
        self.um = self.train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(self.mean_rating)
        
        # SVD 분해
        print(f"  🔄 SVD 분해 중...")
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
        print(f"  🎯 Bias 계산 중...")
        for user_id in self.train['userId'].unique():
            user_ratings = self.train[self.train['userId'] == user_id]
            user_mean = user_ratings['rating'].mean()
            self.user_bias[user_id] = user_mean - self.mean_rating
        
        for movie_id in self.train['movieId'].unique():
            movie_ratings = self.train[self.train['movieId'] == movie_id]
            movie_mean = movie_ratings['rating'].mean()
            self.movie_bias[movie_id] = movie_mean - self.mean_rating
        
        # 콘텐츠 특징
        print(f"  🎬 콘텐츠 특징 생성 중...")
        movies_in_train = self.movies[self.movies['movieId'].isin(
            self.train['movieId'].unique()
        )].copy()
        
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(
            movies_in_train['genres'].str.split('|')
        )
        
        self.movie_features = {}
        self.item_to_idx = {}
        for i, movie_id in enumerate(movies_in_train['movieId']):
            self.movie_features[movie_id] = genres_matrix[i]
            self.item_to_idx[movie_id] = i
        
        self.item_similarity = cosine_similarity(genres_matrix)
        
        # 인기도
        popularity_series = self.train.groupby('movieId')['rating'].count()
        popularity_normalized = (popularity_series - popularity_series.min()) / (
            popularity_series.max() - popularity_series.min() + 1e-8
        )
        self.popularity = popularity_normalized.to_dict()
        
        print(f"✅ {self.name} 전처리 완료")
    
    # ===== 예측 메서드 =====
    
    def predict_cf(self, user_id, movie_id):
        """협업 필터링"""
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
        """콘텐츠 기반"""
        user_ratings = self.train[self.train['userId'] == user_id]
        
        if user_ratings.empty or movie_id not in self.movie_features:
            return self.mean_rating
        
        rated_movies = user_ratings['movieId'].values
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for rated_id in rated_movies:
            if rated_id in self.item_to_idx and movie_id in self.item_to_idx:
                idx_rated = self.item_to_idx[rated_id]
                idx_movie = self.item_to_idx[movie_id]
                similarity = self.item_similarity[idx_movie][idx_rated]
                rating = user_ratings[user_ratings['movieId'] == rated_id]['rating'].values[0]
                weighted_sum += similarity * rating
                similarity_sum += similarity
        
        if similarity_sum > 0:
            return weighted_sum / similarity_sum
        else:
            return self.mean_rating
    
    def predict_weighted_avg(self, user_id, movie_id, alpha=0.4):
        """가중 평균"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        pred = alpha * cf + (1 - alpha) * cb
        return np.clip(pred, 1, 5)
    
    def predict_feature_combo(self, user_id, movie_id):
        """특징 결합"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        ub = self.user_bias.get(user_id, 0)
        mb = self.movie_bias.get(movie_id, 0)
        pred = 0.5*cf + 0.3*cb + 0.1*ub + 0.1*mb
        return np.clip(pred, 1, 5)
    
    def predict_mixed(self, user_id, movie_id):
        """혼합"""
        cf = self.predict_cf(user_id, movie_id)
        cb = self.predict_cb(user_id, movie_id)
        confidence = 0.5
        pred = confidence * cf + (1 - confidence) * cb
        return np.clip(pred, 1, 5)
    
    def get_recommendations(self, user_id, n=10, method='weighted_avg'):
        """추천 생성"""
        watched = set(self.train[self.train['userId'] == user_id]['movieId'])
        predictions = []
        
        for movie_id in self.movies['movieId']:
            if movie_id not in self.movie_features:
                continue
            
            if movie_id not in watched:
                if method == 'cf':
                    pred = self.predict_cf(user_id, movie_id)
                elif method == 'cb':
                    pred = self.predict_cb(user_id, movie_id)
                elif method == 'weighted_avg':
                    pred = self.predict_weighted_avg(user_id, movie_id)
                elif method == 'feature_combo':
                    pred = self.predict_feature_combo(user_id, movie_id)
                elif method == 'mixed':
                    pred = self.predict_mixed(user_id, movie_id)
                else:
                    pred = self.predict_weighted_avg(user_id, movie_id)
                
                if 1 <= pred <= 5:
                    predictions.append((movie_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in predictions[:n]]
    
    def evaluate(self):
        """종합 평가 - 모든 알고리즘 포함"""
        print(f"\n📈 {self.name} 평가 중...")
        
        results = {}
        methods = {
            'CF': self.predict_cf,
            'CB': self.predict_cb,
            'Weighted_Avg': self.predict_weighted_avg,
            'Feature_Combo': self.predict_feature_combo,
            'Mixed': self.predict_mixed
        }
        
        for method_name, predict_func in methods.items():
            print(f"  📊 {method_name} 평가 중...")
            
            # Step 1: 기본 정확도 지표
            actuals = []
            preds = []
            
            for _, row in self.test.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                actual = row['rating']
                pred = predict_func(user_id, movie_id)
                actuals.append(actual)
                preds.append(pred)
            
            # Step 2: Sparsity-Aware Score
            num_users = len(self.um.index)
            num_items = len(self.um.columns)
            sparsity_result = self.metrics.sparsity_aware_score(
                test_data=np.array(actuals),
                predictions=np.array(preds),
                num_users=num_users,
                num_items=num_items
            )
            
            # Step 3: 추천 지표 계산
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
                
                # 추천 생성
                if method_name == 'CF':
                    recs = self.get_recommendations(user_id, n=10, method='cf')
                elif method_name == 'CB':
                    recs = self.get_recommendations(user_id, n=10, method='cb')
                elif method_name == 'Weighted_Avg':
                    recs = self.get_recommendations(user_id, n=10, method='weighted_avg')
                elif method_name == 'Feature_Combo':
                    recs = self.get_recommendations(user_id, n=10, method='feature_combo')
                elif method_name == 'Mixed':
                    recs = self.get_recommendations(user_id, n=10, method='mixed')
                else:
                    recs = []
                
                relevant = user_test[user_test['rating'] >= 4]['movieId'].tolist()
                
                if recs and relevant:
                    precisions.append(self.metrics.precision_at_k(recs, relevant, 10))
                    recalls.append(self.metrics.recall_at_k(recs, relevant, 10))
                    f1s.append(self.metrics.f1_at_k(recs, relevant, 10))
                    maps.append(self.metrics.map_at_k(recs, relevant, 10))
                    mrrs.append(self.metrics.mrr_at_k(recs, relevant, 10))
                    relevance = [1 if m in relevant else 0 for m in recs]
                    ndcgs.append(self.metrics.ndcg_at_k(relevance, 10))
            
            # Step 4: 다양성 지표
            all_recs = []
            for user_id in test_users:
                if method_name == 'CF':
                    recs = self.get_recommendations(user_id, n=10, method='cf')
                elif method_name == 'CB':
                    recs = self.get_recommendations(user_id, n=10, method='cb')
                elif method_name == 'Weighted_Avg':
                    recs = self.get_recommendations(user_id, n=10, method='weighted_avg')
                elif method_name == 'Feature_Combo':
                    recs = self.get_recommendations(user_id, n=10, method='feature_combo')
                elif method_name == 'Mixed':
                    recs = self.get_recommendations(user_id, n=10, method='mixed')
                else:
                    recs = []
                
                if recs:
                    all_recs.append(recs)
            
            diversity = self.metrics.intra_list_diversity(
                [r for recs in all_recs for r in recs],
                self.item_similarity,
                self.item_to_idx
            )
            
            coverage = self.metrics.coverage(all_recs, len(self.movies))
            
            novelty = self.metrics.novelty(
                [r for recs in all_recs for r in recs],
                self.popularity
            )
            
            pop_bias = self.metrics.popularity_bias(
                [r for recs in all_recs for r in recs],
                self.popularity
            )
            
            # Step 5: 결과 저장
            results[method_name] = {
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
            
            print(f"    ✅ {method_name}: RMSE={results[method_name]['RMSE']:.4f}")
        
        return results

# ================================
# 메인 실행
# ================================

def main():
    # ⭐ 두 데이터셋 모두 처리
    datasets_to_test = ['Small', '1M']
    all_results = []
    
    for dataset in datasets_to_test:
        print(f"\n{'='*100}")
        print(f"📊 {dataset} 데이터셋 처리 중...")
        print(f"{'='*100}")
        
        # 데이터 로드
        ratings, movies = load_movielens(dataset)
        
        if ratings is None or movies is None:
            print(f"❌ {dataset} 데이터셋 로드 실패, 건너뜀")
            continue
        
        # 모델 생성 및 평가
        model = OptimizedHybridRecommender(ratings, movies, name=dataset, svd_dim=200)
        model._prepare()
        results = model.evaluate()
        
        # 결과 저장
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
        
        print("\n✅ 추천 정확성 지표:")
        print(results_df[['Dataset', 'Method', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']].to_string(index=False))
        
        print("\n✅ 순위 지표:")
        print(results_df[['Dataset', 'Method', 'MAP@10', 'MRR@10', 'NDCG@10']].to_string(index=False))
        
        print("\n✅ 다양성 지표:")
        print(results_df[['Dataset', 'Method', 'Diversity', 'Coverage', 'Novelty', 'PopularityBias']].to_string(index=False))
        
        # CSV 저장
        output_filename = 'hybrid_results_small_1m_ver7.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\n✅ 결과 저장: {output_filename}")
        
        # 데이터셋별 비교
        print("\n📊 데이터셋별 비교:")
        print("-" * 100)
        
        for dataset in datasets_to_test:
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            if len(dataset_results) > 0:
                best_method = dataset_results.loc[dataset_results['RMSE'].idxmin(), 'Method']
                best_rmse = dataset_results['RMSE'].min()
                best_adj_rmse = dataset_results['Adjusted_RMSE'].min()
                best_precision = dataset_results['Precision@10'].max()
                
                print(f"\n{dataset}:")
                print(f"  최고 성능 알고리즘 (RMSE): {best_method}")
                print(f"  최저 RMSE: {best_rmse:.4f}")
                print(f"  최저 Adjusted_RMSE: {best_adj_rmse:.4f}")
                print(f"  최고 Precision@10: {best_precision:.4f}")
    else:
        print("❌ 처리된 데이터셋이 없습니다")

if __name__ == "__main__":
    main()
