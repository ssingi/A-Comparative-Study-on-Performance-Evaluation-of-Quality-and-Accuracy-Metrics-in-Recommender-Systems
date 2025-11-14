import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ✅ 수정: 직접 import 방식으로 변경
try:
    from src.data_loader import load_movielens
    from src.algorithms.CF_collaborative_filtering import CollaborativeFiltering
    from src.algorithms.CB_content_based import ContentBased
    from src.algorithms.weighted_hybrid import WeightedHybrid
    from src.algorithms.feature_combination import FeatureCombination
    from src.algorithms.mixed_hybrid import MixedHybrid
    from src.evaluator import RecommenderEvaluator
except ModuleNotFoundError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {project_root}")
    print(f"sys.path: {sys.path[:3]}")
    sys.exit(1)


print("=" * 100)
print("🎬 MovieLens 하이브리드 추천 시스템 - 개별 알고리즘 평가")
print("=" * 100)


def main():
    """MovieLens 데이터셋 평가 실행"""
    # ⭐ 수정: 두 데이터셋 모두 사용
    datasets_to_test = ['Small', '1M']  # Small과 1M 모두 평가
    all_results = []

    for dataset in datasets_to_test:
        print(f"\n{'='*100}")
        print(f"📊 {dataset} 데이터셋 평가 중...")
        print(f"{'='*100}")

        # 데이터 로드
        ratings, movies = load_movielens(dataset)
        if ratings is None or movies is None:
            print(f"❌ {dataset} 데이터셋 로드 실패")
            continue

        # 5가지 모델 생성
        # ⭐ 수정: 1M 데이터셋은 더 큰 svd_dim 사용 (성능 최적화)
        svd_dim = 200 if dataset == 'Small' else 300  # 1M은 300차원
        
        models = [
            CollaborativeFiltering(ratings, movies, name='CF', svd_dim=svd_dim),
            ContentBased(ratings, movies, name='CB'),
            WeightedHybrid(ratings, movies, name='WeightedHybrid', svd_dim=svd_dim),
            FeatureCombination(ratings, movies, name='FeatureCombination', svd_dim=svd_dim),
            MixedHybrid(ratings, movies, name='MixedHybrid', svd_dim=svd_dim)
        ]

        # 각 모델 평가
        for model in models:
            try:
                print(f"\n{'─'*100}")
                print(f"🔍 {model.name} 평가 시작 ({dataset} 데이터셋)")
                print(f"{'─'*100}")
                
                # 학습
                model.fit()
                
                # 평가
                evaluator = RecommenderEvaluator(model)
                result = evaluator.evaluate()
                result['Dataset'] = dataset  # 데이터셋 이름 추가
                all_results.append(result)
                
                print(f"✅ {model.name} 평가 완료")
                
            except Exception as e:
                print(f"❌ {model.name} 평가 실패: {e}")
                import traceback
                traceback.print_exc()

    # 최종 결과 출력
    if all_results:
        print(f"\n{'='*100}")
        print("📈 최종 평가 결과")
        print(f"{'='*100}")

        results_df = pd.DataFrame(all_results)

        # ⭐ 데이터셋별 결과 출력
        for dataset in datasets_to_test:
            dataset_results = results_df[results_df['Dataset'] == dataset]
            if len(dataset_results) == 0:
                continue
                
            print(f"\n{'='*100}")
            print(f"📊 {dataset} 데이터셋 - 정확도 지표 (RMSE, MAE)")
            print(f"{'='*100}")
            accuracy_cols = ['Model', 'RMSE', 'MAE', 'Adjusted_RMSE']
            print(dataset_results[accuracy_cols].to_string(index=False))

            print(f"\n{'='*100}")
            print(f"📊 {dataset} 데이터셋 - 순위 지표 (Precision, Recall, F1, NDCG, MAP, MRR)")
            print(f"{'='*100}")
            ranking_cols = ['Model', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10', 'MAP@10', 'MRR@10']
            print(dataset_results[ranking_cols].to_string(index=False))

            print(f"\n{'='*100}")
            print(f"📊 {dataset} 데이터셋 - 다양성 지표 (Diversity, Coverage, Novelty, PopularityBias)")
            print(f"{'='*100}")
            diversity_cols = ['Model', 'Diversity', 'Coverage', 'Novelty', 'PopularityBias']
            print(dataset_results[diversity_cols].to_string(index=False))

        # ⭐ 수정: 데이터셋별 CSV 저장
        for dataset in datasets_to_test:
            dataset_results = results_df[results_df['Dataset'] == dataset]
            if len(dataset_results) > 0:
                output_filename = f'hybrid_recommender_{dataset}_results.csv'
                dataset_results.to_csv(output_filename, index=False)
                print(f"\n✅ {dataset} 결과 저장: {output_filename}")
        
        # ⭐ 추가: 전체 결과 통합 CSV 저장
        all_output_filename = 'hybrid_recommender_all_results.csv'
        results_df.to_csv(all_output_filename, index=False)
        print(f"\n✅ 전체 결과 저장: {all_output_filename}")
        
    else:
        print("❌ 처리된 데이터셋이 없습니다")


if __name__ == "__main__":
    main()