# OTT 플랫폼을 위한 하이브리드 추천 시스템의 다면적 성능평가

**Multi-perspective Performance Evaluation of Hybrid Recommendation Systems for OTT Platform**

> 📄 2025 한국지능시스템학회(KIIS) 추계학술대회 발표 — *Proceedings of KIIS Autumn Conference 2025, Vol. 35, No. 2*
> 🏆 대구대학교 SW중심대학사업단 성과 공유회 **우수상** (2025.11)

MovieLens 데이터셋에서 **단일 추천 모델 2종(CB, CF)** 과 **하이브리드 추천 모델 3종**을 같은 조건으로 구현하고,
정확도뿐 아니라 **순위·다양성·인기 편향까지 14개 지표**로 비교한 실험 코드입니다.

---

## 🎯 연구 동기

기존 추천 시스템 연구는 주로 RMSE·MAE 같은 **정확도 지표**로 성능을 평가합니다.
하지만 실제 OTT 서비스에서는 인기작만 반복해서 추천하는 **필터 버블**과 **다양성 부족**도 중요한 문제입니다.
이 프로젝트는 "정확도가 높은 모델이 좋은 추천을 하는가?"를 여러 관점의 지표로 검증합니다.

---

## 🧠 구현한 모델

모든 모델은 `BaseRecommender`(`fit` / `predict` / `recommend`)를 상속해 같은 인터페이스로 구현했습니다.

| 모델 | 파일 | 방식 |
|---|---|---|
| **CF** 협업 필터링 | `src/algorithms/CF_collaborative_filtering.py` | 사용자-영화 평점 행렬(결측치는 전체 평균으로 채움) → `TruncatedSVD` 행렬 분해 → 잠재 벡터 내적으로 평점 예측 |
| **CB** 콘텐츠 기반 | `src/algorithms/CB_content_based.py` | 장르 원-핫 벡터 간 코사인 유사도 × 사용자의 기존 평점으로 가중 평균 |
| **Weighted Hybrid** | `src/algorithms/weighted_hybrid.py` | bias 포함 SVD CF + 5개 특징(장르·인기도·참신성·연도·평균 평점) CB를 **사용자 활동량에 따라 가중 평균** |
| **Feature Combination** | `src/algorithms/feature_combination.py` | CF 점수 · CB 점수 · 상호작용 항(CF×CB)을 0.4 / 0.4 / 0.2로 비선형 결합 |
| **Mixed Hybrid** | `src/algorithms/mixed_hybrid.py` | CF·CB·Weighted·FeatureCombo 4개 예측값에서 IQR로 이상치를 제거한 뒤, 활동량에 따라 평균 / 중앙값 / CB 단독으로 통합 |

---

## 📏 평가 지표 (14종)

| 분류 | 지표 | 파일 |
|---|---|---|
| 정확도 | RMSE, MAE, Adjusted RMSE, Adjusted MAE (`RMSE / (1 + Sparsity)`) | `src/metrics.py` |
| 순위 (@10) | Precision, Recall, F1, NDCG, MAP, MRR (평점 4점 이상 = 관련 있음) | `src/metrics.py` |
| 품질 | Diversity, Coverage, Novelty, Popularity Bias | `src/metrics.py` |

평가 절차는 `src/evaluator.py`에 있습니다.

- 데이터 분할: 평점 단위 무작위 80:20 (`random_state=42`)
- 정확도: 테스트셋의 모든 평점으로 계산
- 순위·품질: 테스트 사용자 중 무작위 표본(**Small 100명 / 1M 300명**, `seed=42`)으로 계산
- SVD 차원: Small 200, 1M 300 (`main.py`)

---

## 📊 실험 결과

전체 결과는 [`hybrid_recommender_all_results.csv`](hybrid_recommender_all_results.csv)에 있습니다. **굵게** = 해당 데이터셋 1위.

### MovieLens Small (610명 · 9,742편 · 약 10만 평점)

| 모델 | RMSE ↓ | MAE ↓ | Precision@10 ↑ | NDCG@10 ↑ | MRR@10 ↑ | Diversity ↑ | Coverage ↑ | Novelty ↑ | Pop. Bias ↓ |
|---|---|---|---|---|---|---|---|---|---|
| CF | 1.039 | 0.823 | **0.080** | **0.287** | **0.234** | 0.712 | 0.043 | 10.16 | 0.301 |
| CB | **0.924** | **0.716** | 0.021 | 0.094 | 0.076 | **0.842** | 0.045 | **13.27** | **0.141** |
| Weighted | 1.339 | 1.052 | 0.057 | 0.194 | 0.164 | 0.655 | 0.038 | 10.71 | 0.251 |
| FeatureCombo | 1.558 | 1.268 | 0.011 | 0.052 | 0.037 | 0.710 | 0.011 | 11.23 | 0.204 |
| Mixed | 1.355 | 1.067 | 0.018 | 0.068 | 0.048 | 0.709 | **0.067** | 12.07 | 0.154 |

### MovieLens 1M (6,040명 · 3,706편 · 약 100만 평점)

| 모델 | RMSE ↓ | MAE ↓ | Precision@10 ↑ | NDCG@10 ↑ | MRR@10 ↑ | Diversity ↑ | Coverage ↑ | Novelty ↑ | Pop. Bias ↓ |
|---|---|---|---|---|---|---|---|---|---|
| CF | 1.066 | 0.874 | **0.104** | **0.332** | **0.276** | 0.757 | 0.189 | 10.21 | 0.272 |
| CB | **1.016** | **0.812** | 0.015 | 0.058 | 0.039 | **0.883** | 0.149 | **12.20** | **0.162** |
| Weighted | 1.390 | 1.100 | 0.074 | 0.251 | 0.199 | 0.653 | 0.130 | 10.11 | 0.337 |
| FeatureCombo | 1.606 | 1.267 | 0.031 | 0.142 | 0.125 | 0.693 | 0.031 | 10.40 | 0.264 |
| Mixed | 1.407 | 1.114 | 0.029 | 0.101 | 0.063 | 0.744 | **0.261** | 11.39 | 0.220 |

### 핵심 발견

- **정확도는 CB, 순위는 CF가 1위**: 두 데이터셋 모두 같은 경향 → 정확도와 순위 사이의 trade-off
- **CB는 다양성·참신성이 가장 높고 인기 편향이 가장 낮음**: 인기작에 덜 쏠림
- **하이브리드는 정확도·순위에서 단일 모델을 넘지 못함**: 하이브리드가 1위를 한 지표는 Mixed의 **Coverage**(1M 0.261) 하나
- → 모델을 복잡하게 결합한다고 성능이 좋아지지는 않으며, **서비스 목표에 맞는 지표 선택과 가중치 조절**이 중요함

---

## 🚀 실행 방법

```bash
git clone https://github.com/ssingi/A-Comparative-Study-on-Performance-Evaluation-of-Quality-and-Accuracy-Metrics-in-Recommender-Systems.git
cd A-Comparative-Study-on-Performance-Evaluation-of-Quality-and-Accuracy-Metrics-in-Recommender-Systems

pip install -r requirements.txt   # numpy, pandas, scikit-learn, tqdm
python main.py
```

- MovieLens 데이터가 `movielens_data/`에 없으면 `src/data_loader.py`가 [GroupLens](https://grouplens.org/datasets/movielens/)에서 자동으로 내려받습니다.
- 실행이 끝나면 `hybrid_recommender_{Small,1M,all}_results.csv`가 생성됩니다.
- 1M 데이터셋은 정확도 평가(테스트 평점 약 20만 개)에 시간이 오래 걸릴 수 있습니다.

---

## 📁 저장소 구조

```
├── main.py                          # Small·1M 데이터셋으로 5개 모델 학습 → 평가 → CSV 저장
├── test_small_feature.py            # FeatureCombination 디버깅용 스크립트
├── requirements.txt
├── hybrid_recommender_*_results.csv # 실험 결과
├── src/
│   ├── data_loader.py               # MovieLens 다운로드·로드
│   ├── evaluator.py                 # 정확도·순위·품질 평가 절차
│   ├── metrics.py                   # 14개 평가 지표 구현
│   └── algorithms/
│       ├── base_recommender.py      # 공통 인터페이스 (fit / predict / recommend)
│       ├── CF_collaborative_filtering.py
│       ├── CB_content_based.py
│       ├── weighted_hybrid.py
│       ├── feature_combination.py
│       └── mixed_hybrid.py
├── 기존 과거 코드/                   # 단일 파일 버전(first, v12) 및 참고 자료
├── 평가지표 및 SVD설명 보는법/        # SVD·편향 정규화·평가 지표 설명 문서
└── 희소성 지표/                      # MovieLens 희소성 설명 문서
```

---

## ⚠️ 한계 및 향후 과제

- **NDCG@10**: 현재 구현은 이상적인 순위(IDCG)를 추천된 10개 안에서만 계산하므로 표준 NDCG보다 값이 높게 나옵니다. 모든 모델에 같은 식을 적용했기 때문에 모델 간 비교는 유효하지만, 절대값은 다른 연구와 직접 비교할 수 없습니다. 사용자의 전체 관련 항목 기준 IDCG로 수정할 예정입니다.
- **Adjusted RMSE/MAE**: 같은 데이터셋 안에서는 모든 모델이 같은 값으로 나눠지므로 모델 순위는 바뀌지 않습니다.
- **평가 설계**: 무작위 분할·단일 시드·사용자 표본으로 평가했습니다. 시간 기준 분할, 여러 시드의 평균과 신뢰구간이 다음 단계입니다.
- **하이퍼파라미터**: 하이브리드 가중치는 선행 연구를 참고한 값이며, 검증 세트로 튜닝하지 않았습니다.
- **CF 구현**: 결측치를 평균으로 채운 뒤 SVD를 적용하는 방식이라, 관측값만으로 학습하는 bias-MF(SGD/ALS)로 바꾸면 결과가 달라질 수 있습니다.

---

## 👥 연구진

대구대학교 컴퓨터정보공학부 컴퓨터공학전공
김수현 · 김찬호 · 김민성 · 박세영 · 문준영 · 김지연(지도교수)

본 연구는 과학기술정보통신부 및 정보통신기획평가원의 'SW중심대학사업'의 지원을 받아 수행되었습니다. (2025-0-00077)

## 📚 주요 참고문헌

- R. Burke, "Hybrid recommender systems: Survey and experiments," *User Modeling and User-Adapted Interaction*, 2002.
- Y. Koren, R. Bell, C. Volinsky, "Matrix factorization techniques for recommender systems," *Computer*, 2009.
- M. Claypool et al., "Combining content-based and collaborative filters in an online newspaper," *ACM SIGIR Workshop*, 1999.
- P. Melville, R. J. Mooney, R. Nagarajan, "Content-boosted collaborative filtering for improved recommendations," *AAAI*, 2002.
