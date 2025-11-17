# A Comparative Study on Performance Evaluation of Quality and Accuracy Metrics in Recommender Systems

이 저장소는 추천 시스템의 다양한 성능 지표(품질 및 정확도)를 비교하고 평가하는 연구 프로젝트의 자료를 포함합니다.

## 📝 개요 (Overview)

본 프로젝트는 추천 시스템의 성능을 평가하는 데 사용되는 여러 지표들을 비교 분석합니다. SVD, SVD++와 같은 협업 필터링 알고리즘 및 가중 하이브리드 모델을 구현하고, 이 모델들의 성능을 다양한 관점에서 평가한 내용을 담고 있습니다.

## 📁 저장소 구성 (Repository Contents)

- **`Project_Report.pdf`**:
  프로젝트의 전체 내용을 다루는 최종 보고서입니다. 연구 배경, 방법론, 실험 설정, 결과 및 결론이 포함되어 있습니다.

- **`Presentation.pdf`**:
  프로젝트의 주요 내용을 요약한 발표 자료입니다.

- **`SVD.ipynb`**:
  **SVD (Singular Value Decomposition)** 알고리즘을 구현한 Jupyter Notebook 파일입니다.

- **`SVD_pp.ipynb`**:
  **SVD++** (SVD의 개선 버전) 알고리즘을 구현한 Jupyter Notebook 파일입니다.

- **`Weighted_Hybrid.ipynb`**:
  **가중 하이브리드 (Weighted Hybrid)** 추천 모델을 구현한 Jupyter Notebook 파일입니다.

- **`Dataset/`**:
  본 연구에 사용된 데이터셋 파일(`movies.csv`, `ratings.csv`, `tags.csv` 등)이 포함된 디렉토리입니다. (MovieLens 데이터셋 기반으로 보입니다.)

- **`LICENSE`**:
  프로젝트 라이선스 파일입니다.

## 💡 주요 구현 알고리즘 (Algorithms Implemented)

- **SVD (Singular Value Decomposition)**
- **SVD++**
- **Weighted Hybrid Model**

## 💻 사용 방법 (Usage)

1.  이 저장소를 클론(clone)합니다.
    ```bash
    git clone [https://github.com/ssingi/A-Comparative-Study-on-Performance-Evaluation-of-Quality-and-Accuracy-Metrics-in-Recommender-Systems.git](https://github.com/ssingi/A-Comparative-Study-on-Performance-Evaluation-of-Quality-and-Accuracy-Metrics-in-Recommender-Systems.git)
    ```
2.  필요한 Python 라이브러리 (예: `pandas`, `numpy`, `scikit-learn`, `surprise` 등)를 설치합니다.
3.  Jupyter Notebook 또는 Jupyter Lab을 실행하여 `.ipynb` 파일들을 열람하고 실행할 수 있습니다.
    ```bash
    jupyter notebook
    ```
4.  자세한 연구 내용과 결과는 `Project_Report.pdf` 파일을 참고하십시오.

## 📄 라이선스 (License)

이 프로젝트는 `LICENSE` 파일에 명시된 라이선스를 따릅니다.
