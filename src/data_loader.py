import os
import zipfile
import urllib.request
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


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