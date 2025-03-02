# AWS機械学習関連用語解説

## 線形学習アルゴリズム
入力特徴と出力の間に線形関係を仮定する機械学習アルゴリズムです。シンプルで解釈しやすく、計算効率が高い特性があります。

**主な特徴:**
- 入力特徴の線形結合で予測を行う
- モデルパラメータの解釈が容易
- 計算効率が高く、大規模データにも適用可能
- 過学習のリスクが比較的低い
- 特徴間の相互作用を直接モデル化しない

**代表的な線形アルゴリズム:**
- 線形回帰
- ロジスティック回帰
- 線形判別分析（LDA）
- サポートベクターマシン（線形カーネル）
- 単層パーセプトロン

**数学的表現:**
- 回帰: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
- 分類: P(y=1) = 1/(1+e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))

**最適化手法:**
- 最小二乗法
- 勾配降下法
- 確率的勾配降下法（SGD）
- L1/L2 正則化
- 座標降下法

**AWS での実装:**
- SageMaker の線形学習器アルゴリズム
- SageMaker AutoPilot での自動モデル選択
- AWS Glue での機械学習変換
- Amazon Redshift ML での SQL ベース実装
- AWS Lambda での軽量推論

**ユースケース:**
- 需要予測
- 価格最適化
- リスクスコアリング
- A/B テスト結果分析
- 特徴重要度の解釈が必要なシナリオ

## 線形学習器（Linear learner）
AmazonSageMakerが提供する組み込みアルゴリズムで、線形モデルを使用して分類または回帰問題を解決します。大規模データセットに対して効率的です。

**主な特徴:**
- 分類と回帰の両方に対応
- 複数の最適化アルゴリズムをサポート
- 自動ハイパーパラメータ最適化
- 分散トレーニングに対応
- 組み込みの正則化機能

**サポートする問題タイプ:**
- バイナリ分類
- マルチクラス分類
- 回帰
- 量子化バイナリ分類（QBC）

**ハイパーパラメータ:**
- 学習率
- 正則化パラメータ（L1、L2）
- エポック数
- ミニバッチサイズ
- 損失関数タイプ
- オプティマイザータイプ

**入力データ形式:**
- CSV
- RecordIO-protobuf
- Parquet
- 疎行列形式

**実装例:**
```python
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

# SageMaker セッションの作成
session = sagemaker.Session()

# 線形学習器のコンテナイメージを取得
container = get_image_uri(session.boto_region_name, 'linear-learner')

# 線形学習器の設定
linear = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    sagemaker_session=session
)

# ハイパーパラメータの設定
linear.set_hyperparameters(
    predictor_type='binary_classifier',
    feature_dim=features_dim,
    mini_batch_size=100,
    epochs=10,
    l1=0.01
)

# トレーニングの実行
linear.fit({'train': train_input})
```

**パフォーマンス最適化:**
- 特徴のスケーリングと正規化
- 特徴選択による次元削減
- 正則化パラメータのチューニング
- 学習率スケジューリング
- 早期停止の活用

**ユースケース:**
- クリックスルー率（CTR）予測
- 顧客解約予測
- 信用リスク評価
- 在庫需要予測
- 異常検知（線形モデルベース）

## ニュートラルトピックモデル
文書内の単語分布からトピックを抽出する確率的モデルです。文書コレクションの潜在的な意味構造を発見するのに役立ちます。

**主な特徴:**
- 文書を潜在的なトピックの混合として表現
- 教師なし学習アプローチ
- 次元削減と意味的クラスタリングを実現
- 文書間の類似性を計算可能
- 大規模テキストコーパスの分析に有効

**代表的なトピックモデル:**
- Latent Dirichlet Allocation (LDA)
- Probabilistic Latent Semantic Analysis (PLSA)
- Hierarchical Dirichlet Process (HDP)
- Correlated Topic Model (CTM)
- Neural Topic Model (NTM)

**LDA の数学的基礎:**
- 文書はトピック分布から生成される
- 各トピックは単語分布から生成される
- ディリクレ分布を事前分布として使用
- ギブスサンプリングやバリエーション推論で学習

**AWS での実装:**
- SageMaker の組み込み LDA アルゴリズム
- SageMaker の Neural Topic Model
- Amazon Comprehend のトピックモデリング機能
- カスタムモデルの SageMaker スクリプトモード
- AWS Batch での大規模処理

**実装例:**
```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# SageMaker セッションの作成
session = sagemaker.Session()
role = get_execution_role()

# LDA のコンテナイメージを取得
container = get_image_uri(session.boto_region_name, 'lda')

# LDA の設定
lda = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    sagemaker_session=session
)

# ハイパーパラメータの設定
lda.set_hyperparameters(
    num_topics=10,
    feature_dim=vocabulary_size,
    mini_batch_size=128,
    alpha0=1.0
)

# トレーニングの実行
lda.fit({'train': train_input})
```

**評価指標:**
- Perplexity（パープレキシティ）
- Topic Coherence（トピック一貫性）
- Topic Diversity（トピック多様性）
- Log-likelihood（対数尤度）
- Silhouette Score（シルエットスコア）

**ユースケース:**
- 文書分類と整理
- コンテンツ推薦システム
- トレンド分析と検出
- 顧客フィードバック分析
- 学術研究の文献マッピング
- ニュース記事のカテゴリ化

## K-means クラスタリング
データポイントをK個のクラスターに分割する教師なし学習アルゴリズムです。各データポイントを最も近い中心点のクラスターに割り当てます。

**主な特徴:**
- 教師なし学習アルゴリズム
- 事前に指定した K 個のクラスターにデータを分割
- ユークリッド距離に基づくクラスター割り当て
- 反復的な最適化プロセス
- スケーラブルで実装が容易

**アルゴリズムのステップ:**
1. K 個のクラスター中心をランダムに初期化
2. 各データポイントを最も近いクラスター中心に割り当て
3. 各クラスターの新しい中心点を計算（所属ポイントの平均）
4. クラスター割り当てが変化しなくなるまで 2-3 を繰り返す

**AWS での実装:**
- SageMaker の組み込み K-means アルゴリズム
- SageMaker の webscale K-means（大規模データ向け）
- AWS Glue での K-means 変換
- Amazon Redshift ML での SQL ベース実装
- EMR での Spark MLlib K-means

**実装例:**
```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# SageMaker セッションの作成
session = sagemaker.Session()
role = get_execution_role()

# K-means のコンテナイメージを取得
container = get_image_uri(session.boto_region_name, 'kmeans')

# K-means の設定
kmeans = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    sagemaker_session=session
)

# ハイパーパラメータの設定
kmeans.set_hyperparameters(
    k=10,
    feature_dim=feature_dim,
    mini_batch_size=500,
    init_method='kmeans++'
)

# トレーニングの実行
kmeans.fit({'train': train_input})
```

**K 値の選択方法:**
- エルボー法（Elbow Method）
- シルエット分析（Silhouette Analysis）
- ギャップ統計量（Gap Statistic）
- Bayesian Information Criterion (BIC)
- クロスバリデーション

**最適化バリエーション:**
- K-means++（初期化の改善）
- Mini-batch K-means（大規模データ向け）
- K-medoids（外れ値に頑健）
- Spherical K-means（方向性データ向け）
- Bisecting K-means（階層的アプローチ）

**ユースケース:**
- 顧客セグメンテーション
- 異常検知
- 画像圧縮（色量子化）
- 文書クラスタリング
- IoT センサーデータのグループ化
- 地理的クラスタリング

## LightGBM
勾配ブースティングフレームワークの一種で、高速で効率的なモデルトレーニングを実現します。大規模データセットでも優れたパフォーマンスを発揮します。

**主な特徴:**
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- 葉優先（leaf-wise）の木成長戦略
- カテゴリ変数の効率的な処理
- 分散トレーニングのサポート

**他の勾配ブースティングとの比較:**
- **XGBoost**: レベル優先（level-wise）の木成長、より多くのメモリを使用
- **CatBoost**: カテゴリ変数に特化、順序付きブースティング
- **LightGBM**: 葉優先の木成長、メモリ効率が高い、高速

**主要パラメータ:**
- `num_leaves`: 葉の最大数（木の複雑さを制御）
- `learning_rate`: 学習率
- `max_depth`: 木の最大深さ
- `min_data_in_leaf`: 葉ノードの最小データポイント数
- `feature_fraction`: 特徴のサブサンプリング率
- `bagging_fraction`: データのサブサンプリング率
- `lambda_l1`, `lambda_l2`: L1/L2 正則化パラメータ

**AWS での実装:**
- SageMaker スクリプトモードでのカスタム実装
- SageMaker Processing ジョブでの前処理と統合
- AWS Batch での大規模トレーニング
- AWS Lambda での軽量推論
- Amazon SageMaker Neo での最適化

**実装例:**
```python
import lightgbm as lgb
from sagemaker.sklearn.estimator import SKLearn

# LightGBM スクリプトの作成
script_path = 'lightgbm_script.py'

# SKLearn エスティメータの設定
sklearn_estimator = SKLearn(
    entry_point=script_path,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    framework_version='0.23-1',
    hyperparameters={
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'objective': 'binary',
        'metric': 'auc'
    }
)

# トレーニングの実行
sklearn_estimator.fit({'train': train_input})
```

**最適化テクニック:**
- ハイパーパラメータチューニング
- 特徴の重要度に基づく選択
- 早期停止の活用
- 学習率スケジューリング
- カテゴリ変数の適切なエンコーディング
- 並列処理の最適化

**ユースケース:**
- クリックスルー率（CTR）予測
- 金融市場予測
- 異常検知
- ランキングシステム
- 推薦エンジン
- 時系列予測
- 顧客行動予測
