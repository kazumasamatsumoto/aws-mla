# AWS機械学習関連用語解説

## AWS Lake Formation
データレイクの設定、セキュリティ、管理を簡素化するサービスです。一元的なアクセス制御と統合されたデータカタログを提供します。

**主な特徴:**
- データレイクの迅速な構築と管理
- きめ細かなアクセス制御
- 中央集権的なセキュリティとガバナンス
- ブループリントを使用した自動データ取り込み
- AWS Glue との緊密な統合
- 機械学習ワークフローのデータ準備を効率化

**アーキテクチャコンポーネント:**
- データカタログ: メタデータリポジトリ
- セキュリティ管理: きめ細かなアクセス制御
- ブループリント: データ取り込みワークフローテンプレート
- データフィルタリング: 行レベル、列レベルのセキュリティ
- データ品質管理: データ検証と監視

**セットアップ手順:**
1. AWS Lake Formation コンソールでデータレイク管理者を設定
2. データカタログデータベースとテーブルを作成
3. データソースを定義し、クローラを設定
4. アクセス許可とポリシーを構成
5. ブループリントを使用してデータ取り込みワークフローを自動化
6. 分析サービスとの統合を設定

**AWS サービスとの統合:**
- Amazon S3: データストレージ
- AWS Glue: ETL 処理とカタログ
- Amazon Athena: SQL クエリ
- Amazon Redshift: データウェアハウス
- Amazon EMR: 大規模データ処理
- Amazon QuickSight: ビジネスインテリジェンス
- Amazon SageMaker: 機械学習モデル開発

**セキュリティ機能:**
- IAM との統合
- 列レベルのセキュリティ
- 行レベルのフィルタリング
- タグベースのアクセス制御
- クロスアカウント共有
- 暗号化オプション

**機械学習ワークフローでの活用:**
- 特徴量エンジニアリングのためのデータ準備
- トレーニングデータセットの作成と管理
- データ系統の追跡と監査
- モデル開発のためのデータ探索
- 複数チーム間でのデータ共有
- データ品質の確保とモニタリング

**ユースケース:**
- 機械学習モデル開発のためのデータレイク
- 規制対象データの安全な分析
- 部門間データ共有と協業
- ビッグデータ分析プラットフォーム
- IoT データの集約と分析
- データサイエンスプラットフォーム

## Amazon SageMaker Feature Store
機械学習特徴量を保存、更新、取得、共有するための専用リポジトリです。特徴量の一貫性と再利用性を確保します。

**主な特徴:**
- オンラインストアとオフラインストアの二重アーキテクチャ
- 特徴量の一元管理とバージョン管理
- 低レイテンシのオンライン推論サポート
- 特徴量の再利用と共有
- 時間ポイント検索機能
- データの一貫性と系統追跡

**アーキテクチャ:**
- **オンラインストア**: 低レイテンシの読み取り用に最適化された高可用性ストア
- **オフラインストア**: S3 に保存された履歴データ、トレーニング用
- **特徴量グループ**: 関連する特徴量の論理的なコレクション
- **レコード**: タイムスタンプ付きの特徴量値のセット

**実装例:**
```python
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# SageMaker セッションの作成
session = sagemaker.Session()
region = session.boto_region_name
s3_bucket_name = session.default_bucket()

# 特徴量グループの定義
feature_group_name = "customer-churn-features"
feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)

# 特徴量定義の追加
feature_group.load_feature_definitions(data_frame[feature_definitions])

# 特徴量グループの作成
feature_group.create(
    s3_uri=f"s3://{s3_bucket_name}/feature-store/{feature_group_name}",
    record_identifier_name="customer_id",
    event_time_feature_name="timestamp",
    role_arn=role,
    enable_online_store=True
)

# データの取り込み
feature_group.ingest(data_frame)
```

**ワークフロー統合:**
- SageMaker Studio との統合
- SageMaker Pipelines でのモデル開発
- SageMaker Processing ジョブでの特徴量生成
- SageMaker Model Monitor との連携
- AWS Glue ETL ジョブからの特徴量生成
- Amazon EMR での大規模特徴量計算

**ユースケース:**
- リアルタイム推論のための特徴量提供
- 特徴量の再利用と共有
- 時間ポイント検索による再現可能なトレーニングデータセット
- 特徴量の一貫性確保
- 特徴量のバージョン管理と監査
- クロスチーム特徴量共有

## Amazon SageMaker Data Wrangler
データ準備と特徴量エンジニアリングを視覚的に行うためのツールです。データの前処理、変換、分析を効率化します。

**主な特徴:**
- 300以上の組み込みデータ変換
- カスタム変換のサポート
- データ品質と統計情報の可視化
- データフロー管理
- 自動データ型推論
- コード生成機能

**サポートするデータソース:**
- Amazon S3
- Amazon Athena
- Amazon Redshift
- AWS Lake Formation
- Amazon EMR
- Snowflake
- データベース (via JDBC)

**組み込み変換:**
- 欠損値処理
- 外れ値検出と処理
- 特徴量エンコーディング
- 特徴量選択
- テキスト特徴量抽出
- 時系列特徴量生成
- 数値特徴量のビニング
- 標準化とスケーリング

**データ分析機能:**
- データプロファイリング
- ターゲット漏洩検出
- 特徴量相関分析
- 時系列可視化
- クラスタリング分析
- 異常検出

**ワークフロー統合:**
- SageMaker Processing ジョブへのエクスポート
- SageMaker Pipelines との統合
- SageMaker Feature Store への特徴量登録
- Python コードへのエクスポート
- AWS Glue ETL ジョブへの変換

**ユースケース:**
- 機械学習モデルのデータ準備
- 探索的データ分析
- 特徴量エンジニアリング
- データクレンジングと変換
- データ品質評価
- 再現可能なデータ準備パイプライン

## Amazon SageMaker Clarify
機械学習モデルのバイアス検出と説明可能性を提供するサービスです。モデルの公平性と透明性を向上させます。

**主な特徴:**
- トレーニング前のデータバイアス検出
- トレーニング後のモデルバイアス検出
- モデル説明可能性（グローバルとローカル）
- 特徴量重要度の計算
- 部分依存プロット (PDP)
- SHAP 値による説明

**バイアス指標:**
- Class Imbalance (CI)
- Difference in Positive Proportions in Predicted Labels (DPPL)
- Disparate Impact (DI)
- Conditional Demographic Disparity (CDD)
- Accuracy Difference (AD)
- Treatment Equality (TE)
- Conditional Demographic Disparity in Labels (CDDL)

**説明可能性手法:**
- SHAP (SHapley Additive exPlanations)
- 部分依存プロット (PDP)
- 特徴量重要度
- グローバル説明
- ローカル（インスタンスレベル）説明

**実装方法:**
```python
from sagemaker import clarify

# Clarify プロセッサの設定
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    sagemaker_session=session
)

# バイアス設定
bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],
    facet_name='age',
    facet_values_or_threshold=[40],
    group_name='gender'
)

# SHAP 設定
shap_config = clarify.SHAPConfig(
    baseline=[baseline_data],
    num_samples=100,
    agg_method='mean_abs'
)

# 分析の実行
clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
```

**ワークフロー統合:**
- SageMaker モデルモニタリング
- SageMaker Pipelines
- SageMaker Studio での可視化
- モデルガバナンスフレームワーク
- CI/CD パイプライン

**ユースケース:**
- 規制対象 ML モデルの透明性確保
- モデルの公平性評価と改善
- モデル開発プロセスの品質向上
- ステークホルダーへの説明責任
- モデル動作の理解と改善
- バイアス軽減戦略の開発

## Amazon SageMaker Model Monitor
本番環境のモデルを継続的に監視し、品質の低下を検出するサービスです。データドリフトやモデルドリフトを早期に発見します。

**主な特徴:**
- データ品質モニタリング
- モデル品質モニタリング
- バイアスドリフトモニタリング
- 特徴量アトリビューションドリフトモニタリング
- カスタムモニタリング
- 自動アラート設定

**モニタリングタイプ:**
1. **データ品質モニタリング**:
   - 入力データの統計的プロパティの変化を検出
   - スキーマドリフト、特徴量分布の変化を監視
   - 欠損値、外れ値の増加を検出

2. **モデル品質モニタリング**:
   - モデルの精度、AUC などの性能指標を追跡
   - グラウンドトゥルースラベルとの比較
   - 時間経過による性能低下を検出

3. **バイアスドリフト**:
   - 保護属性に関するバイアス指標の変化を監視
   - 公平性の経時的変化を追跡
   - SageMaker Clarify との統合

4. **特徴量アトリビューション**:
   - 特徴量の重要度の変化を監視
   - モデル説明可能性の変化を追跡
   - SHAP 値の経時的変化を分析

**設定手順:**
1. ベースライン計算ジョブの作成
2. モニタリングスケジュールの設定
3. 制約としきい値の定義
4. アラートの設定
5. 結果の分析とレポート確認

**実装例:**
```python
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# データキャプチャの設定
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=s3_capture_upload_path
)

# モデルのデプロイ（データキャプチャ有効）
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
)

# モニターの設定
my_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# ベースラインの作成
my_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DatasetFormat.csv()
)

# モニタリングスケジュールの作成
my_monitor.create_monitoring_schedule(
    monitor_schedule_name=schedule_name,
    endpoint_input=predictor.endpoint_name,
    statistics=my_monitor.baseline_statistics(),
    constraints=my_monitor.suggested_constraints(),
    schedule_cron_expression=cron_expression
)
```

**統合機能:**
- Amazon CloudWatch アラーム
- Amazon SNS 通知
- AWS Lambda による自動対応
- SageMaker Pipelines との統合
- SageMaker Experiments での追跡
- Amazon EventBridge によるイベント処理

**ユースケース:**
- 本番環境モデルの品質保証
- データドリフトの早期検出
- コンセプトドリフトへの対応
- 規制要件への準拠
- モデル再トレーニングの自動化トリガー
- モデルパフォーマンスの継続的改善
