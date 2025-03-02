# AWS機械学習関連用語解説

## シャドウテスト

シャドウテストは、新しいシステムや機能を本番環境で実際のトラフィックを使って安全に評価するテスト手法です。実際のユーザートラフィックを既存の本番システム（プライマリ）と新システム（シャドウ）の両方に複製して送信し、ユーザーに影響を与えることなく新システムの性能を評価します。

### 主な特徴

- **リスク軽減**: ユーザーエクスペリエンスに影響を与えずにテスト可能
- **実環境データ**: 実際のユーザートラフィックを使用した現実的な評価
- **並行処理**: 既存システムと新システムが同時に動作
- **比較分析**: 両システムの応答を直接比較可能
- **段階的導入**: 本番移行前のリスク評価と問題特定

### シャドウテストの仕組み

1. **トラフィック複製**: 本番環境への実際のリクエストを複製
2. **並行処理**: 複製されたリクエストを新システムに送信
3. **応答比較**: 両システムの応答を比較・分析
4. **メトリクス収集**: レイテンシー、エラー率、リソース使用率などを測定
5. **結果の評価**: 新システムの本番環境への準備状況を判断

### 実装アーキテクチャ

#### AWS上での一般的な実装パターン
```
ユーザー → API Gateway → Lambda (トラフィック分配)
                          ├→ 既存システム → ユーザーに応答
                          └→ 新システム → 応答を記録（ユーザーには返さない）
```

#### 主要コンポーネント
- **トラフィック複製層**: AWS Lambda、Amazon API Gateway、Amazon EventBridge
- **モニタリング**: Amazon CloudWatch、Amazon OpenSearch Service
- **分析**: Amazon Athena、Amazon QuickSight
- **ストレージ**: Amazon S3（テスト結果の保存）

### 機械学習システムでの応用

- **モデル比較**: 既存モデルと新モデルの予測精度の比較
- **パフォーマンス評価**: 推論レイテンシーとスループットの測定
- **リソース使用率**: 計算リソースとメモリ使用量の比較
- **エラー分析**: 新モデルの潜在的な問題やエッジケースの特定
- **ビジネスメトリクス**: コンバージョン率や推奨の質などの評価

### 実装上の考慮事項

1. **リソース計画**:
   - 複製されたトラフィックを処理するための十分なリソース確保
   - コスト管理（テスト中は実質的に2倍のリソースが必要）

2. **データ整合性**:
   - ステートフルな操作の適切な処理
   - 外部システムへの副作用の防止（二重課金など）

3. **モニタリング戦略**:
   - 詳細なメトリクス収集
   - アラートの設定
   - 異常検出の実装

4. **テスト期間**:
   - 十分なデータ収集のための適切な期間設定
   - トラフィックパターンの変動を考慮（平日/週末、昼/夜など）

### 利点

- 実際のユーザーデータに基づく現実的な評価
- 本番環境への影響なしでの新機能テスト
- 予期せぬ問題の早期発見
- データに基づく導入判断
- ロールバックの必要性の低減

シャドウテストは、特に機械学習モデルのような複雑なシステムの評価において、制御された環境でのテストでは発見できない問題を特定するための強力な手法です。本番環境の実際のトラフィックパターンとデータ分布を使用することで、より信頼性の高い評価結果を得ることができます。

## Amazon SageMaker のシャドウテスト

Amazon SageMaker のシャドウテストは、機械学習モデルの新バージョンを本番環境で安全に評価するための専用機能です。実際のエンドポイントへのリクエストを複製して新モデルバージョンに送信し、両者の応答を比較することで、ユーザーエクスペリエンスに影響を与えることなく新モデルの性能を評価できます。

### 主な特徴

- **本番トラフィックの活用**: 実際のユーザーリクエストを使用した現実的な評価
- **ゼロインパクト**: エンドユーザーに影響を与えないテスト
- **自動比較**: 本番モデルと新モデルの応答の自動比較
- **詳細なメトリクス**: レイテンシー、エラー率、予測分布などの包括的な分析
- **段階的デプロイとの統合**: カナリアデプロイやブルー/グリーンデプロイへの移行が容易

### 実装方法

#### シャドウバリアントの設定
```python
import boto3
from sagemaker.model_monitor import CaptureConfig

# シャドウバリアントの設定
shadow_variant_config = {
    'ExistingVariantName': 'Primary',  # 既存の本番バリアント
    'ShadowVariantName': 'Shadow',     # 新しいシャドウバリアント
    'ShadowVariantWeight': 0           # ウェイト0でユーザーにレスポンスを返さない
}

# エンドポイント設定の更新
client = boto3.client('sagemaker')
response = client.update_endpoint_weights_and_capacities(
    EndpointName='my-ml-endpoint',
    DesiredWeightsAndCapacities=[
        {
            'VariantName': 'Primary',
            'DesiredWeight': 100,
            'DesiredInstanceCount': 2
        },
        {
            'VariantName': 'Shadow',
            'DesiredWeight': 0,
            'DesiredInstanceCount': 2
        }
    ],
    ShadowVariantConfig=shadow_variant_config
)
```

#### データキャプチャの設定
```python
# データキャプチャ設定
capture_config = CaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://my-bucket/shadow-test-data/'
)

# エンドポイントの更新
client.update_endpoint(
    EndpointName='my-ml-endpoint',
    DataCaptureConfig=capture_config
)
```

### モニタリングと分析

#### 主要メトリクス
- **予測の一致率**: 本番モデルとシャドウモデルの予測の一致度
- **レイテンシー比較**: 応答時間の差異
- **エラー率**: シャドウモデルのエラー発生頻度
- **リソース使用率**: CPU、メモリ、GPU使用率の比較
- **ビジネスメトリクス**: コンバージョン率や推奨の質などの予測評価

#### CloudWatch統合
```python
# CloudWatchメトリクスの設定例
client.put_metric_data(
    Namespace='ShadowTest/ModelComparison',
    MetricData=[
        {
            'MetricName': 'PredictionDifference',
            'Value': difference_percentage,
            'Unit': 'Percent',
            'Dimensions': [
                {
                    'Name': 'EndpointName',
                    'Value': 'my-ml-endpoint'
                },
                {
                    'Name': 'Comparison',
                    'Value': 'Primary-vs-Shadow'
                }
            ]
        }
    ]
)
```

### 実装パターン

1. **段階的評価**:
   - 初期段階: 少量のトラフィック（例：10%）をシャドウテスト
   - 中間段階: 問題がなければトラフィック量を増加（例：50%）
   - 最終段階: 全トラフィックでのシャドウテスト

2. **A/B/Nテストとの組み合わせ**:
   - 複数の新モデルバージョンを同時にシャドウテスト
   - 最も性能の良いモデルを本番環境に昇格

3. **自動評価と昇格**:
   - 事前定義された基準に基づく自動評価
   - 基準を満たした場合の自動デプロイ
   - CI/CDパイプラインとの統合

### ユースケース

- **モデルアップデート**: 既存モデルの改良版の評価
- **アルゴリズム変更**: 異なるアルゴリズムの比較
- **インフラ最適化**: 異なるインスタンスタイプやコンフィグの評価
- **特徴量エンジニアリング**: 新しい特徴量セットの効果検証
- **コスト最適化**: 軽量モデルへの移行評価

### ベストプラクティス

- **十分なテスト期間**: 様々なトラフィックパターンをカバーする期間設定
- **包括的なメトリクス**: 技術的指標とビジネス指標の両方を評価
- **自動アラート**: 重大な差異の検出時に通知
- **段階的アプローチ**: 小規模から始めて徐々に拡大
- **詳細な記録**: テスト結果の詳細な記録と分析

Amazon SageMaker のシャドウテストは、機械学習モデルの本番環境への安全な移行を実現する強力な機能です。実際のユーザートラフィックを使用した現実的な評価により、新モデルの導入に伴うリスクを大幅に軽減し、データに基づいた意思決定を可能にします。

## SageMaker Debugger

SageMaker Debuggerは、Amazon SageMakerにおける機械学習モデルのトレーニングプロセスをリアルタイムでモニタリングし、問題を検出するための強力なツールです。トレーニング中のテンソル、パラメータ、勾配などの内部状態を可視化し、一般的な問題（勾配消失・爆発、過学習など）を自動的に検出することで、モデル開発の効率と品質を向上させます。

### 主な機能

- **リアルタイムモニタリング**: トレーニング中のモデル内部状態の継続的な監視
- **自動問題検出**: 一般的なトレーニング問題の自動検出と通知
- **テンソル分析**: 重み、勾配、活性化などのテンソルデータの詳細分析
- **システムメトリクス**: CPU、GPU、メモリ使用率などのハードウェアリソース監視
- **カスタムルール**: 特定のユースケースに合わせた独自の監視ルールの作成

### 対応する問題タイプ

#### トレーニング問題
- **勾配消失/爆発**: 勾配値が極端に小さいまたは大きい状態の検出
- **過学習/過少学習**: トレーニングと検証のロス差の監視
- **パラメータ更新問題**: 重みの更新が停滞または不安定な状態の検出
- **活性化関数の飽和**: ニューロンの活性化値の分布異常の検出
- **学習率の問題**: 学習率が大きすぎる/小さすぎる状態の検出

#### システム問題
- **GPU使用率の低下**: 計算リソースの非効率的な使用の検出
- **I/Oボトルネック**: データロードの遅延による処理効率低下の検出
- **メモリリーク**: トレーニング中のメモリ使用量の異常な増加の検出
- **バッチサイズの問題**: 最適でないバッチサイズによる非効率の検出

### 実装方法

#### 基本的な設定
```python
import sagemaker
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, CollectionConfig

# デバッガーフックの設定
hook_config = DebuggerHookConfig(
    s3_output_path='s3://my-bucket/debugger-output',
    collection_configs=[
        CollectionConfig(name="weights"),
        CollectionConfig(name="gradients", save_interval=100),
        CollectionConfig(name="losses", save_interval=50)
    ]
)

# 組み込みルールの設定
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    Rule.sagemaker(rule_configs.loss_not_decreasing())
]

# トレーニングジョブの設定
estimator = sagemaker.estimator.Estimator(
    image_uri='...',
    role='...',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    # デバッガー設定を追加
    debugger_hook_config=hook_config,
    rules=rules
)

estimator.fit(...)
```

#### カスタムルールの作成
```python
from sagemaker.debugger import Rule, CollectionConfig

# カスタムルールスクリプト（別ファイルで定義）
custom_rule = Rule.custom(
    name='CustomGradientRule',
    image_uri='...',  # カスタムルール用のコンテナイメージ
    instance_type='ml.m5.xlarge',
    source='s3://my-bucket/rules/custom_gradient_rule.py',
    rule_parameters={
        'threshold': '0.01',
        'base_trial': 'current'
    },
    collections_to_analyze=[
        CollectionConfig(name="gradients")
    ]
)

# トレーニングジョブにカスタムルールを追加
estimator = sagemaker.estimator.Estimator(
    # ... 他の設定 ...
    rules=[custom_rule]
)
```

### データ分析とビジュアライゼーション

#### SMDebugライブラリを使用した分析
```python
from smdebug.trials import create_trial

# デバッグデータの読み込み
trial = create_trial('s3://my-bucket/debugger-output/trial-name')

# 利用可能なテンソルの確認
tensor_names = trial.tensor_names()
print(tensor_names)

# 特定のテンソルデータの取得
weights = trial.tensor('weights').value(step_num=100)

# 勾配の統計情報の分析
gradients = trial.tensor('gradients')
for step in trial.steps():
    grad_val = gradients.value(step)
    print(f"Step {step}: Mean={grad_val.mean()}, Std={grad_val.std()}")
```

#### SageMaker Studioでの可視化
SageMaker Studioは、Debuggerが収集したデータを視覚的に分析するための組み込みツールを提供します：
- テンソル分布のヒストグラム
- 重みと勾配の時系列プロット
- ヒートマップによるパラメータ変化の可視化
- システムメトリクスのダッシュボード

### 対応フレームワーク

- TensorFlow
- PyTorch
- MXNet
- XGBoost
- SageMaker独自のアルゴリズム

### ユースケース

1. **モデル開発の効率化**:
   - トレーニング問題の早期発見と修正
   - ハイパーパラメータ調整の効率化
   - モデルアーキテクチャの最適化

2. **コスト最適化**:
   - 無駄なトレーニング時間の削減
   - 計算リソースの効率的な使用
   - 早期停止による節約

3. **モデル品質の向上**:
   - 過学習の防止
   - 収束性の改善
   - エッジケースの特定と対応

4. **教育と知識共有**:
   - チーム内でのベストプラクティス共有
   - モデル動作の理解促進
   - デバッグ手法の標準化

### ベストプラクティス

- **段階的アプローチ**: 基本的なメトリクスから始めて徐々に詳細化
- **保存間隔の最適化**: 重要なテンソルのみを適切な間隔で保存
- **カスタムルールの活用**: 特定のモデルやドメインに合わせたルールの作成
- **自動化との統合**: CI/CDパイプラインへのデバッガー統合
- **チーム共有**: デバッグ結果の共有と協力的な問題解決

SageMaker Debuggerは、機械学習モデルのトレーニングにおける「ブラックボックス」問題を解決し、開発者がモデルの内部動作を理解し最適化するための強力なツールです。適切に活用することで、トレーニング時間の短縮、モデル品質の向上、コスト削減を実現できます。

## 一括トラフィックシフト（All At Once トラフィックシフト）

一括トラフィックシフト（All At Once トラフィックシフト）は、機械学習モデルを含むアプリケーションの新バージョンをデプロイする際に、すべてのトラフィックを一度に現行バージョンから新バージョンに切り替える方法です。シンプルかつ迅速なデプロイ方法ですが、新バージョンに問題があった場合のリスクも高くなります。

### 主な特徴

- **即時切り替え**: 全トラフィックを一度に新バージョンへ移行
- **シンプルな実装**: 複雑な設定や段階的な移行ロジックが不要
- **明確な状態**: システムが常に単一バージョンで動作
- **迅速なデプロイ**: 新機能やバグ修正を素早く展開可能
- **リソース効率**: 一時的な重複リソースが最小限

### 実装方法

#### SageMakerエンドポイントでの実装
```python
import boto3

# SageMakerクライアントの初期化
client = boto3.client('sagemaker')

# 新しいエンドポイント設定の作成
response = client.create_endpoint_config(
    EndpointConfigName='new-model-config-v2',
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': 'new-model-v2',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 2,
            'InitialVariantWeight': 1.0
        }
    ]
)

# 既存エンドポイントの更新（一括トラフィックシフト）
response = client.update_endpoint(
    EndpointName='my-production-endpoint',
    EndpointConfigName='new-model-config-v2',
    RetainAllVariantProperties=False
)
```

#### AWS Lambda + API Gatewayでの実装
```python
# AWS CDKを使用した例
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    core
)

class AllAtOnceDeploymentStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # 新しいLambda関数の定義
        new_lambda = lambda_.Function(
            self, 'NewVersionHandler',
            runtime=lambda_.Runtime.PYTHON_3_8,
            code=lambda_.Code.from_asset('lambda'),
            handler='new_version.handler'
        )
        
        # API Gatewayの既存のルートを新しいLambdaに一括で切り替え
        api = apigw.RestApi(self, 'MyApi')
        api.root.add_method('ANY', apigw.LambdaIntegration(new_lambda))
```

### 適したシナリオ

- **開発/テスト環境**: 本番前の検証環境での迅速なイテレーション
- **十分にテスト済みの変更**: 事前に徹底的に検証された更新
- **重要度の低いシステム**: 一時的な問題が許容されるシステム
- **ロールバック体制が整っている場合**: 迅速に前バージョンに戻せる体制
- **トラフィックが少ない時間帯**: 影響を最小化できるタイミング

### リスク軽減策

1. **事前検証の徹底**:
   - 包括的な自動テスト
   - 負荷テストと性能検証
   - セキュリティスキャン

2. **モニタリングの強化**:
   - リアルタイムアラートの設定
   - 主要メトリクスの継続的監視
   - 異常検出の実装

3. **ロールバック計画**:
   - 自動ロールバックトリガーの設定
   - 前バージョンの保持
   - ロールバック手順の文書化と訓練

4. **段階的な展開**:
   - 地域やアベイラビリティゾーン単位での段階的展開
   - カナリアリリースとの組み合わせ

### メリットとデメリット

#### メリット
- デプロイプロセスのシンプルさ
- 迅速な展開と検証
- リソース使用の効率性
- 設定の複雑さの低減
- 明確なバージョン状態

#### デメリット
- 問題発生時の広範な影響
- ロールバック時のダウンタイムリスク
- 段階的な検証ができない
- 全ユーザーへの同時影響

### 機械学習特有の考慮事項

- **モデル互換性**: 入出力形式の変更による互換性問題
- **パフォーマンス変化**: 新モデルの推論時間やリソース要件の変化
- **予測分布の変化**: 新モデルによる予測分布の急激な変化
- **依存サービスへの影響**: 下流のシステムへの影響評価

### ベストプラクティス

- **デプロイウィンドウの設定**: 影響を最小化できる時間帯の選択
- **事前通知**: 関係者への事前通知と準備
- **リハーサル**: 本番環境に似たステージング環境でのデプロイリハーサル
- **ロールバックテスト**: ロールバック手順の事前検証
- **段階的な機能有効化**: デプロイ後の機能フラグによる段階的な機能有効化

一括トラフィックシフトは、適切なリスク管理と準備を行うことで、シンプルかつ効率的なデプロイ方法として活用できます。特に小規模なチームや、迅速な展開が求められるシナリオにおいて有効ですが、重要なシステムでは追加の安全策との組み合わせが推奨されます。

## ブルー/グリーンデプロイ

ブルー/グリーンデプロイは、新環境（グリーン）を準備し、テスト後に全トラフィックを現行環境（ブルー）から切り替えるデプロイ方法です。この手法により、ダウンタイムを最小限に抑えながら、安全に新バージョンをリリースすることができます。特に機械学習モデルのような複雑なシステムのデプロイに適しています。

### 主な特徴

- **並行環境**: 現行環境（ブルー）と新環境（グリーン）を並行して運用
- **完全な検証**: 本番と同等の環境で新バージョンを事前検証
- **即時切り替え**: トラフィックの瞬時の切り替えによるダウンタイム最小化
- **簡単なロールバック**: 問題発生時に元の環境へ迅速に戻せる
- **リスク軽減**: 新バージョンの問題が全ユーザーに影響する前に検出可能

### 実装アーキテクチャ

#### AWS上での一般的な実装パターン
```
                      ┌─────────────────┐
                      │                 │
ユーザー → Route 53 → │ ロードバランサー │
                      │                 │
                      └─────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
        ┌────────▼────────┐     ┌────────▼────────┐
        │                 │     │                 │
        │  ブルー環境      │     │  グリーン環境    │
        │ (現行バージョン)  │     │ (新バージョン)   │
        │                 │     │                 │
        └─────────────────┘     └─────────────────┘
```

### SageMakerでの実装方法

#### 1. グリーン環境（新モデル）の準備
```python
import boto3
import sagemaker

# 新モデルのデプロイ（グリーン環境）
sagemaker_client = boto3.client('sagemaker')

# 新しいモデルの作成
sagemaker_client.create_model(
    ModelName='my-model-green',
    ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    PrimaryContainer={
        'Image': '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-model:latest',
        'ModelDataUrl': 's3://my-bucket/model-artifacts/new-version/model.tar.gz'
    }
)

# 新しいエンドポイント設定の作成
sagemaker_client.create_endpoint_config(
    EndpointConfigName='my-endpoint-config-green',
    ProductionVariants=[
        {
            'VariantName': 'green',
            'ModelName': 'my-model-green',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 2
        }
    ]
)

# グリーン環境のエンドポイント作成
sagemaker_client.create_endpoint(
    EndpointName='my-endpoint-green',
    EndpointConfigName='my-endpoint-config-green'
)
```

#### 2. グリーン環境のテストと検証
```python
# グリーン環境のテスト
runtime_client = boto3.client('sagemaker-runtime')

# テストデータでの推論
response = runtime_client.invoke_endpoint(
    EndpointName='my-endpoint-green',
    ContentType='application/json',
    Body='{"data": [1, 2, 3, 4, 5]}'
)

# 結果の検証
result = response['Body'].read().decode()
print(f"Green environment test result: {
