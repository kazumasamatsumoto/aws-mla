# AWS機械学習関連用語解説

## カナリアトラフィックシフト

カナリアトラフィックシフトは、トラフィックの一部を新バージョンに徐々に移行するデプロイ方法です。この手法は、炭鉱でガス漏れを検知するためにカナリアを使用していた歴史に由来し、新バージョンの問題を早期に検出しながら、影響範囲を限定することができます。機械学習モデルの新バージョンをリスクを最小限に抑えながらデプロイする際に特に有効です。

### 主な特徴

- **段階的な移行**: トラフィックの一部（例：5%、10%、25%...）を段階的に新バージョンに移行
- **リスク軽減**: 問題発生時の影響を限定的にとどめる
- **早期検出**: 本番環境での実際のトラフィックを使用した問題の早期発見
- **データに基づく判断**: 実際のパフォーマンスメトリクスに基づいた展開判断
- **迅速なロールバック**: 問題発生時に少量のトラフィックのみを元に戻せばよい

### SageMakerでの実装方法

#### 1. 初期設定（トラフィックの5%を新モデルに割り当て）
```python
import boto3

# SageMakerクライアントの初期化
client = boto3.client('sagemaker')

# 新しいエンドポイント設定の作成
response = client.create_endpoint_config(
    EndpointConfigName='dual-variant-config',
    ProductionVariants=[
        {
            'VariantName': 'ExistingModel',
            'ModelName': 'existing-model',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 2,
            'InitialVariantWeight': 0.95  # 95%のトラフィック
        },
        {
            'VariantName': 'NewModel',
            'ModelName': 'new-model',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 0.05  # 5%のトラフィック
        }
    ]
)

# エンドポイントの更新
response = client.update_endpoint(
    EndpointName='my-production-endpoint',
    EndpointConfigName='dual-variant-config'
)
```

#### 2. トラフィック比率の段階的な調整
```python
# トラフィック比率を調整（25%に増加）
response = client.update_endpoint_weights_and_capacities(
    EndpointName='my-production-endpoint',
    DesiredWeightsAndCapacities=[
        {
            'VariantName': 'ExistingModel',
            'DesiredWeight': 0.75  # 75%に減少
        },
        {
            'VariantName': 'NewModel',
            'DesiredWeight': 0.25  # 25%に増加
        }
    ]
)
```

#### 3. 完全移行（100%のトラフィックを新モデルに）
```python
# 完全に新モデルへ移行
response = client.update_endpoint_weights_and_capacities(
    EndpointName='my-production-endpoint',
    DesiredWeightsAndCapacities=[
        {
            'VariantName': 'ExistingModel',
            'DesiredWeight': 0.0  # 0%に減少
        },
        {
            'VariantName': 'NewModel',
            'DesiredWeight': 1.0  # 100%に増加
        }
    ]
)
```

### モニタリングと評価

#### 主要メトリクス
- **モデル精度**: 予測の正確さや品質
- **レイテンシー**: 応答時間の変化
- **エラー率**: 新モデルでのエラー発生頻度
- **リソース使用率**: CPU、メモリ、GPU使用率
- **ビジネスメトリクス**: コンバージョン率、ユーザーエンゲージメントなど

#### CloudWatchによるモニタリング
```python
import boto3
from datetime import datetime, timedelta

# CloudWatchクライアントの初期化
cloudwatch = boto3.client('cloudwatch')

# 新旧モデルのエラー率を比較
response = cloudwatch.get_metric_data(
    MetricDataQueries=[
        {
            'Id': 'existing_model_errors',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'ModelError',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': 'my-production-endpoint'
                        },
                        {
                            'Name': 'VariantName',
                            'Value': 'ExistingModel'
                        }
                    ]
                },
                'Period': 300,
                'Stat': 'Sum'
            }
        },
        {
            'Id': 'new_model_errors',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'ModelError',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': 'my-production-endpoint'
                        },
                        {
                            'Name': 'VariantName',
                            'Value': 'NewModel'
                        }
                    ]
                },
                'Period': 300,
                'Stat': 'Sum'
            }
        }
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow()
)
```

### 自動化戦略

#### 段階的な自動デプロイパイプライン
```python
# AWS Step Functionsを使用した例（疑似コード）
{
  "Comment": "Canary Deployment for ML Model",
  "StartAt": "DeployWithInitialWeight",
  "States": {
    "DeployWithInitialWeight": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:updateEndpoint",
      "Parameters": {
        "EndpointName": "my-production-endpoint",
        "EndpointConfigName": "dual-variant-config-5-percent"
      },
      "Next": "WaitAndEvaluate5Percent"
    },
    "WaitAndEvaluate5Percent": {
      "Type": "Wait",
      "Seconds": 1800,
      "Next": "Check5PercentMetrics"
    },
    "Check5PercentMetrics": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:function:evaluate-metrics",
      "Next": "Is5PercentSuccessful"
    },
    "Is5PercentSuccessful": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.metricsOk",
          "BooleanEquals": true,
          "Next": "IncreaseTo25Percent"
        }
      ],
      "Default": "Rollback"
    },
    // 以下、25%、50%、100%と続く...
  }
}
```

### 適したシナリオ

- **ミッションクリティカルなシステム**: 高可用性が求められるシステム
- **大規模ユーザーベース**: 多数のユーザーに影響を与える可能性があるシステム
- **予測困難な問題**: テスト環境では検出できない問題が懸念される場合
- **段階的なフィードバック収集**: ユーザーからのフィードバックを段階的に収集したい場合
- **リソース最適化**: インスタンス数を段階的に調整したい場合

### メリットとデメリット

#### メリット
- リスクの最小化と早期検出
- 実際のトラフィックを使用した検証
- 段階的なスケーリングが可能
- 問題発生時の影響範囲の限定
- データに基づく意思決定

#### デメリット
- デプロイ時間の長期化
- 複数バージョンの同時運用による複雑さ
- リソースの一時的な増加
- バージョン間の整合性維持の必要性

### ベストプラクティス

- **明確な評価基準**: 次の段階に進むための明確なメトリクスと閾値の設定
- **自動ロールバック**: 問題検出時の自動ロールバック機構の実装
- **段階的なスケジュール**: トラフィック増加のタイミングと割合の事前計画
- **詳細なモニタリング**: 各段階での詳細なメトリクス収集と分析
- **ユーザーセグメンテーション**: 特定のユーザーグループから段階的に展開する戦略

カナリアトラフィックシフトは、特に機械学習モデルのような複雑なシステムのデプロイにおいて、リスクを最小限に抑えながら新バージョンを安全に展開するための効果的な方法です。適切に実装することで、ユーザーエクスペリエンスを維持しながら、継続的な改善とイノベーションを実現することができます。

## AWS::SageMaker::NotebookInstance

AWS::SageMaker::NotebookInstanceは、AWS CloudFormationを使用してAmazon SageMakerノートブックインスタンスをインフラストラクチャとしてコード（IaC）で定義するためのリソースタイプです。これにより、Jupyter Notebookベースの機械学習開発環境を自動的かつ一貫性を持って作成・管理することができます。

### 主な特徴

- **インフラストラクチャのコード化**: ノートブックインスタンスの設定を宣言的に定義
- **バージョン管理**: 環境設定の変更履歴を追跡可能
- **再現性**: 同一の開発環境を複数のプロジェクトやチームで再現可能
- **自動化**: CI/CDパイプラインとの統合による自動プロビジョニング
- **コンプライアンス**: セキュリティ設定の標準化と監査

### 基本的な定義例

```yaml
Resources:
  MyNotebookInstance:
    Type: AWS::SageMaker::NotebookInstance
    Properties:
      NotebookInstanceName: ml-research-notebook
      InstanceType: ml.t3.medium
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
      SubnetId: !Ref PrivateSubnet1
      SecurityGroupIds: 
        - !Ref NotebookSecurityGroup
      VolumeSizeInGB: 50
      DefaultCodeRepository: https://github.com/myorg/ml-notebooks.git
      LifecycleConfigName: !GetAtt NotebookLifecycleConfig.NotebookInstanceLifecycleConfigName
      DirectInternetAccess: Disabled
      RootAccess: Enabled
      Tags:
        - Key: Project
          Value: CustomerChurn
        - Key: Environment
          Value: Development
```

### 主要なプロパティ

#### 基本設定
- **NotebookInstanceName**: インスタンスの名前（オプション、指定しない場合は自動生成）
- **InstanceType**: コンピューティングリソースのタイプ（例：ml.t3.medium、ml.m5.xlarge）
- **RoleArn**: ノートブックが使用するIAMロールのARN
- **VolumeSizeInGB**: EBSボリュームのサイズ（デフォルト：5GB）

#### ネットワーク設定
- **SubnetId**: ノートブックを配置するVPCサブネットのID
- **SecurityGroupIds**: 適用するセキュリティグループのIDリスト
- **DirectInternetAccess**: インターネットアクセスの有効/無効（Enabled/Disabled）

#### 開発環境設定
- **DefaultCodeRepository**: デフォルトのGitリポジトリURL
- **AdditionalCodeRepositories**: 追加のGitリポジトリURLのリスト
- **LifecycleConfigName**: 起動時に実行するライフサイクル設定の名前
- **RootAccess**: rootアクセスの有効/無効（Enabled/Disabled）

#### 高度な設定
- **AcceleratorTypes**: 使用するアクセラレーター（例：ml.eia1.medium）
- **KmsKeyId**: EBSボリューム暗号化用のKMSキーID
- **PlatformIdentifier**: ノートブックのプラットフォーム（例：notebook-al2-v1）

### ライフサイクル設定の例

ノートブックインスタンスの起動時に自動的に実行されるスクリプトを定義できます：

```yaml
NotebookLifecycleConfig:
  Type: AWS::SageMaker::NotebookInstanceLifecycleConfig
  Properties:
    NotebookInstanceLifecycleConfigName: ml-notebook-setup
    OnStart:
      - Content: !Base64 |
          #!/bin/bash
          set -e
          
          # 必要なパッケージのインストール
          sudo -u ec2-user -i <<EOF
          conda activate python3
          pip install scikit-learn==1.0.2 xgboost==1.5.1 lightgbm==3.3.2
          pip install matplotlib seaborn plotly
          pip install pytest pytest-cov black isort
          EOF
          
          # データディレクトリの作成
          mkdir -p /home/ec2-user/SageMaker/data
          chown -R ec2-user:ec2-user /home/ec2-user/SageMaker/data
```

### 高度な使用例

#### 複数環境の管理（開発/テスト/本番）
```yaml
Parameters:
  Environment:
    Type: String
    Default: Development
    AllowedValues:
      - Development
      - Testing
      - Production
  
  InstanceTypeMap:
    Type: Map
    Default:
      Development: ml.t3.medium
      Testing: ml.m5.xlarge
      Production: ml.m5.4xlarge

Resources:
  EnvironmentNotebook:
    Type: AWS::SageMaker::NotebookInstance
    Properties:
      NotebookInstanceName: !Sub ml-notebook-${Environment}
      InstanceType: !FindInMap [InstanceTypeMap, !Ref Environment, !Ref AWS::NoValue]
      # その他のプロパティ...
```

#### 条件付きリソース作成
```yaml
Conditions:
  IsProductionEnvironment: !Equals [!Ref Environment, Production]

Resources:
  ProductionNotebook:
    Type: AWS::SageMaker::NotebookInstance
    Condition: IsProductionEnvironment
    Properties:
      # 本番環境特有の設定...
```

### セキュリティのベストプラクティス

1. **VPC内配置**:
   ```yaml
   SubnetId: !Ref PrivateSubnet
   DirectInternetAccess: Disabled
   ```

2. **暗号化の有効化**:
   ```yaml
   KmsKeyId: !Ref NotebookEncryptionKey
   ```

3. **最小権限の原則**:
   ```yaml
   RoleArn: !GetAtt MinimalPermissionRole.Arn
   ```

4. **ネットワークアクセス制限**:
   ```yaml
   SecurityGroupIds: 
     - !Ref RestrictedAccessSecurityGroup
   ```

### 運用上の考慮事項

- **コスト管理**: 不要なインスタンスの自動停止設定
- **リソースサイジング**: ワークロードに適したインスタンスタイプの選択
- **バックアップ戦略**: 重要なノートブックの定期的なバックアップ
- **更新管理**: ノートブックインスタンスの定期的な更新とパッチ適用

AWS::SageMaker::NotebookInstanceリソースタイプを使用することで、機械学習開発環境の一貫性、再現性、セキュリティを確保しながら、インフラストラクチャのプロビジョニングと管理を自動化することができます。これにより、データサイエンティストは環境構築よりもモデル開発に集中することができます。

## AWS::SageMaker::Model

AWS::SageMaker::Modelは、AWS CloudFormationを使用してAmazon SageMakerモデルリソースをインフラストラクチャとしてコード（IaC）で定義するためのリソースタイプです。このリソースタイプを使用することで、機械学習モデルのデプロイ設定を宣言的に定義し、バージョン管理、再現性、自動化を実現できます。

### 主な特徴

- **モデル設定のコード化**: モデル設定を宣言的に定義し、バージョン管理
- **コンテナ設定**: 単一または複数のコンテナを使用したモデル定義
- **アーティファクト管理**: S3に保存されたモデルアーティファクトとの連携
- **インフラ自動化**: CI/CDパイプラインとの統合による自動デプロイ
- **セキュリティ設定**: VPC、IAM、KMSなどのセキュリティ設定の一元管理

### 基本的な定義例

```yaml
Resources:
  MyXGBoostModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: customer-churn-prediction-model
      ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
      PrimaryContainer:
        Image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.5-1
        ModelDataUrl: s3://my-models-bucket/xgboost/model.tar.gz
        Environment:
          SAGEMAKER_PROGRAM: inference.py
          MODEL_SERVER_TIMEOUT: "60"
      Tags:
        - Key: Project
          Value: CustomerChurn
        - Key: Environment
          Value: Production
```

### 主要なプロパティ

#### 基本設定
- **ModelName**: モデルの名前（オプション、指定しない場合は自動生成）
- **ExecutionRoleArn**: モデルが使用するIAMロールのARN
- **EnableNetworkIsolation**: ネットワーク分離の有効/無効（true/false）

#### コンテナ設定
- **PrimaryContainer**: 主要なモデルコンテナの設定
  - **Image**: コンテナイメージのURI
  - **ModelDataUrl**: モデルアーティファクトのS3 URI
  - **Environment**: 環境変数のキーと値のマップ
  - **Mode**: コンテナのモード（SingleModel/MultiModel）

- **Containers**: 複数のコンテナを使用する場合のリスト（推論パイプラインなど）

#### ネットワーク設定
- **VpcConfig**: VPC内でモデルを実行するための設定
  - **SecurityGroupIds**: セキュリティグループのIDリスト
  - **Subnets**: サブネットのIDリスト

#### セキュリティ設定
- **InferenceExecutionConfig**: 推論実行の設定
  - **Mode**: 推論実行モード（Serial/Direct）

### 高度な使用例

#### 推論パイプライン（複数コンテナ）
```yaml
MyInferencePipeline:
  Type: AWS::SageMaker::Model
  Properties:
    ModelName: text-classification-pipeline
    ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
    Containers:
      - Image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/text-preprocessor:latest
        ModelDataUrl: s3://my-models-bucket/preprocessor/model.tar.gz
      - Image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/bert-classifier:latest
        ModelDataUrl: s3://my-models-bucket/bert/model.tar.gz
```

#### マルチモデルエンドポイント
```yaml
MyMultiModel:
  Type: AWS::SageMaker::Model
  Properties:
    ModelName: multi-model-endpoint
    ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
    PrimaryContainer:
      Image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.6.3-cpu
      Mode: MultiModel
      ModelDataUrl: s3://my-models-bucket/multi-model/
      Environment:
        SAGEMAKER_PROGRAM: inference.py
        SAGEMAKER_SUBMIT_DIRECTORY: s3://my-models-bucket/code/inference.tar.gz
```

#### VPC内でのモデル実行
```yaml
SecureModel:
  Type: AWS::SageMaker::Model
  Properties:
    ModelName: secure-model
    ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
    PrimaryContainer:
      Image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-model:latest
      ModelDataUrl: s3://my-models-bucket/model.tar.gz
    VpcConfig:
      SecurityGroupIds:
        - !Ref ModelSecurityGroup
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
    EnableNetworkIsolation: true
```

### 条件付きリソース作成
```yaml
Parameters:
  ModelVersion:
    Type: String
    Default: v1
    AllowedValues: [v1, v2]

Conditions:
  IsVersionTwo: !Equals [!Ref ModelVersion, v2]

Resources:
  MyModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Sub customer-churn-${ModelVersion}
      PrimaryContainer:
        Image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest
        ModelDataUrl: !If 
          - IsVersionTwo
          - s3://my-models-bucket/v2/model.tar.gz
          - s3://my-models-bucket/v1/model.tar.gz
```

### セキュリティのベストプラクティス

1. **ネットワーク分離**:
   ```yaml
   EnableNetworkIsolation: true
   ```

2. **VPC内配置**:
   ```yaml
   VpcConfig:
     SecurityGroupIds: [!Ref ModelSecurityGroup]
     Subnets: [!Ref PrivateSubnet1, !Ref PrivateSubnet2]
   ```

3. **最小権限の原則**:
   ```yaml
   ExecutionRoleArn: !GetAtt MinimalPermissionRole.Arn
   ```

4. **暗号化**:
   ```yaml
   # S3バケットポリシーで暗号化を強制
   # KMSキーを使用したモデルアーティファクトの暗号化
   ```

### モデル管理のベストプラクティス

- **命名規則**: 一貫性のある命名規則でモデルを管理
- **タグ付け**: プロジェクト、環境、バージョンなどでタグ付け
- **バージョン管理**: モデルバージョンを明示的に管理
- **ドキュメント化**: モデルのメタデータや性能指標を記録

AWS::SageMaker::Modelリソースタイプを使用することで、機械学習モデルのデプロイ設定を一貫性を持って管理し、インフラストラクチャのプロビジョニングを自動化することができます。これにより、モデルのバージョン管理、再現性、セキュリティが向上し、MLOps（機械学習オペレーション）の効率化が実現します。

## AWS::SageMaker::Pipeline

AWS::SageMaker::Pipelineは、AWS CloudFormationを使用してAmazon SageMaker Pipelinesをインフラストラクチャとしてコード（IaC）で定義するためのリソースタイプです。SageMaker Pipelinesは、機械学習ワークフローを自動化するための完全マネージド型のCI/CDサービスであり、このリソースタイプを使用することで、再現可能で監査可能な機械学習パイプラインを構築できます。

### 主な特徴

- **ワークフロー自動化**: 機械学習ワークフローの全ステップを自動化
- **パイプライン定義のコード化**: パイプライン設定を宣言的に定義
- **バージョン管理**: パイプライン定義の変更履歴を追跡
- **再現性**: 同一のワークフローを一貫して実行可能
- **監査とコンプライアンス**: 各ステップの実行履歴と結果を追跡

### 基本的な定義例

```yaml
Resources:
  MyMLPipeline:
    Type: AWS::SageMaker::Pipeline
    Properties:
      PipelineName: customer-churn-training-pipeline
      PipelineDisplayName: Customer Churn Training Pipeline
      RoleArn: !GetAtt SageMakerPipelineExecutionRole.Arn
      PipelineDefinition:
        PipelineDefinitionBody: |
          {
            "Version": "2020-12-01",
            "Parameters": [
              {
                "Name": "InputDataUrl",
                "Type": "String",
                "DefaultValue": "s3://my-bucket/data/input/customer-churn.csv"
              },
              {
                "Name": "ModelApprovalStatus",
                "Type": "String",
                "DefaultValue": "PendingManualApproval"
              }
            ],
            "Steps": [
              {
                "Name": "ProcessingStep",
                "Type": "Processing",
                "Arguments": {
                  "ProcessingInputs": [...],
                  "ProcessingOutputConfig": {...},
                  "AppSpecification": {...},
                  "ProcessingResources": {...}
                }
              },
              {
                "Name": "TrainingStep",
                "Type": "Training",
                "DependsOn": ["ProcessingStep"],
                "Arguments": {
                  "InputDataConfig": [...],
                  "OutputDataConfig": {...},
                  "ResourceConfig": {...},
                  "AlgorithmSpecification": {...}
                }
              },
              {
                "Name": "EvaluationStep",
                "Type": "Processing",
                "DependsOn": ["TrainingStep"],
                "Arguments": {...}
              },
              {
                "Name": "ModelRegistrationStep",
                "Type": "RegisterModel",
                "DependsOn": ["EvaluationStep"],
                "Arguments": {...}
              }
            ]
          }
      Tags:
        - Key: Project
          Value: CustomerChurn
        - Key: Environment
          Value: Development
```

### 主要なプロパティ

#### 基本設定
- **PipelineName**: パイプラインの名前
- **PipelineDisplayName**: UI表示用のパイプライン名（オプション）
- **RoleArn**: パイプラインが使用するIAMロールのARN

#### パイプライン定義
- **PipelineDefinition**: パイプラインの定義
  - **PipelineDefinitionBody**: JSON形式のパイプライン定義本体
  - **PipelineDefinitionS3Location**: S3に保存されたパイプライン定義ファイルの場所

#### パイプライン定義の主要コンポーネント
- **Parameters**: パイプラインのパラメータ定義
- **Steps**: パイプラインのステップ定義
  - **Processing**: データ処理ステップ
  - **Training**: モデルトレーニングステップ
  - **Condition**: 条件分岐ステップ
  - **RegisterModel**: モデル登録ステップ
  - **Transform**: バッチ変換ステップ
  - **Callback**: カスタムコードを実行するコールバックステップ

### 高度な使用例

#### 外部ファイルからのパイプライ
