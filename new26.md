# AWS機械学習関連用語解説

## C シリーズ

C シリーズは、コンピューティング最適化インスタンスファミリーで、高性能コンピューティングや計算集約型アプリケーションに適しています。C6i、C6a、C7g などの世代があり、Intel Xeon、AMD EPYC、AWS Graviton プロセッサを搭載しています。機械学習のデータ前処理、特徴エンジニアリング、バッチ推論などに適しています。

### 主な特徴

- **高いコンピューティング性能**: vCPU あたりの計算能力が最適化されたインスタンス
- **コスト効率**: 計算集約型ワークロードに対する優れたコストパフォーマンス
- **スケーラビリティ**: 最大 128 vCPU と 256 GiB のメモリをサポート
- **高速ネットワーク**: 最大 30 Gbps のネットワーク帯域幅
- **ストレージ最適化**: EBS 最適化とネットワーク強化機能を標準装備
- **プロセッサ多様性**: Intel、AMD、AWS Graviton プロセッサの選択肢

### 世代別の特徴

#### C6i インスタンス（Intel）
- 第3世代 Intel Xeon Scalable プロセッサ (Ice Lake)
- オールコアターボ周波数 3.5 GHz
- DDR4メモリ
- 最大 32 Gbps の EBS 帯域幅
- Nitro System 上に構築

#### C6a インスタンス（AMD）
- 第3世代 AMD EPYC プロセッサ (Milan)
- 最大 3.6 GHz の周波数
- コスト効率に優れたパフォーマンス
- 最大 20 Gbps のネットワーク帯域幅

#### C7g インスタンス（AWS Graviton）
- AWS Graviton3 プロセッサ
- ARM ベースのカスタムシリコン
- 前世代と比較して最大 25% 優れた計算性能
- 最大 30% 優れたエネルギー効率
- 最大 30 Gbps のネットワーク帯域幅

### 機械学習ワークフローでの活用

#### データ前処理
```python
# C シリーズインスタンスでの分散データ前処理の例
import dask.dataframe as dd
import numpy as np

# 大規模データセットの読み込み
df = dd.read_csv('s3://my-bucket/large-dataset/*.csv')

# 並列処理による特徴エンジニアリング
df['log_feature'] = df['raw_feature'].map_partitions(np.log)
df['scaled_feature'] = (df['feature'] - df['feature'].mean()) / df['feature'].std()

# 欠損値処理
df = df.fillna(df.mean())

# 結果の保存
df.to_parquet('s3://my-bucket/processed-data/')
```

#### CPU 依存の機械学習アルゴリズム
```python
# XGBoost トレーニングの例
import xgboost as xgb
from sklearn.model_selection import train_test_split

# データの準備
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# C シリーズインスタンスの複数コアを活用
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'eta': 0.1,
    'nthread': 64  # C シリーズの多数のコアを活用
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# トレーニング
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50
)
```

#### 分散バッチ推論
```python
# Spark を使用した分散バッチ推論の例
from pyspark.sql import SparkSession
import mlflow.pyfunc

# Spark セッションの初期化（C シリーズインスタンスのクラスター上）
spark = SparkSession.builder \
    .appName("Batch Inference") \
    .config("spark.executor.cores", "16") \
    .config("spark.executor.memory", "32g") \
    .getOrCreate()

# データの読み込み
inference_data = spark.read.parquet("s3://my-bucket/inference-data/")

# MLflow からモデルをロード
model = mlflow.pyfunc.spark_udf(spark, "runs:/run_id/model_name")

# 分散バッチ推論の実行
predictions = inference_data.withColumn("predictions", model(*inference_data.columns))

# 結果の保存
predictions.write.parquet("s3://my-bucket/inference-results/")
```

### インスタンスサイズとユースケース

| インスタンスサイズ | vCPU | メモリ (GiB) | 推奨ユースケース |
|-----------------|------|------------|--------------|
| c6i.large | 2 | 4 | 小規模データ処理、開発環境 |
| c6i.xlarge | 4 | 8 | 中規模データ変換、特徴エンジニアリング |
| c6i.2xlarge | 8 | 16 | バッチ推論、中規模モデルトレーニング |
| c6i.4xlarge | 16 | 32 | 分散データ処理、CPU 依存アルゴリズム |
| c6i.8xlarge | 32 | 64 | 大規模データ変換、並列処理 |
| c6i.12xlarge | 48 | 96 | 高性能分析パイプライン |
| c6i.16xlarge | 64 | 128 | 大規模分散処理 |
| c6i.24xlarge | 96 | 192 | エンタープライズ ML パイプライン |
| c6i.32xlarge | 128 | 256 | 最大規模の分散処理ワークロード |

### コスト最適化戦略

1. **インスタンスファミリーの選択**:
   - コスト重視: C6a (AMD) インスタンス
   - 電力効率: C7g (Graviton) インスタンス
   - 互換性優先: C6i (Intel) インスタンス

2. **スポットインスタンスの活用**:
   - 耐障害性のあるワークロードには最大 90% の割引
   - バッチ処理や耐障害性のある分散処理に最適

3. **自動スケーリング**:
   - 需要に応じたインスタンス数の自動調整
   - ピーク時のみリソースを追加し、コストを最適化

4. **Savings Plans**:
   - 1年または3年の利用コミットメントで最大 72% の割引
   - 柔軟なインスタンスファミリーやサイズの変更が可能

### ベストプラクティス

- **ワークロードに適したインスタンスサイズの選択**: 過剰なプロビジョニングを避ける
- **コンテナ化**: Docker コンテナを使用して環境の一貫性を確保
- **メモリ管理**: データ処理パイプラインでのメモリ使用量の最適化
- **並列処理**: 利用可能なすべての vCPU を活用するための並列処理の実装
- **モニタリング**: CloudWatch を使用した CPU 使用率とメモリ消費の監視
- **ストレージ最適化**: 一時データには Instance Store を活用

C シリーズインスタンスは、機械学習ワークフローの計算集約型コンポーネントに最適なバランスを提供します。特に大規模なデータ前処理、特徴エンジニアリング、CPU 依存のアルゴリズムトレーニング、バッチ推論などのタスクで、コスト効率と高性能を両立させることができます。

## P シリーズ

P シリーズは、NVIDIA GPU を搭載した高性能インスタンスファミリーで、機械学習トレーニングや高性能コンピューティングに特化しています。P3、P4d、P5 などの世代があり、NVIDIA Tesla V100、A100、H100 GPU を搭載しています。深層学習モデルのトレーニングや大規模な並列処理に最適なインスタンスです。

### 主な特徴

- **高性能 GPU**: 最新の NVIDIA データセンター GPU を搭載
- **大容量 GPU メモリ**: GPU あたり最大 80GB の HBM3 メモリ
- **高速相互接続**: NVLink による GPU 間の高帯域幅接続
- **クラスターネットワーキング**: EFA（Elastic Fabric Adapter）によるクラスター間の低レイテンシ通信
- **超高速ネットワーク**: 最大 3,200 Gbps のネットワーク帯域幅
- **大容量インスタンスストレージ**: 最大 8 TB の NVMe SSD ストレージ

### 世代別の特徴

#### P3 インスタンス
- NVIDIA Tesla V100 GPU
- GPU あたり 16GB または 32GB の HBM2 メモリ
- 最大 8 GPU（p3.16xlarge）
- 第1世代 NVLink による GPU 間接続
- 最大 25 Gbps のネットワーク帯域幅

#### P4d インスタンス
- NVIDIA A100 GPU（40GB HBM2 メモリ）
- 最大 8 GPU（p4d.24xlarge）
- 第3世代 NVLink と NVSwitch による GPU 間接続
- EFA ネットワークインターフェイス
- 最大 400 Gbps のネットワーク帯域幅
- 最大 8 TB の NVMe インスタンスストレージ

#### P5 インスタンス
- NVIDIA H100 GPU（80GB HBM3 メモリ）
- 最大 8 GPU（p5.48xlarge）
- 第4世代 NVLink による GPU 間接続
- 最大 3,200 Gbps のネットワーク帯域幅
- 最大 8 TB の NVMe インスタンスストレージ
- 前世代と比較して最大 6 倍の AI トレーニング性能

### 機械学習ワークフローでの活用

#### 大規模言語モデル（LLM）トレーニング
```python
# PyTorch と Accelerate を使用した分散 LLM トレーニングの例
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from accelerate import Accelerator
from datasets import load_dataset

# 分散トレーニングの初期化
accelerator = Accelerator()
device = accelerator.device

# モデルとトークナイザーのロード
model_name = "gpt2-xl"  # より大きなモデルには P5 インスタンスが最適
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# データセットの準備
dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=512),
    batched=True,
    remove_columns=["text"]
)

# トレーニング設定
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    fp16=True,  # 混合精度トレーニングで GPU メモリを節約
    gradient_checkpointing=True,  # 大規模モデル用のメモリ最適化
    save_steps=1000,
    logging_steps=100,
)

# トレーナーの初期化と実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# トレーニングの実行
trainer.train()
```

#### 分散深層学習（DeepSpeed との統合）
```python
# DeepSpeed を使用した ZeRO-3 最適化の例
# P シリーズインスタンスの複数 GPU と複数ノードを活用

# deepspeed_config.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "steps_per_print": 100,
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0
}

# 実行コマンド
# deepspeed --num_gpus=8 --num_nodes=4 train.py --deepspeed deepspeed_config.json
```

#### コンピュータビジョンモデルのトレーニング
```python
# PyTorch Lightning を使用した分散ビジョントランスフォーマートレーニング
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet

# モデル定義
class ViTClassifier(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = vit_b_16(pretrained=True)
        self.model.heads = nn.Linear(self.model.hidden_dim, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

# データセットとデータローダー
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageNet('/path/to/imagenet', split='train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8)

# トレーナーの設定と実行
trainer = pl.Trainer(
    accelerator='gpu',
    devices=8,  # P シリーズの 8 GPU を活用
    strategy=DDPStrategy(find_unused_parameters=False),
    precision=16,  # 混合精度トレーニング
    max_epochs=90,
    gradient_clip_val=1.0,
)

model = ViTClassifier()
trainer.fit(model, dataloader)
```

### インスタンスタイプとユースケース

| インスタンスタイプ | GPU | GPU メモリ | ユースケース |
|-----------------|-----|----------|-----------|
| p3.2xlarge | 1x V100 | 16 GB | 単一 GPU トレーニング、小規模モデル開発 |
| p3.8xlarge | 4x V100 | 64 GB | 中規模モデルトレーニング、データ並列処理 |
| p3.16xlarge | 8x V100 | 128 GB | 大規模モデルトレーニング、分散学習 |
| p3dn.24xlarge | 8x V100 | 256 GB | 大規模分散トレーニング、高速ネットワーク |
| p4d.24xlarge | 8x A100 | 320 GB | 大規模 LLM、マルチノードトレーニング |
| p5.48xlarge | 8x H100 | 640 GB | 最大規模 LLM、最高性能トレーニング |

### コスト最適化戦略

1. **適切なインスタンスの選択**:
   - モデルサイズと複雑さに基づいて最適なインスタンスを選択
   - 小規模モデルには P3、大規模モデルには P4d/P5

2. **スポットインスタンスの活用**:
   - チェックポイントを頻繁に保存して中断耐性を確保
   - 最大 90% のコスト削減が可能

3. **混合精度トレーニング**:
   - FP16/BF16 を使用して計算速度を向上させメモリ使用量を削減
   - より大きなバッチサイズが可能になりトレーニング効率が向上

4. **モデル並列化とシャーディング**:
   - DeepSpeed ZeRO や Megatron-LM などの技術を活用
   - 複数 GPU にモデルを分散して大規模モデルをトレーニング

5. **Savings Plans**:
   - 予測可能なワークロードには 1 年または 3 年のコミットメント
   - 最大 72% のコスト削減が可能

### ベストプラクティス

- **データローダーの最適化**: NVMe ストレージからの高速データロード
- **GPU メモリ管理**: 勾配チェックポイント、アクティベーションオフロードの活用
- **通信の最適化**: NCCL パラメータの調整、EFA の活用
- **バッチサイズの最適化**: 累積勾配を使用した効果的なバッチサイズの増加
- **モニタリング**: NVIDIA DCGM による GPU 使用率と温度のモニタリング
- **冷却と電力**: インスタンスの適切な冷却と電力管理

P シリーズインスタンスは、最先端の機械学習モデル開発において、トレーニング時間を大幅に短縮し、より大規模で複雑なモデルの構築を可能にします。特に大規模言語モデル（LLM）、コンピュータビジョン、マルチモーダルモデルなど、計算要求の高いディープラーニングワークロードに最適です。

## G シリーズ

G シリーズは、NVIDIA GPU を搭載したインスタンスファミリーで、グラフィックス処理や機械学習推論に最適化されています。G4dn、G5、G5g などの世代があり、NVIDIA T4、A10G GPU や AWS Graviton プロセッサと NVIDIA T4g GPU の組み合わせを提供します。コスト効率の高い推論処理や中規模のトレーニングに最適です。

### 主な特徴

- **推論最適化 GPU**: 機械学習推論に最適化された NVIDIA GPU
- **バランスの取れたリソース**: GPU、CPU、メモリのバランスの取れた構成
- **コスト効率**: P シリーズと比較して低コストで高性能な推論処理
- **多様な GPU オプション**: T4、A10G、T4g など様々な GPU タイプ
- **高速ネットワーク**: 最大 100 Gbps のネットワーク帯域幅
- **NVMe ストレージ**: 高速ローカルストレージによるデータアクセスの高速化

### 世代別の特徴

#### G4dn インスタンス
- NVIDIA T4 GPU
- GPU あたり 16GB GDDR6 メモリ
- 最大 8 GPU（g4dn.metal）
- 最大 50 Gbps のネットワーク帯域幅
- 最大 1.8 TB の NVMe SSD ストレージ
- NVIDIA TensorRT、CUDA、cuDNN のサポート

#### G5 インスタンス
- NVIDIA A10G GPU
- GPU あたり 24GB GDDR6 メモリ
- 最大 8 GPU（g5.48xlarge）
- 第4世代 PCIe インターフェイス
- 最大 100 Gbps のネットワーク帯域幅
- 最大 7.6 TB の NVMe SSD ストレージ

#### G5g インスタンス
- AWS Graviton2 プロセッサ + NVIDIA T4G GPU
- ARM ベースの CPU アーキテクチャ
- GPU あたり 16GB GDDR6 メモリ
- 最大 2 GPU（g5g.16xlarge）
- 最大 25 Gbps のネットワーク帯域幅
- 優れたコストパフォーマンス

### 機械学習ワークフローでの活用

#### リアルタイム推論エンドポイント
```python
# SageMaker エンドポイントでの G シリーズインスタンスの活用例
import boto3
import sagemaker
from sagemaker.model import Model

# セッションの設定
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# モデルの作成
model = Model(
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-gpu-py38',
    model_data='s3://my-bucket/model/model.tar.gz',
    role=role,
    predictor_cls=sagemaker.predictor.Predictor
)

# G5 インスタンスを使用したエンドポイントのデプロイ
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.xlarge',  # G5 インスタンスを使用
    endpoint_name='my-inference-endpoint'
)

# 推論リクエストの送信
response = predictor.predict({
    'inputs': 'What is machine learning?'
})
```

#### マルチモデルエンドポイント
```python
# 複数のモデルを単一の G シリーズインスタンスにデプロイ
from sagemaker.multidatamodel import MultiDataModel

# マルチモデルエンドポイントの作成
multi_model = MultiDataModel(
    name='my-multi-model-endpoint',
    model_data_prefix='s3://my-bucket/models/',
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-gpu-py38',
    role=role
)

# G5 インスタンスを使用したデプロイ
predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge'
)

# 特定のモデルを使用した推論
response = predictor.predict(
    target_model='bert-base-uncased/model.tar.gz',
    data='What is the capital of France?'
)
```

#### TensorRT を使用した最適化推論
```python
# NVIDIA TensorRT を使用した推論最適化の例
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# TensorRT エンジンの作成
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# ONNX モデルの解析
with open('model.onnx', 'rb') as model:
    parser.parse(model.read())

# 推論設定の構成
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # 半精度を有効化

# エンジンの構築
engine = builder.build_engine(network, config)

# 推論の実行
context = engine.create_execution_context()
# 入力と出力バッファの設定...
# 推論の実行...
```

#### ビデオ処理と分析
```python
# G シリーズインスタンスでのリアルタイムビデオ分析の例
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# モデルのロード
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.cuda()  # GPU に移動

# ビデオストリームの処理
cap = cv2.VideoCapture('rtsp://camera-stream-url')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 前処理
    img_tensor = F.to_tensor(frame).unsqueeze(0).cuda()
    
    # 推論
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # 結果の処理と可視化
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # 検出結果の描画
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.7:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 結果の表示または保存
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### インスタンスタイプとユースケース

| インスタンスタイプ | GPU | GPU メモリ | ユースケース |
|-----------------|-----|----------|-----------|
| g4dn.xlarge | 1x T4 | 16 GB | 単一モデル推論、小規
