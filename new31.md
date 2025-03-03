# AWS機械学習関連用語解説

## SageMaker Training Compiler
深層学習モデルのトレーニングを高速化するコンパイラで、GPUメモリの使用効率を向上させます。計算グラフを最適化し、ハードウェアリソースの利用効率を高めることで、トレーニング時間とコストを削減します。

**主な特徴:**
- 計算グラフの最適化と再構成
- GPU メモリ使用量の削減（最大 20%）
- トレーニング時間の短縮（最大 50%）
- TensorFlow と PyTorch のネイティブサポート
- 自動混合精度（AMP）との互換性
- コードの変更なしで利用可能

**対応モデルタイプ:**
- 自然言語処理モデル（BERT、GPT など）
- コンピュータビジョンモデル（ResNet、EfficientNet など）
- 時系列予測モデル
- 推薦システム
- カスタムモデルアーキテクチャ

**最適化技術:**
- 演算融合（Operation Fusion）
- メモリ管理の最適化
- カーネル自動チューニング
- データレイアウトの最適化
- 計算スケジューリングの改善
- デッドコード除去

**ユースケース:**
- 大規模言語モデルの効率的なトレーニング
- コスト削減が必要な長時間トレーニング
- リソース制約のある環境でのモデル開発
- 反復的な実験サイクルの高速化
- 本番環境向けモデルの最終トレーニング
- 複雑なアーキテクチャの最適化

## Amazon SageMaker マネージドスポットトレーニング
スポットインスタンスを使用してモデルトレーニングのコストを削減する機能です。中断時にチェックポイントから再開できます。オンデマンドインスタンスと比較して最大 90% のコスト削減が可能です。

**主な特徴:**
- 自動チェックポイント管理
- 中断時の自動再開
- コスト最適化（最大 90% 削減）
- トレーニングジョブの進捗追跡
- 最大スポット使用率の設定
- ハイブリッドモード（スポットとオンデマンドの混在）

**設定オプション:**
- 最大待機時間
- チェックポイント頻度
- S3 チェックポイント保存先
- 最大スポットインスタンス割合
- フォールバック戦略
- インスタンスタイプと分散設定

**ベストプラクティス:**
- 適切なチェックポイント間隔の設定
- 耐障害性のあるトレーニングスクリプトの作成
- 複数のアベイラビリティゾーンの活用
- 需要の低い時間帯でのジョブスケジューリング
- 適切なタイムアウト設定
- 進捗モニタリングの実装

**ユースケース:**
- 長時間実行される非緊急トレーニングジョブ
- 予算制約のあるプロジェクト
- バッチ処理や夜間処理
- 研究開発環境での実験
- 定期的な再トレーニングパイプライン
- 大規模なハイパーパラメータ最適化

## Amazon SageMaker ノートブック
Jupyter互換のノートブックインスタンスで、データの探索や機械学習モデルの開発に使用されます。フルマネージド環境でデータサイエンスワークフローを効率化します。

**主な特徴:**
- フルマネージド Jupyter 環境
- 事前設定された ML フレームワーク
- インスタンスタイプの柔軟な選択
- Git 統合によるバージョン管理
- ノートブックの共有と協業機能
- ライフサイクル設定によるカスタマイズ

**提供形態:**
1. **SageMaker ノートブックインスタンス**:
   - 独立したマネージドインスタンス
   - 自動起動/停止のスケジューリング
   - インスタンスごとのリソース分離

2. **SageMaker Studio ノートブック**:
   - SageMaker Studio 内の統合環境
   - 迅速な起動と柔軟なリソース変更
   - 共有環境とコラボレーション機能

**ライフサイクル設定の例:**
- カスタムパッケージのインストール
- データセットの自動マウント
- セキュリティ設定の適用
- 環境変数の設定
- Git リポジトリの自動クローン
- カスタム拡張機能の有効化

**ユースケース:**
- データ探索と可視化
- モデルプロトタイピングと実験
- 対話型機械学習開発
- チーム間の知識共有と協業
- 教育とトレーニング
- レポート作成とドキュメント化

## AWS Glue Data Quality
データの品質を評価し、問題を検出するためのツールです。データの整合性と信頼性を確保します。機械学習モデルの入力データの品質を保証し、「ガベージイン、ガベージアウト」問題を防止します。

**主な特徴:**
- ルールベースのデータ品質チェック
- 統計的プロファイリングと異常検出
- データ品質スコアの計算
- 自動ルール推奨
- スケジュール実行と監視
- ETL パイプラインとの統合

**データ品質ルールの種類:**
- 完全性チェック（欠損値、空値）
- 一意性チェック（重複データ）
- 整合性チェック（参照整合性）
- 有効性チェック（データ型、範囲、パターン）
- 鮮度チェック（タイムスタンプ）
- 正確性チェック（ビジネスルール）

**実装方法:**
- AWS Glue Studio での視覚的設定
- DQDL（Data Quality Definition Language）によるルール定義
- Python API を使用したプログラム的アプローチ
- AWS Glue ジョブへの組み込み
- CloudWatch との統合によるアラート設定

**ユースケース:**
- ML パイプラインの入力データ検証
- データレイク/データウェアハウスの品質保証
- データ統合プロセスの監視
- 規制コンプライアンスの証明
- ビジネスインテリジェンスの信頼性向上
- データガバナンスの強化

## sagemaker-model-monitor-analyzer
SageMaker Model Monitorの一部で、デプロイされたモデルのデータドリフトやモデル品質を分析します。本番環境でのモデルパフォーマンスを継続的に監視し、問題を早期に検出します。

**主な監視機能:**
- データ品質モニタリング
- モデル品質モニタリング
- バイアスドリフトモニタリング
- 特徴アトリビューションドリフトモニタリング
- カスタムメトリクスの定義と追跡
- しきい値ベースのアラート

**分析プロセス:**
1. ベースラインの計算（トレーニングデータの統計）
2. 推論データの収集と前処理
3. 統計的分析と比較
4. 違反の検出とレポート生成
5. アラートの発行（CloudWatch、SNS）
6. 結果の可視化と保存

**検出可能な問題:**
- 特徴分布の変化（データドリフト）
- 予測分布の変化（コンセプトドリフト）
- モデル精度の低下
- 欠損値や外れ値の増加
- バイアス指標の変化
- 特徴重要度の変化

**ユースケース:**
- 本番環境でのモデルの継続的な品質保証
- 再トレーニングの必要性の自動検出
- コンプライアンス要件の遵守
- モデルパフォーマンスの透明性確保
- データソースの問題の早期発見
- モデルのライフサイクル管理の自動化
