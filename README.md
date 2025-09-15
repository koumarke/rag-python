# RAG Python - 檢索增強生成問答系統

本專案為一個基本的 Python RAG (Retrieval-Augmented Generation) 問答系統，使用 ChromaDB 進行向量檢索，Google Gemini 進行文本生成，並整合了重新排序機制以提升回答品質。

## 功能特色

- 🔍 **智能檢索**：使用 ChromaDB 進行高效的向量相似度搜尋
- 🤖 **AI 生成**：整合 Google Gemini 2.5 Flash 模型進行智能回答
- 📊 **重新排序**：使用 CrossEncoder 對檢索結果進行重新排序，提升相關性
- 🌏 **中文支援**：使用中文語義嵌入模型，優化中文文本處理
- ⚡ **快速部署**：使用 uv 進行依賴管理，快速安裝和運行

## 系統需求

- Python >= 3.12
- Google Gemini API Key
- 至少 4GB RAM (用於載入嵌入模型)

## 安裝設定

```bash
# step.1 下載專案
git clone <repository-url>
cd rag-python

# step.2 安裝 uv (依賴管理工具)
# On macOS and Linux. (更多安裝方式請參考 https://github.com/astral-sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# step.3 安裝專案依賴套件
uv pip install -e .

# step.4 設定環境變數
# 建立 .env 檔案並加入您的 Google Gemini API Key
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

> **注意**：請到 [Google AI Studio](https://makersuite.google.com/app/apikey) 申請 Gemini API Key

## 使用方式

### 基本語法

```bash
uv run python main.py "您的問題" [--doc-path "文檔路徑"]
```

### 參數說明

- `query`：要查詢的問題（必需）
- `--doc-path`：文檔路徑（可選，預設為 `./story.txt`）

### 使用範例

```bash
# 使用預設文檔 (story.txt) 進行查詢
uv run python main.py "亞倫是誰？"

# 指定自訂文檔進行查詢
uv run python main.py "故事的主要情節是什麼？" --doc-path "/path/to/other/article.txt"
```

## 系統架構

本系統採用 RAG (Retrieval-Augmented Generation) 架構，包含以下主要組件：

### 1. 文檔處理流程

```
原始文檔 → 文本分割 → 嵌入向量化 → 向量資料庫儲存
```

### 2. 查詢處理流程

```
用戶問題 → 問題嵌入 → 向量檢索 → 重新排序 → LLM 生成 → 最終回答
```

### 3. 技術堆疊

- **向量資料庫**：ChromaDB (EphemeralClient)
- **嵌入模型**：shibing624/text2vec-base-chinese
- **重新排序**：cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- **生成模型**：Google Gemini 2.5 Flash
- **依賴管理**：uv

## 專案結構

```
rag-python/
├── main.py              # 主程式入口
├── pyproject.toml       # 專案配置和依賴
├── README.md            # 專案說明文件
├── story.txt            # 預設文檔檔案
└── uv.lock              # 依賴鎖定檔案
```

## 總結

本專案用最基本的架構與程式來展示 RAG 的實現邏輯，您可以依此作為起點發展自己的 RAG 問答系統。
