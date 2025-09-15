import argparse
from typing import List

import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='RAG 問答系統')
    parser.add_argument(
        'query', 
        type=str, 
        help='要查詢的問題'
    )
    parser.add_argument(
        '--doc-path', 
        type=str, 
        default='./story.txt',
        help='文檔路徑 (預設: ./story.txt)'
    )
    return parser.parse_args()


def split_into_chunks(doc_file: str) -> List[str]:
    """資料分塊"""
    try:
        with open(doc_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {doc_file}")
        return []
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        return []


def embed_chunk(chunk: str, embedding_model: SentenceTransformer) -> List[float]:
    """將文本資料轉換為嵌入向量"""
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


def save_embeddings(chunks: List[str], embeddings: List[List[float]], collection) -> None:
    """將嵌入向量保存到資料庫"""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


def retrieve(query: str, top_k: int, collection, embedding_model: SentenceTransformer) -> List[str]:
    """檢索(搜尋)相關文檔資料"""
    query_embedding = embed_chunk(query, embedding_model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    """重新排序檢索結果"""
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


def generate(query: str, chunks: List[str], google_client) -> str:
    """使用 LLM 生成回答"""
    prompt = f"""你是一位知識助手，請根據使用者的問題和下列相關片段生成準確的回答。

用戶問題: {query}

相關片段:
{"\n\n".join(chunks)}

請基於上述內容作答，不要編造資訊。"""

    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"生成回答時發生錯誤：{e}"


def main():
    """主要執行函數"""

    # 解析命令列參數
    args = parse_arguments()
    query = args.query
    doc_path = args.doc_path

    # 載入環境變數
    load_dotenv()

    # 初始化 Google Gemini 客戶端
    google_client = genai.Client()

    
    # 分割文檔
    chunks = split_into_chunks(doc_path)
    if not chunks:
        print("無法載入文檔，程式結束")
        return
    
    print(f"成功載入 {doc_path}, 內容已經切分成 {len(chunks)} 個分塊 ...")
    
    # 初始化嵌入模型
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    
    # 生成嵌入向量
    embeddings = [embed_chunk(chunk, embedding_model) for chunk in chunks]
    
    # 初始化 ChromaDB
    chromadb_client = chromadb.EphemeralClient()
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    
    # 保存嵌入向量
    save_embeddings(chunks, embeddings, chromadb_collection)
    
    # 查詢和檢索
    retrieved_chunks = retrieve(query, 5, chromadb_collection, embedding_model)
    
    # 重新排序
    reranked_chunks = rerank(query, retrieved_chunks, 3)
    
    # 生成回答
    answer = generate(query, reranked_chunks, google_client)
    print(f"\n問題：\n{query}")
    print(f"\n回答：\n{answer}")


if __name__ == "__main__":
    main()