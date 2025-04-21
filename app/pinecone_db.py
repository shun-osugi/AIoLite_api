from vectorizer import vectorize_text
import os
from pinecone import Pinecone, ServerlessSpec
import uuid
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAPIキーを取得
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pineconeクラスのインスタンスを作成
pc = Pinecone(api_key=pinecone_api_key)

# インデックスが存在しない場合、作成する
index_name = 'myindex'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # 埋め込みベクトルの次元数
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # 利用可能なリージョンに変更
        )
    )

# インデックスに接続
index = pc.Index(index_name)

def assign_labels_to_text(text: str, threshold: float=0.74, top_k: int=4) -> List[str]:
    """
    問題文に対して複数のラベルを付ける関数
    :param text: 解析する問題文
    :param threshold: ラベルを付ける際の類似度スコアの閾値
    :param top_k: 検索するラベルの数
    :return: 類似度スコアが閾値以上のラベル
    """
    # 問題文をベクトル化
    text_vector = vectorize_text(text)
    # Pineconeで類似するラベルを検索
    results = index.query(vector=text_vector, top_k=top_k, include_metadata=True)
    #setは重複を許さないリスト
    labels = set()
    #matchesキーが取得できなければからリストを返す
    for match in results.get("matches", []):
        if match["score"] >= threshold and "labels" in match["metadata"]:
            meta_labels = match["metadata"]["labels"]
            #meta_labelsがstrならばlistにする
            if isinstance(meta_labels, str):
                meta_labels = [meta_labels]
            labels.update(meta_labels)

    return list(labels) if labels else ["その他 - その他"]


def store_text(text: str, labels: list[str]) -> bool:
    """
    テキストをベクトル化してPineconeに保存
    """
    vector = vectorize_text(text)

    # 類似チェック
    existing = index.query(vector=vector, top_k=1, include_metadata=True)
    for match in existing.get("matches", []):
        if match["score"] > 0.96:  # かなり近い
            if match["metadata"].get("text") == text:
                print(f"⚠️ 重複テキスト：保存スキップ：{text}")
                return False

    id = str(uuid.uuid4())
    metadata = {"text": text, "labels": labels}
    index.upsert([(id, vector, metadata)])
    print(f"✅ 保存完了: {id},{text}")
    return True

def search_similar(text: str, labels: list[str], top_k=3):
    """
    ラベルでフィルタリングしながら類似検索を実行
    渡されたテキストと全く同じ問題は類題として採用しない
    """
    # 渡されたテキストのベクトルを取得
    vector = vectorize_text(text)
    
    # ラベルでフィルタリング
    filter_conditions = {"labels": {"$in": labels}}

    # 類似検索を実行
    results = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter_conditions)
    
    return [
        {
            "text": match["metadata"].get("text", ""),
            "labels": match["metadata"].get("labels", []),
            "score": match["score"]
        }
        for match in results["matches"]
        if match["metadata"].get("text") != text
    ]
