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





# #========ラベルの保存コード=========
# def store_labels(labels):
#     """
#     ラベルセットをPineconeに保存する
#     :param labels: ラベルのリスト
#     """
#     for label in labels:
#         # ラベルに対応する埋め込みベクトルを生成
#         vector = vectorize_text(label)  # classify_textでラベルの埋め込みを取得
#         # ラベルに対して自動生成したUUIDをIDとして使用
#         generated_id = str(uuid.uuid4())  # 一意なIDを生成
#         index.upsert([(generated_id, vector, {"labels": label})])  # ラベルをPineconeに保存


# # 最初にラベルセットを保存
# labels = [
#     "数学 - 正の数・負の数",
# "数学 - 文字式",
# "数学 - 1次方程式",
# "数学 - 比例・反比例",
# "数学 - 平面図形",
# "数学 - 空間図形",
# "数学 - データの活用",
# "数学 - 式と計算",
# "数学 - 連立方程式",
# "数学 - 図形の調べ方",
# "数学 - 三角形と四角形",
# "数学 - 1次関数",
# "数学 - 四分位範囲と箱ひげ図",
# "数学 - 確率",
# "数学 - 平方根",
# "数学 - 2次方程式",
# "数学 - 相似な図形",
# "数学 - 円",
# "数学 - 三平方の定理",
# "数学 - 標本調査",
# "数学 - 数と式",
# "数学 - 図形と計算",
# "数学 - 二次関数",
# "数学 - データの分析",
# "数学 - 図形の性質",
# "数学 - 場合の数と確率",
# "数学 - 数学と人間の活動(整数)",
# "数学 - いろいろな式",
# "数学 - 図形と方程式",
# "数学 - 指数関数・対数関数",
# "数学 - 三角関数",
# "数学 - 微分・積分の考え",
# "数学 - 数列",
# "数学 - 統計的な推測(確率・統計)",
# "数学 - 数学と社会生活(データの分析)",
# "数学 - 極限",
# "数学 - 微分法",
# "数学 - 積分法",
# "数学 - ベクトル",
# "数学 - 平面上の曲線と複素平面",
# "数学 - 数学的な表現の工夫",
# "理科 - 光と音",
# "理科 - 物質のすがた",
# "理科 - 力と圧力",
# "理科 - 生物の観察",
# "理科 - 植物又は動物の生活と種類",
# "理科 - 大地の変化",
# "理科 - 化学変化と原子・分子",
# "理科 - 電流と磁界",
# "理科 - 放射線",
# "理科 - 細胞と体の仕組み",
# "理科 - 動物の生活と仕組み",
# "理科 - 生物の進化",
# "理科 - 天気とその変化",
# "理科 - 化学変化とイオン",
# "理科 - 仕事とエネルギー",
# "理科 - 科学と人間生活",
# "理科 - 物質の科学",
# "理科 - 生命の科学",
# "理科 - 光や熱の科学",
# "理科 - 宇宙や地球の科学",
# "理科 - これからの科学と人間生活",
# "理科 - 力学",
# "理科 - 熱",
# "理科 - 波動",
# "理科 - 電磁気",
# "理科 - 物質の構成と化学結合",
# "理科 - 物質の変化",
# "理科 - 生物の特徴",
# "理科 - 遺伝子とその働き",
# "理科 - 人間の体の調節",
# "理科 - 生物の多様性と生態系",
# "理科 - 個体地球とその活動",
# "理科 - 大気と海洋",
# "理科 - 移り変わる地球",
# "理科 - 自然との共生",
# "理科 - 生物の進化",
# "理科 - 生命現象と物質",
# "理科 - 遺伝子情報の発現と発生",
# "理科 - 生物の環境応答",
# "理科 - 生態と環境",
# "理科 - 個体宇宙の概観と活動",
# "理科 - 地球の歴史",
# "理科 - 大気と海洋",
# "理科 - 宇宙の構造",
# "理科 - 理科課題研究",
# "理科 - 理数探求基礎",
# "社会 - 世界のすがた",
# "社会 - 世界の気候とくらし",
# "社会 - 世界各地の文化",
# "社会 - 世界の様々な地域",
# "社会 - 世界の様々な地域の調査",
# "社会 - 日本のすがた",
# "社会 - 世界と比べてみた日本",
# "社会 - 日本の諸地域",
# "社会 - 身近な地域の調査",
# "社会 - 日本の都道府県の調査",
# "社会 - 人類の出現",
# "社会 - 古代国家の成立と展開",
# "社会 - 中世",
# "社会 - 近世",
# "社会 - 近代",
# "社会 - 第二次世界大戦",
# "社会 - 現代社会と私たちの生活",
# "社会 - 人権思想と民主主義の歩み",
# "社会 - 現代の民主政治",
# "社会 - 民主政治と政治参加",
# "社会 - 国民生活と経済",
# "社会 - 国際社会",
# "社会 - 地球規模の問題",
# "社会 - 法解釈",
# "国語 - 小説",
# "国語 - 随筆",
# "国語 - 古文",
# "国語 - 漢文",
# "国語 - 敬語",
# "英語 - be動詞・一般動詞",
# "英語 - 冠詞・名詞の複数形",
# "英語 - 代名詞",
# "英語 - 命令文・感嘆文",
# "英語 - 疑問視",
# "英語 - 現在進行形",
# "英語 - 助動詞(can:～できる)",
# "英語 - 前置詞",
# "英語 - 過去形・過去進行形",
# "英語 - 未来を表す文",
# "英語 - 接続詞",
# "英語 - There is構文",
# "英語 - 不定詞",
# "英語 - 比較",
# "英語 - 現在完了",
# "英語 - 過去形・過去進行形",
# "英語 - 接続詞",
# "英語 - 助動詞(must)",
# "英語 - 動名詞と不定詞",
# "英語 - 比較",
# "英語 - 受動態",
# "英語 - 現在完了",
# "英語 - 受動態",
# "英語 - 現在完了進行形",
# "英語 - 文型",
# "英語 - いろいろな疑問文",
# "英語 - 不定詞の構文",
# "英語 - 分詞",
# "英語 - 関係代名詞",
# "英語 - 仮定法",
# "情報 - 情報社会の問題解決",
# "情報 - コミュニケーションと情報デザイン",
# "情報 - コンピュータとプログラミング",
# "情報 - 情報通信ネットワークとデータの活用"
# ]


# # ラベルを保存
# store　　
# _labels　　
# (labels)　　

