import re
import uuid
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pinecone_db import store_text, search_similar, assign_labels_to_text
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
# 環境変数からAPIキーを取得
API_KEY = os.getenv("API_KEY")

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class StoreRequest(BaseModel):
    text: str
    labels: list[str]  # ユーザーが確定したラベル


# CORS 設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて変更
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/classify")
async def classify_text(request: TextRequest,x_api_key: str = Header(...)):
    # リクエストヘッダーからAPIキーを取得して認証
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    """
    入力された問題文に対して、類似検索を行い、推奨ラベルを返すAPI
    """
    text = request.text
    suggested_labels = assign_labels_to_text(text)

    # レスポンスをJSONで返す際、Content-TypeをUTF-8に設定
    return JSONResponse(
        content={"input": text, "suggested_labels": suggested_labels},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )


@app.post("/store")
async def store_text_api(request: StoreRequest,x_api_key: str = Header(...)):
    # リクエストヘッダーからAPIキーを取得して認証
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    text = request.text
    labels = request.labels

    # Pineconeに保存
    store = store_text(text, labels)

@app.post("/meta_store")
async def metastore_text_api(request: TextRequest):

    # コロン (:) または 改行 (\n) で分割
    lines = [line.strip() for line in request.text.strip().split('\n') if line.strip()]
    
    stored_count = 0
    all_labels = []

    for line in lines:
        # コロンでラベル部分と問題部分を分割（最初のコロンのみ対象）
        label_part, text = line.split(':', 1)
        # カンマでラベルを複数分割し、空白を削除
        label_list = [label.strip() for label in label_part.split(',') if label.strip()]
        text = text.strip()

        #データを保存
        store_text(text, label_list)
        stored_count += 1
        all_labels.append(label_list)

    return {
        "status": "success",
        "stored": stored_count,
        "labels": all_labels
    }

@app.post("/search")
async def search_api(request: StoreRequest,x_api_key: str = Header(...)):
    # リクエストヘッダーからAPIキーを取得して認証
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    text = request.text
    labels = request.labels
    
    # ラベルを使用した類似検索
    similar_texts = search_similar(text, labels)
    
    return JSONResponse(
        content={
            "message": "Text searched successfully!",
            "text": text,
            "labels": labels,
            "similar_texts": similar_texts
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




#=========確認用================
# class SimilarRequest(BaseModel):
#     text: str
#     labels: list[str]  # 確認したいラベル
#===========確認用=============
# @app.post("/similar")
# async def check_similar_texts(request: SimilarRequest):
#     """ 保存せずに、指定されたラベルの範囲で類似検索を行うAPI """
#     text = request.text
#     labels = request.labels

#     similar_texts = search_similar(text, labels)

#     return {
#         "input": text,
#         "labels": labels,
#         "similar_texts": similar_texts
#     }