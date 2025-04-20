from sentence_transformers import SentenceTransformer

# モデルを1回だけロードして再利用
_model = SentenceTransformer("all-MiniLM-L6-v2")

def vectorize_text(text: str) -> list[float]:
    """
    テキストをベクトル化（list[float]）
    """
    vector = _model.encode(text, convert_to_tensor=True)
    return vector.cpu().detach().numpy().tolist()
