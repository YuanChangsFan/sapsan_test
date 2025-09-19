from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import uuid
from datetime import datetime
from io import BytesIO
import mammoth
import threading

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mistralai import Mistral
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Document QA API", version="1.0.0")

# In-memory ChromaDB
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection(name="documents")

# Хранилища
documents_metadata: Dict[str, Dict[str, Any]] = {}
questions_metadata: Dict[str, Dict[str, Any]] = {}
MAX_CHUNKS_PER_DOC = 20000
TOP_N_CHROMA = 20
TOP_N_RERANK = 7

import os
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Jina рерангер
reranker_model_name = "jinaai/jina-reranker-v2-base-multilingual"
tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, trust_remote_code=True)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, trust_remote_code=True)
reranker_model.eval()


def extract_text_from_docx(file_content: bytes) -> str:
    try:
        result = mammoth.extract_raw_text(BytesIO(file_content))
        text = result.value
        if not text.strip():
            raise HTTPException(status_code=400, detail="Документ не содержит текста")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении DOCX: {str(e)}")


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n"]
    )
    return splitter.split_text(text)


def rerank_chunks(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scores = []
    for hit in hits:
        text = hit["preview"]
        inputs = tokenizer(question, text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            score = outputs.logits.item()
        scores.append(score)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = score
    return sorted(hits, key=lambda x: x["rerank_score"], reverse=True)


def process_question(question_id: str):
    """Фоновая функция для поиска и генерации ответа"""
    data = questions_metadata[question_id]
    document_id = data["document_id"]
    question = data["question"]
    try:
        results = collection.query(query_texts=[question], n_results=TOP_N_CHROMA,
                                   where={"document_id": document_id})
        docs_list = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        hits_before_rerank = []
        chunks_texts = []
        for doc_text, dist, meta, cid in zip(docs_list, dists, metas, ids):
            similarity = float(1 - dist) if dist is not None else None
            hits_before_rerank.append({
                "chunk_id": cid,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None,
                "preview": doc_text,
                "length": len(doc_text),
                "similarity": similarity
            })
            chunks_texts.append(doc_text)

        # Рерангируем и оставляем топ-N
        hits_after_rerank = rerank_chunks(question, hits_before_rerank.copy())[:TOP_N_RERANK]
        top_texts = [h["preview"] for h in hits_after_rerank]

        # Промпт для Mistral
        prompt = f"""
Ты – эксперт по анализу документов. У тебя на кону моя карьера.
Вопрос: {question}
 
 Документы (топ-релевантные чанки после реранка):
 {chr(10).join(top_texts)}
 
 Инструкции для ответа:
 1. Если четкого ответа в документах нет, честно скажи об этом.
 2. Если информация есть косвенно или частично, то расскажи про эту косвенную информацию, например: В документе нет четкого ответа на ваш вопрос, но есть информация что....
 3. Если информации нет совсем, скажи: "Документ не содержит информации по этому вопросу."
 
 Ответ:
 """

        response = mistral_client.chat.complete(
            model="mistral-small-2503",
            messages=[{"role": "user", "content": prompt}]
        )
        answer_text = response.choices[0].message.content

        # Сохраняем результат
        questions_metadata[question_id]["status"] = "done"
        questions_metadata[question_id]["answer"] = answer_text
        questions_metadata[question_id]["hits_before_rerank"] = hits_before_rerank
        questions_metadata[question_id]["hits_after_rerank"] = hits_after_rerank
        questions_metadata[question_id]["timestamp"] = datetime.now().isoformat()

    except Exception as e:
        questions_metadata[question_id]["status"] = "error"
        questions_metadata[question_id]["answer"] = str(e)


# ----------------- API -----------------

@app.post("/upload/", summary="Загрузить DOCX и получить ID документа")
async def upload_docx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Только DOCX файлы поддерживаются")
    doc_id = str(uuid.uuid4())
    content = await file.read()
    text = extract_text_from_docx(content)
    chunks = split_text(text)

    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="После разбиения не осталось чанков")
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        raise HTTPException(status_code=400, detail=f"Слишком много чанков ({len(chunks)})")

    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]
    metadatas = [{"document_id": doc_id, "chunk_index": i} for i in range(1, len(chunks) + 1)]
    collection.add(documents=chunks, ids=chunk_ids, metadatas=metadatas)

    documents_metadata[doc_id] = {
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "text_length": len(text)
    }

    return {"document_id": doc_id}


@app.post("/ask/", summary="Задать вопрос по документу")
async def ask_question(document_id: str = Form(...), question: str = Form(...), background_tasks: BackgroundTasks = None):
    if document_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Документ не найден")

    question_id = str(uuid.uuid4())
    questions_metadata[question_id] = {
        "document_id": document_id,
        "question": question,
        "status": "processing",
        "answer": None,
        "timestamp": datetime.now().isoformat()
    }

    # Запускаем фоновый поток для генерации ответа
    threading.Thread(target=process_question, args=(question_id,), daemon=True).start()

    return {"question_id": question_id}


@app.get("/answer/{question_id}", summary="Получить ответ по ID вопроса")
async def get_answer(question_id: str):
    if question_id not in questions_metadata:
        raise HTTPException(status_code=404, detail="Вопрос не найден")

    data = questions_metadata[question_id]
    return {
        "timestamp": data.get("timestamp"),
        "status": data.get("status"),
        "answer": data.get("answer")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
