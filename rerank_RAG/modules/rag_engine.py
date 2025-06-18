from modules.models import get_response
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化 reranker 模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained("maidalun1020/bce-reranker-base_v1", token=None)
model = AutoModelForSequenceClassification.from_pretrained("maidalun1020/bce-reranker-base_v1", token=None)
model.eval()  # 設定為評估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def rerank_documents(question, documents, top_n=3):
    ranked_docs = []
    
    for doc in documents:
        # 準備輸入：結合問題和文件內容
        input_text = f"{question} [SEP] {doc.page_content}"
        try:
            # 編碼輸入
            inputs = tokenizer(
                input_text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # 獲取模型輸出
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.sigmoid(outputs.logits).item()  # 將 logits 轉為 0-1 分數
            print(f"文件：{doc.page_content[:50]}... 得分：{score}")  # 添加日誌
            ranked_docs.append((doc, score))
        except Exception as e:
            st.warning(f"文件排序錯誤: {e}")
            ranked_docs.append((doc, 0))
    # 按分數排序並選取 top_n
    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Top {top_n} 文件：{[doc.page_content[:50] for doc, score in ranked_docs]}")  # 添加日誌
    return [doc for doc, score in ranked_docs]

def generate_answer(question, vectorstore, generation_model):
    """生成回答，優先使用文件資料庫"""
    print(f"處理問題：{question}")  # 添加日誌
    doc_context = ""
    source_info = []
    # 從文件資料庫檢索相關資訊
    if vectorstore:
        print("vectorstore 存在，開始檢索")  # 添加日誌
        with st.spinner("正在文件中搜尋相關資訊..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            print(f"檢索到 {len(docs)} 個文件，問題：{question}")  # 日誌
            if docs:
                for i, doc in enumerate(docs):
                    print(f"文件 {i+1}：{doc.page_content[:100]}...")  # 修正日誌
                # 使用自定義 reranker
                docs = rerank_documents(question, docs, top_n=3)
                doc_context = "\n\n".join([doc.page_content for doc in docs])
                source_info.append("文件資料庫")
                # 檢查文件是否包含足夠資訊
                evaluation_prompt = f"""
                評估以下文件內容是否包含回答問題所需的資訊。
                如果包含足夠資訊（即使部分相關），回答 "yes"；否則回答 "no"。
                問題: {question}
                文件內容: {doc_context}
                """
                messages = [{"role": "user", "content": evaluation_prompt}]
                evaluation = get_response(messages, model="deepseek-r1:7b").strip().lower()
                print(f"文件是否足以回答：{evaluation}，文件內容：{doc_context[:100]}...")  # 添加日誌
                
                if "yes" in evaluation:
                    st.info("在文件中找到相關資訊")
            else:
                print("未在文件中找到相關資訊")
                prompt = f"回答以下問題。如果不確定答案，請誠實說明: {question}"
                messages = [{"role": "user", "content": prompt}]
                answer = get_response(messages, model="EntropyYue/chatglm3:latest")
                print("無足夠文件資訊，使用模型知識回答")  # 添加日誌
                return answer, "直接使用模型知識"
    # 若找到文件，使用文件生成回答
    if doc_context:
        prompt = f"""根據以下資訊回答問題。如果資訊不足以完整回答，盡量根據所有提供的資訊給出最全面的回答。
資訊:
{doc_context}
問題: {question}
"""
        messages = [{"role": "user", "content": prompt}]
        answer = get_response(messages, model="EntropyYue/chatglm3:latest")
        source_text = "和".join(source_info)
        print(f"使用文件回答，來源：{source_text}")  # 添加日誌
        return answer, f"來自 {source_text}"
    
    # 若無任何資訊，作為後備方案
    prompt = f"回答以下問題。如果不確定答案，請誠實說明: {question}"
    messages = [{"role": "user", "content": prompt}]
    answer = get_response(messages, model="EntropyYue/chatglm3:latest")
    print("無任何資訊，使用模型知識回答")  # 添加日誌
    return answer, "直接使用模型知識"
