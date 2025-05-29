import pandas as pd
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed  # 또는 Gemini 함수
from lightrag.llm.gemini import gemini_2_0_flash_complete, gemini_embed  # 또는 OpenAI 함수
from dotenv import load_dotenv

import os
import asyncio
from lightrag import LightRAG, QueryParam


from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger


setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def process_large_dataframe_to_lightrag(df, rag, batch_size=16):
    """대용량 DataFrame을 배치 단위로 처리"""
    
    total_rows = len(df)
    print(f"총 {total_rows}개 문서를 {batch_size}개씩 배치 처리...")
    
    # DataFrame을 배치로 나누기
    for i in range(0, total_rows, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        documents = []
        document_ids = []
        file_paths = []
        
        for idx, (title, row) in enumerate(batch_df.iterrows()):
            # 제목을 안전한 ID로 변환
            # 완전한 문서 생성
            full_document = f"""=== {title} ===
                Publication Date: {row['date']}
                Document Content:
                {row['content']}
                Document Type: USTR Trade Policy Document
                Source: United States Trade Representative
                """
            
            documents.append(full_document.strip())
            document_ids.append(f"ustr_{i+idx}_{title}")
            file_paths.append("ustr_trade_documents")
        
        print(f"배치 {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} 처리 중... ({len(documents)}개 문서)")
        
        # 배치 삽입
        await rag.ainsert(
            input=documents,
            ids=document_ids,
            file_paths=file_paths
        )
    
    print("모든 문서 처리 완료!")




async def main():
    # LightRAG 초기화

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,  # 또는 gemini_embed
        llm_model_func=gemini_2_0_flash_complete,
        vector_storage="PGVectorStorage",
        kv_storage = "PGKVStorage",
        graph_storage = "Neo4JStorage",
        doc_status_storage = "PGDocStatusStorage",
         max_parallel_insert=4
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # DataFrame 처리
    await process_large_dataframe_to_lightrag(df, rag)



if __name__ == "__main__":
    load_dotenv()  # .env 파일에서 환경 변수 로드
    df = pd.read_csv("./data/inputs/external/USTR/ustr_2023_press_releases_tailed.csv") 
    print(df.head())  # DataFrame의 첫 몇 행을 출력하여 확인
    asyncio.run(main()) 