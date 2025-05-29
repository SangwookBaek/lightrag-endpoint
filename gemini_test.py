import os
import asyncio
from lightrag import LightRAG, QueryParam
# from lightrag.llm.openai import (
#     gpt_4o_mini_complete,
#     openai_embed,
# )

from lightrag.llm.gemini import (
    gemini_2_0_flash_complete,
    gemini_embed,
)
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from dotenv import load_dotenv

load_dotenv()
setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=gemini_embed,
        llm_model_func=gemini_2_0_flash_complete,

        # ↓ PostgreSQL + pgvector 로 바꾸는 부분
        vector_storage="PGVectorStorage",
        kv_storage = "PGKVStorage",
        graph_storage = "Neo4JStorage",
        doc_status_storage = "PGDocStatusStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    rag = None
    try:
        rag = await initialize_rag()
        # await 키워드 추가!
        await rag.ainsert("Marcus Lee had twenty-seven dollars in his checking account and a sticky note on his fridge that read, “Succeed or starve.” He wasn’t dramatic by nature. The sticky note was practical. Inspirational, even. He had written it after watching a free webinar called Unleash the Inner Giant Within You, hosted by Bryce Chandler, America’s favorite millionaire-turned-messianic-life-coach—a man who looked like a cross between an energy drink and a televangelist caught mid-exorcism.")  # insert -> ainsert

        resp = await rag.aquery(  # query -> aquery
            "What are the top themes in this story?",
            param=QueryParam(mode="hybrid")
        )
        print("▶ 결과:", resp)

    except Exception as e:
        print("❌ 에러 발생:", e)
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()

# 이벤트 루프 체크 후 실행
if __name__ == "__main__":
    try:
        # 현재 실행 중인 이벤트 루프가 있는지 확인
        loop = asyncio.get_running_loop()
        # 이미 실행 중이면 태스크로 실행
        asyncio.create_task(main())
    except RuntimeError:

        # 실행 중인 루프가 없으면 새로 시작
        asyncio.run(main())