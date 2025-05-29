**LightRAG Query API**

간단한 RAG(Retrieval-Augmented Generation) 엔진과 Neo4j 쿼리를 제공하는 FastAPI 기반 API입니다. JWT 인증을 적용하여 안전하게 접근할 수 있으며, 서버 시작 시 리소스를 초기화하고 재사용하도록 설계되었습니다.

---

## 📁 프로젝트 구조

```
├── app.py              # FastAPI 애플리케이션 진입점
├── requirements.txt    # Python 패키지 목록
├── Dockerfile          # 컨테이너 이미지 빌드 설정
├── docker-compose.yml  # Neo4j, Postgres, LightRAG 서비스 정의
└── README.md           # 프로젝트 설명 (이 파일)
```

## 🔧 주요 구성 요소

* **LightRAG**: LLM 기반 RAG 엔진
* **Neo4j**: 지식 그래프 저장소 및 쿼리
* **PostgreSQL+pgvector**: 벡터 저장 및 유사도 검색 (docker-compose)
* **FastAPI**: RESTful API 프레임워크
* **JWT 인증**: OAuth2 password flow 기반 액세스 토큰 발급/검증


## 🛠️ API 설계

### 인증

* **POST /login**

  * Form data: `username`, `password`
  * 응답: `{ access_token, token_type }`

### Neo4j 쿼리 실행

* **POST /run\_neo4j\_query**

  * Header: `Authorization: Bearer <token>`
  * Query param: `query` (Cypher 문자열)
  * 응답: `{ elapsed_ms, rows }`

### RAG 질의

* **POST /run\_query**

  * Header: `Authorization: Bearer <token>`
  * Body: `{ query: string, mode: naive|local|global|hybrid }`
  * 응답: `{ elapsed_ms, result }`

## 🔄 리소스 관리

* `startup` 이벤트에서 LightRAG 인스턴스와 Neo4j 드라이버를 한 번만 초기화
* 요청마다 재사용하여 초기화 오버헤드 제거

