import os
import gc
import torch
import faiss
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
import pdfplumber
from tqdm import tqdm
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_teddynote.retrievers.ensemble import EnsembleRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-m3")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        """Compute relevance scores between query and passages"""
        pairs = [[query, passage] for passage in passages]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            scores = self.model(**inputs).logits.squeeze()
            scores = scores.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
        return scores

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on relevance to query"""
        passages = [doc.page_content for doc in documents]
        scores = self.compute_scores(query, passages)
        scored_docs = list(zip(documents, scores))
        reranked_docs = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
        return reranked_docs[:top_k]


class Config:
    def __init__(self, cache_dir: str):
        # Chunk settings
        self.max_chunks_per_load = 500
        self.chunk_size = 1400
        self.chunk_overlap = 200

        # Batch processing settings
        self.pdf_batch_size = 2
        self.embedding_batch_size = 32

        # Memory management settings
        self.clear_cuda_cache = True
        self.cleanup_interval = 1000

        # Retriever settings
        self.ensemble = True
        self.bm25_w = 0.0
        self.faiss_w = 1.0
        self.use_reranker = True
        self.rerank_top_k = 5
        
        # Cache directory
        self.cache_dir = cache_dir
        self.cache_embeddings = True

        # OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")


class EnhancedRAGSystem:
    _instance = None
    _initialized = False

    def __new__(cls, pdf_directory: str = None, cache_dir: str = None):
        if cls._instance is None:
            cls._instance = super(EnhancedRAGSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self, pdf_directory: str = None, cache_dir: str = None):
        if not self._initialized:
            if pdf_directory is None:
                raise ValueError("pdf_directory is required for first initialization")
            if cache_dir is None:
                raise ValueError("cache_dir is required for first initialization")
                
            self.config = Config(cache_dir)
            self.pdf_directory = pdf_directory
            self._llm = None
            self.reranker = self._initialize_reranker() if self.config.use_reranker else None
            self.document_mapping = {}

            # Create cache directory if it doesn't exist
            os.makedirs(self.config.cache_dir, exist_ok=True)

            # Try to load existing document mapping
            mapping_path = Path(self.config.cache_dir) / "document_mapping.json"
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.document_mapping = json.load(f)
                    print(f"Loaded existing document mapping with {len(self.document_mapping)} documents")
                except Exception as e:
                    print(f"Error loading document mapping: {str(e)}")
                    self.document_mapping = {}

            self._process_pdf_directory()
            self._setup_prompt()

            # Save updated document mapping
            try:
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(self.document_mapping, f, ensure_ascii=False, indent=2)
                print(f"Saved document mapping with {len(self.document_mapping)} documents")
            except Exception as e:
                print(f"Error saving document mapping: {str(e)}")

            EnhancedRAGSystem._initialized = True

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._setup_llm()
        return self._llm

    def _initialize_reranker(self):
        """Initialize BGE reranker"""
        try:
            print("Initializing BGE reranker...")
            return BGEReranker()
        except Exception as e:
            print(f"Warning: Failed to initialize reranker: {str(e)}")
            self.config.use_reranker = False
            return None

    def _setup_llm(self):
        """Setup LLM """
        return ChatOpenAI(
            api_key=self.config.openai_api_key,
            model="gpt-4o-mini",
            temperature=1,
            max_tokens=1024
        )

    def _get_cache_path(self, pdf_path: Path):
        """Generate cache paths for a PDF"""
        pdf_hash = str(hash(pdf_path.stem + str(pdf_path.stat().st_mtime)))
        return {
            'chunks': Path(self.config.cache_dir) / f"{pdf_path.stem}_{pdf_hash}_chunks.json",
            'embeddings': Path(self.config.cache_dir) / f"{pdf_path.stem}_{pdf_hash}_faiss"
        }

    def _extract_text_with_pdfplumber(self, file_path):
        """Extract text from PDF using pdfplumber"""
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text += f"[Page {page_num}]\n{text}\n"
        return full_text

    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF and return chunks with consistent metadata"""
        cache_paths = self._get_cache_path(file_path)

        # Check cache for chunks
        if cache_paths['chunks'].exists():
            with open(cache_paths['chunks'], 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                for chunk in cached_data:
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}
                    chunk['metadata']['source'] = file_path.stem
                return [Document(**chunk) for chunk in cached_data]

        # Extract and process text
        md_text = self._extract_text_with_pdfplumber(file_path)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Create consistent metadata
        metadata = {
            'source': file_path.stem,
            'file_path': str(file_path),
            'date_processed': datetime.now().isoformat()
        }

        chunks = text_splitter.create_documents(
            texts=[md_text],
            metadatas=[metadata]
        )

        # Cache chunks with consistent metadata
        with open(cache_paths['chunks'], 'w', encoding='utf-8') as f:
            json.dump([{
                'page_content': chunk.page_content,
                'metadata': chunk.metadata
            } for chunk in chunks], f, ensure_ascii=False, indent=2)

        return chunks

    def _process_pdf_directory(self):
        """Process PDFs and load/create vectorstore"""
        print("\nChecking for existing vectorstore...")

        vectorstore_path = Path(self.config.cache_dir) / "unified_vectorstore"

        if vectorstore_path.exists():
            try:
                print("Found existing vectorstore. Loading...")
                embeddings = OpenAIEmbeddings(
                    api_key=self.config.openai_api_key,
                    model="text-embedding-3-small",
                    dimensions=1536,
                )
                self.unified_vectorstore = FAISS.load_local(
                    str(vectorstore_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Successfully loaded existing vectorstore")

                # Initialize retrievers with loaded vectorstore
                success = self._initialize_retrievers()
                if success:
                    print("Existing system loaded successfully")
                    return
                else:
                    print("Failed to initialize retrievers. Rebuilding...")

            except Exception as e:
                print(f"Error loading vectorstore: {str(e)}")
                print("Building new vectorstore...")

        # Process PDFs and build new vectorstore
        print("\nProcessing PDFs...")
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")

        all_chunks = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            chunks = self._process_pdf(pdf_path)
            print(f"Created {len(chunks)} chunks from {pdf_path.stem}")
            all_chunks.extend(chunks)

            self.document_mapping[pdf_path.stem] = {
                'path': str(pdf_path),
                'chunk_count': len(chunks)
            }

        # Create vectorstore
        self._create_unified_vectorstore(all_chunks)

        # Initialize retrievers
        success = self._initialize_retrievers()
        if not success:
            raise Exception("Failed to initialize new system")

    def _create_unified_vectorstore(self, all_chunks: List[Document]):
        """faiss를 사용하여 HNSWFlat 인덱스로 벡터 저장소를 생성하고 저장합니다."""
        print("\n벡터 저장소 생성 시작...")
        print(f"총 {len(all_chunks)}개 청크 처리")
        
        embeddings = OpenAIEmbeddings(
            api_key=self.config.openai_api_key,
            model="text-embedding-3-small",
            dimensions=1536,
            chunk_size=100  # OpenAI API 내부 배치 크기
        )
        vectorstore_path = Path(self.config.cache_dir) / "unified_vectorstore"

        # 배치 처리로 임베딩 생성 (50개씩)
        print("\n임베딩 생성 중...")
        all_embeddings = []
        batch_size = 50
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="임베딩 배치"):
            batch = all_chunks[i:i+batch_size]
            batch_texts = [c.page_content for c in batch]
            
            try:
                batch_embeddings = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"\n배치 {i//batch_size + 1} 실패: {e}")
                raise
            
            # 메모리 정리
            if i % 200 == 0 and i > 0:
                gc.collect()

        print(f"\n✅ 임베딩 생성 완료: {len(all_embeddings)}개")

        # 임베딩을 numpy 배열로 변환
        all_embeddings_np = np.array(all_embeddings, dtype=np.float32)

        d = 1536
        M = 16
        efConstruction = 200
        efSearch = 200
        
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        index.add(all_embeddings_np)

        # FAISS 벡터 저장소 생성
        print("벡터 저장소 생성 중...")
        self.unified_vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip([c.page_content for c in all_chunks], all_embeddings_np.tolist())),
            embedding=embeddings,
            metadatas=[c.metadata for c in all_chunks]
        )

        # 저장
        print("벡터 저장소 저장 중...")
        self.unified_vectorstore.save_local(str(vectorstore_path))
        print("✅ 벡터 저장소 저장 완료!\n")

    def _initialize_retrievers(self):
        """Initialize retrievers with better error handling"""
        try:
            print("\nInitializing retrievers...")

            # Create FAISS retriever
            faiss_retriever = self.unified_vectorstore.as_retriever(search_kwargs={"k": 7})
            print("FAISS retriever initialized")

            # Get sample documents for BM25 WITH metadata
            docs = self.unified_vectorstore.similarity_search("", k=100)
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]

            # Create BM25 retriever
            kiwi_bm25_retriever = KiwiBM25Retriever.from_texts(
                texts,
                metadatas=metadatas,
                search_type="similarity"
            )
            print("BM25 retriever initialized")

            # Create ensemble retriever
            self.unified_retriever = EnsembleRetriever(
                retrievers=[kiwi_bm25_retriever, faiss_retriever],
                weights=[self.config.bm25_w, self.config.faiss_w]
            )
            print("Ensemble retriever initialized successfully")
            return True

        except Exception as e:
            print(f"Error initializing retrievers: {str(e)}")
            return False

    def _setup_prompt(self):
        """Setup prompt template"""
        template = """
        Based on the following information from multiple documents, please answer the question.
        Include the source document name when referring to specific information.

        Context:
        {context}

        Question:
        {question}

        Please provide a comprehensive answer based on all relevant information from the documents.
        Make sure to cite the source documents in your answer.

        answer in korean

        Answer:
        """

        self.prompt = PromptTemplate.from_template(template)

    def _get_relevant_context(self, query: str, document_filter: Optional[str] = None) -> str:
        """Retrieve and rerank documents to get relevant context with safe metadata handling"""
        if self.unified_retriever is None:
            raise ValueError("Retriever is not initialized")

        retrieved_docs = self.unified_retriever.invoke(query)

        # Filter by document if specified
        if document_filter:
            retrieved_docs = [
                doc for doc in retrieved_docs
                if doc.metadata.get('source', '') == document_filter
            ]

        if self.config.use_reranker and self.reranker is not None:
            try:
                reranked_docs = self.reranker.rerank(
                    query,
                    retrieved_docs,
                    top_k=self.config.rerank_top_k
                )
            except Exception as e:
                print(f"Warning: Reranking failed: {str(e)}")
                reranked_docs = retrieved_docs[:self.config.rerank_top_k]
        else:
            reranked_docs = retrieved_docs[:self.config.rerank_top_k]

        # Format context with safe metadata handling
        context_parts = []
        for doc in reranked_docs:
            try:
                source = doc.metadata.get('source', 'Unknown Source')

                metadata_items = []
                for k, v in doc.metadata.items():
                    try:
                        metadata_items.append(f"- {k}: {v}")
                    except Exception:
                        continue
                metadata_str = "\n".join(metadata_items) if metadata_items else "No metadata available"

                context_part = (
                    f"[Document: {source}]\n"
                    f"Metadata:\n{metadata_str}\n"
                    f"Content:\n{doc.page_content}\n"
                    f"{'='*50}\n"
                )
                context_parts.append(context_part)
            except Exception as e:
                print(f"Warning: Error processing document: {str(e)}")
                continue

        if not context_parts:
            return "No relevant context found."

        return "\n".join(context_parts)

    def answer_question(self, question: str, document_filter: Optional[str] = None, show_metadata: bool = True) -> str:
        """Answer a question about all or specific documents with optional metadata display"""
        if document_filter and document_filter not in self.document_mapping:
            raise ValueError(f"Document '{document_filter}' not found in the system")

        try:
            context = self._get_relevant_context(question, document_filter)

            if not show_metadata:
                context = "\n".join(
                    section.split("Content:\n")[1]
                    for section in context.split("="*50)
                    if section.strip()
                )

            # Setup RAG chain
            rag_chain = (
                {"context": lambda x: context,
                 "question": lambda x: x['question']}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            response = rag_chain.invoke({
                'question': question,
                'document_filter': document_filter
            })
            return response.strip()
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {str(e)}")
            return "Error occurred while processing the question"

    def list_documents(self) -> Dict:
        """Return information about processed documents"""
        return self.document_mapping
    
    def update_retriever_weights(self, bm25_weight: float, faiss_weight: float):
        """실행 중에 검색 가중치를 변경하고 리트리버를 재설정합니다."""
        if not self._initialized:
            raise Exception("System must be fully initialized before updating weights.")
            
        print(f"\n🔄 가중치 변경: BM25({bm25_weight}) + FAISS({faiss_weight})")
        
        # 설정값 업데이트
        self.config.bm25_w = bm25_weight
        self.config.faiss_w = faiss_weight
        
        try:
            # 1. FAISS 리트리버 재생성 (기존 unified_vectorstore 사용)
            faiss_retriever = self.unified_vectorstore.as_retriever(search_kwargs={"k": 7})
            
            # 2. BM25 리트리버 재생성 (BM25는 텍스트 데이터가 필요하므로 다시 만듭니다)
            # 안전하게 전체 청크를 가져오기 위해 넉넉하게 k=1000 사용 (문서 양이 적으면 가능)
            docs = self.unified_vectorstore.similarity_search("", k=1000) 
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            
            kiwi_bm25_retriever = KiwiBM25Retriever.from_texts(
                texts,
                metadatas=metadatas,
                search_type="similarity"
            )
            
            # 3. 앙상블 리트리버 재구성 (새로운 가중치 적용)
            self.unified_retriever = EnsembleRetriever(
                retrievers=[kiwi_bm25_retriever, faiss_retriever],
                weights=[bm25_weight, faiss_weight]
            )
            print("✅ 리트리버 업데이트 완료")
        
        except Exception as e:
            print(f"⚠️ 리트리버 업데이트 실패: {e}")
            raise


def initialize_rag_system(pdf_path: str, cache_dir: str) -> tuple:
    """
    RAG 시스템을 초기화하고 처리된 문서 목록을 반환합니다.

    Args:
        pdf_path (str): PDF 문서들이 있는 디렉토리 경로
        cache_dir (str): 캐시 디렉토리 경로

    Returns:
        tuple: (rag_system, documents) - 초기화된 RAG 시스템과 처리된 문서 목록
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)

        rag_system = EnhancedRAGSystem(pdf_path, cache_dir)
        documents = rag_system.list_documents()

        if not documents:
            raise ValueError("처리된 문서가 없습니다.")

        print(f"\n처리된 문서: {len(documents)}개")
        for doc_name, info in documents.items():
            print(f"- {doc_name}: {info['chunk_count']} chunks")

        return rag_system, documents

    except Exception as e:
        print(f"초기화 오류: {str(e)}")
        raise


def ask_question(rag_system, question: str, document_filter: Optional[str] = None, 
                show_metadata: bool = False, show_retrieved_context: bool = False):
    """
    RAG 시스템에 질문하고 결과를 출력합니다.

    Args:
        rag_system: 초기화된 RAG 시스템
        question (str): 질문
        document_filter (Optional[str]): 특정 문서로 필터링 (선택사항)
        show_metadata (bool): 메타데이터 표시 여부
        show_retrieved_context (bool): 검색된 컨텍스트 표시 여부

    Returns:
        dict: 질문과 답변 정보
    """
    try:
        print(f"\n{'='*50}")
        print("질문:")
        print(question)
        
        if document_filter:
            print(f"\n문서 필터: {document_filter}")

        # 컨텍스트 가져오기
        context = rag_system._get_relevant_context(question, document_filter)
        
        # 답변 생성
        answer = rag_system.answer_question(
            question=question,
            document_filter=document_filter,
            show_metadata=False  # 답변에는 메타데이터 포함 안 함
        )

        print(f"\n{'='*50}")
        print("답변:")
        print(answer)

        # 메타데이터 출력
        if show_metadata:
            print(f"\n{'='*50}")
            print("참조된 문서 메타데이터:")
            contexts = context.split("="*50)
            metadata_count = 0
            for doc in contexts:
                if not doc.strip():
                    continue
                if "[Document:" in doc:
                    metadata_count += 1
                    meta_section = doc.split("Content:")[0].strip()
                    print(f"\n--- 메타데이터 {metadata_count} ---")
                    print(meta_section)

        # 검색된 컨텍스트 출력
        if show_retrieved_context:
            print(f"\n{'='*50}")
            print("검색된 컨텍스트:")
            contexts = context.split("="*50)
            context_count = 0
            for doc in contexts:
                if not doc.strip():
                    continue
                if "Content:" in doc:
                    context_count += 1
                    content = doc.split("Content:")[1].strip()
                    print(f"\n--- 컨텍스트 {context_count} ---")
                    print(content)

        print(f"\n{'='*50}")

        return {
            'question': question,
            'answer': answer,
            'document_filter': document_filter,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"질문 처리 오류: {str(e)}")
        return None


if __name__ == "__main__":
    try:
        # 경로 설정
        base_path = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(base_path, "pdfs")
        cache_dir = os.path.join(base_path, "cache")

        # 시스템 초기화
        print("RAG 시스템 초기화 중...")
        rag_system, documents = initialize_rag_system(pdf_path, cache_dir)

        if not documents:
            print("경고: 처리된 문서가 없습니다.")
            exit(1)

        # 질문 예시
        question = "청약 가점이 낮은 20대가 실제로 당첨 가능성을 높일 수 있는 방법에는 어떤 것들이 있을까요?"
        print("\n질문 처리 중...")

        result = ask_question(
            rag_system=rag_system,
            question=question,
            show_metadata=True,
            show_retrieved_context=True
        )

        if result:
            print("\n✅ 처리 완료!")

    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()