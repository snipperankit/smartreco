"""Seed the catalog: 50 courses -> SQL + vector store (batched embed call).

Run once after configuring .env:
    python seed.py
"""
import asyncio
import hashlib

from sqlalchemy import delete, select

import zlib

from app import vectorstore
from app.database import SessionLocal, init_db
from app.models import Product

# Real technology photography per category (curated Unsplash images).
# Pools are disjoint across categories and each pool is at least as large as
# the category's course count, so every course gets a globally unique photo.
_CAT_PHOTOS = {
    "agentic-ai": [
        "photo-1677442136019-21780ecad995",  # AI chip glow
        "photo-1620712943543-bcc4688e7485",  # robot hand
        "photo-1485827404703-89b55fcc595e",  # white robot
        "photo-1531746790731-6c087fecd65a",  # robot portrait
        "photo-1589254065878-42c9da997008",  # humanoid robot
        "photo-1535378917042-10a22c95931a",  # android profile
        "photo-1555255707-c07966088b7b",  # robot toy
        "photo-1607705703571-c5a8695f18f6",  # circuit macro
        "photo-1518770660439-4636190af475",  # circuit board
        "photo-1562813733-b31f71025d54",  # code neon
    ],
    "machine-learning": [
        "photo-1555949963-ff9fe0c870eb",  # code + math on screen
        "photo-1551288049-bebda4e38f71",  # analytics dashboard
        "photo-1509228468518-180dd4864904",  # math formulas
        "photo-1526628953301-3e589a6a8b74",  # geometry grid
        "photo-1573164713988-8665fc963095",  # woman + AI screen
        "photo-1591453089816-0fbb971b454c",  # deep learning book
        "photo-1542831371-29b0f74f9713",  # code glow
        "photo-1504639725590-34d0984388bd",  # dev at desk
        "photo-1516110833967-0b5716ca1387",  # notebook math
        "photo-1488229297570-58520851e868",  # monitor graphs
        "photo-1550439062-609e1531270e",  # data screen dark
        "photo-1580927752452-89d86da3fa0a",  # code green
        "photo-1573495627361-d9b87960b12d",  # analyst working
    ],
    "llm": [
        "photo-1526374965328-7f61d4dc18c5",  # green code rain
        "photo-1555066931-4365d14bab8c",  # dark code editor
        "photo-1627398242454-45a1465c2479",  # code purple
        "photo-1587620962725-abab7fe55159",  # laptop code close
        "photo-1515879218367-8466d910aaa4",  # code close-up
        "photo-1571171637578-41bc2dd41cd2",  # hologram tech
        "photo-1537432376769-00f5c2f4c8d2",  # terminal screen
    ],
    "python": [
        "photo-1498050108023-c5249f4df085",  # laptop code desk
        "photo-1461749280684-dccba630e2f6",  # laptop code
        "photo-1607799279861-4dd421887fb3",  # coding pair
        "photo-1510511459019-5dda7724fd87",  # keyboard glow
        "photo-1504384308090-c894fdcc538d",  # workspace code
    ],
    "web-dev": [
        "photo-1547658719-da2b51169166",  # responsive design
        "photo-1581291518857-4e27b48ff24e",  # UI design screen
        "photo-1559028012-481c04fa702d",  # web layout art
        "photo-1487014679447-9f8336841d58",  # macbook design
        "photo-1593720213428-28a5b9e94613",  # front-end code
    ],
    "data-engineering": [
        "photo-1518186285589-2f7649de83e0",  # data cables
        "photo-1523474253046-8cd2748b5fd2",  # data dashboard
        "photo-1551434678-e076c223a692",  # devs at screens
        "photo-1504868584819-f8e8b4b6d7e3",  # server room blue
        "photo-1519389950473-47ba0277781c",  # team laptops night
    ],
    "cloud-devops": [
        "photo-1558494949-ef010cbdcc31",  # data center
        "photo-1544197150-b99a580bb7a8",  # network
        "photo-1451187580459-43490279c0fa",  # global network
        "photo-1517180102446-f3ece451e9d8",  # server racks dark
        "photo-1484417894907-623942c8ee29",  # code monitors
        "photo-1605379399642-870262d3d051",  # dev setup dark
    ],
    "security": [
        "photo-1550751827-4bd374c3f58b",  # cyber security
        "photo-1563986768609-322da13575f3",  # padlock
        "photo-1614064641938-3bbee52942c7",  # security camera code
    ],
    "product-design": [
        "photo-1561070791-2526d30994b5",  # design desk
        "photo-1586717791821-3f44a563fa4c",  # design tools
        "photo-1541462608143-67571c6738dd",  # wireframes
    ],
    "career": [
        "photo-1521737604893-d14cc237f11d",  # team collaboration
        "photo-1522202176988-66273c2fd55f",  # laptops group
        "photo-1552664730-d307ca884978",  # whiteboard session
        "photo-1543269865-cbf427effbad",  # meeting table
        "photo-1517245386807-bb43f82c33c4",  # presentation
    ],
}
_DEFAULT_PHOTOS = ["photo-1518770660439-4636190af475"]


def _thumb(title: str, category: str, idx: int | None = None) -> str:
    photos = _CAT_PHOTOS.get(category, _DEFAULT_PHOTOS)
    # Round-robin within a category so adjacent cards never repeat.
    i = idx if idx is not None else zlib.crc32(title.encode())
    pid = photos[i % len(photos)]
    return (
        f"https://images.unsplash.com/{pid}"
        "?w=560&h=315&fit=crop&q=60&auto=format"
    )

COURSES = [
    # agentic-ai
    ("Agentic AI Systems with LangGraph", "Build stateful multi-agent workflows: nodes, edges, conditional routing, and human-in-the-loop.", "agentic-ai", 129, ["langgraph", "agents", "orchestration"], "advanced"),
    ("Designing Autonomous AI Agents", "From ReAct to planning agents: tools, memory, reflection, and evaluation loops.", "agentic-ai", 149, ["agents", "planning", "reflection"], "advanced"),
    ("RAG Engineering in Production", "Chunking, embeddings, hybrid search, re-ranking, and grounding to kill hallucinations.", "agentic-ai", 119, ["rag", "retrieval", "vector-db"], "intermediate"),
    ("Multi-Agent Collaboration Patterns", "Supervisor, swarm, and debate architectures for coordinating specialized agents.", "agentic-ai", 139, ["multi-agent", "crewai", "coordination"], "advanced"),
    ("Building MCP Servers", "Expose tools and data to LLMs with the Model Context Protocol; auth, transports, and clients.", "agentic-ai", 99, ["mcp", "tools", "integration"], "intermediate"),
    ("Tool-Calling & Function Agents", "Reliable structured outputs, schema validation, and robust tool orchestration.", "agentic-ai", 89, ["tool-use", "function-calling"], "intermediate"),
    ("Agent Evaluation & Observability", "Trace, score, and debug agent runs with LangSmith and offline eval harnesses.", "agentic-ai", 109, ["evaluation", "langsmith", "tracing"], "advanced"),

    # machine-learning
    ("Machine Learning Foundations", "Supervised learning, bias-variance, regularization, and model selection from scratch.", "machine-learning", 79, ["ml", "regression", "classification"], "beginner"),
    ("Deep Learning with PyTorch", "Tensors, autograd, CNNs, and training loops built the right way.", "machine-learning", 119, ["pytorch", "neural-nets"], "intermediate"),
    ("Transformers from Scratch", "Attention, positional encoding, and building a mini-GPT line by line.", "machine-learning", 139, ["transformers", "attention", "llm"], "advanced"),
    ("Feature Engineering Masterclass", "Encoding, scaling, leakage prevention, and pipelines that generalize.", "machine-learning", 69, ["features", "pipelines"], "intermediate"),
    ("MLOps: Ship Models to Production", "Experiment tracking, model registry, CI/CD, and monitoring for drift.", "machine-learning", 129, ["mlops", "deployment", "monitoring"], "advanced"),
    ("Recommendation Systems", "Collaborative filtering, embeddings, and modern two-tower retrieval.", "machine-learning", 99, ["recsys", "embeddings"], "intermediate"),

    # llm
    ("Prompt Engineering Deep Dive", "Systematic prompting, few-shot design, and structured output contracts.", "llm", 59, ["prompting", "llm"], "beginner"),
    ("Fine-Tuning LLMs with LoRA", "Parameter-efficient fine-tuning, datasets, and evaluation of adapters.", "llm", 149, ["fine-tuning", "lora", "peft"], "advanced"),
    ("LLM App Architecture", "Streaming, caching, guardrails, and cost control for LLM products.", "llm", 109, ["llm", "architecture", "guardrails"], "intermediate"),
    ("Vector Databases in Depth", "HNSW, cosine vs dot, metadata filtering, and hybrid search tradeoffs.", "llm", 89, ["vector-db", "search"], "intermediate"),

    # python
    ("Python for Data Professionals", "Idiomatic Python, typing, dataclasses, and clean project structure.", "python", 49, ["python", "typing"], "beginner"),
    ("Async Python Mastery", "asyncio, concurrency patterns, and high-throughput I/O services.", "python", 79, ["async", "asyncio", "concurrency"], "advanced"),
    ("FastAPI: Production APIs", "Dependency injection, auth, background tasks, and testing at scale.", "python", 89, ["fastapi", "api", "backend"], "intermediate"),
    ("Testing & Quality in Python", "pytest, fixtures, property-based testing, and coverage that matters.", "python", 59, ["testing", "pytest"], "intermediate"),

    # web-dev
    ("Modern Web Frontends", "Semantic HTML, responsive layout, and component thinking without a framework.", "web-dev", 49, ["html", "css", "frontend"], "beginner"),
    ("React from Zero to Hooks", "Components, state, effects, and data fetching patterns.", "web-dev", 89, ["react", "frontend"], "beginner"),
    ("Full-Stack with FastAPI + React", "Wire a typed backend to a reactive frontend end to end.", "web-dev", 129, ["fullstack", "fastapi", "react"], "intermediate"),
    ("Designing REST & Realtime APIs", "Resource design, pagination, websockets, and versioning.", "web-dev", 79, ["api", "rest", "websockets"], "intermediate"),

    # data-engineering
    ("Data Engineering Fundamentals", "Batch vs streaming, warehouses, and modeling for analytics.", "data-engineering", 89, ["data", "warehouse", "etl"], "beginner"),
    ("Apache Kafka in Practice", "Topics, partitions, consumer groups, and exactly-once semantics.", "data-engineering", 119, ["kafka", "streaming"], "advanced"),
    ("Building ETL Pipelines", "Orchestration, idempotency, and observability for data flows.", "data-engineering", 99, ["etl", "airflow", "pipelines"], "intermediate"),
    ("SQL for Analytics", "Window functions, CTEs, and query performance tuning.", "data-engineering", 59, ["sql", "analytics"], "beginner"),

    # cloud-devops
    ("Cloud-Native on Azure", "App Service, Functions, managed identity, and secure config.", "cloud-devops", 109, ["azure", "cloud"], "intermediate"),
    ("Docker & Containers", "Images, layers, multi-stage builds, and slim production containers.", "cloud-devops", 69, ["docker", "containers"], "beginner"),
    ("Kubernetes Essentials", "Pods, services, deployments, and scaling workloads confidently.", "cloud-devops", 129, ["kubernetes", "k8s", "orchestration"], "advanced"),
    ("CI/CD with GitHub Actions", "Pipelines, secrets, environments, and safe deployments.", "cloud-devops", 59, ["cicd", "github-actions"], "intermediate"),
    ("Observability & Tracing", "Metrics, logs, distributed tracing, and SLOs that hold.", "cloud-devops", 99, ["observability", "tracing", "sre"], "advanced"),

    # security
    ("Web Application Security", "OWASP Top 10, auth flaws, and secure-by-default design.", "security", 99, ["security", "owasp"], "intermediate"),
    ("Secrets & Identity Management", "OAuth2, JWT, and managing credentials without leaks.", "security", 89, ["oauth", "jwt", "identity"], "intermediate"),

    # product-design
    ("UX for Engineers", "Heuristics, information hierarchy, and usable interfaces.", "product-design", 49, ["ux", "design"], "beginner"),
    ("Data Visualization Craft", "Encoding, color, and charts that actually inform decisions.", "product-design", 59, ["dataviz", "design"], "intermediate"),

    # career
    ("System Design Interviews", "Scalability, tradeoffs, and structured answers to open-ended prompts.", "career", 99, ["system-design", "interview"], "advanced"),
    ("Tech Lead Playbook", "Driving technical direction, reviews, and mentoring at scale.", "career", 79, ["leadership", "career"], "advanced"),
    ("Negotiation for Engineers", "Offers, scope, and influence without authority.", "career", 49, ["career", "negotiation"], "beginner"),

    # extra ml/ai to reach 50 with coverage
    ("Computer Vision Basics", "Convolutions, augmentation, and transfer learning for images.", "machine-learning", 99, ["vision", "cnn"], "intermediate"),
    ("NLP with Transformers", "Tokenization, embeddings, and fine-tuning for text tasks.", "machine-learning", 119, ["nlp", "transformers"], "intermediate"),
    ("Reinforcement Learning Intro", "MDPs, Q-learning, and policy gradients with intuition first.", "machine-learning", 129, ["rl", "reinforcement-learning"], "advanced"),
    ("Time Series Forecasting", "Stationarity, ARIMA, and modern deep forecasting.", "machine-learning", 89, ["timeseries", "forecasting"], "intermediate"),
    ("Graph Machine Learning", "Message passing, GNNs, and knowledge-graph embeddings.", "machine-learning", 139, ["graphs", "gnn"], "advanced"),
    ("Evaluating GenAI Systems", "Faithfulness, relevance, and building trustworthy eval sets.", "agentic-ai", 109, ["evaluation", "genai"], "advanced"),
    ("Streaming LLM UIs", "Token streaming, optimistic UI, and responsive AI panels.", "llm", 69, ["streaming", "frontend", "llm"], "intermediate"),
    ("Cost Optimization for AI Apps", "Caching, routing, batching, and model tiering to cut spend.", "llm", 79, ["cost", "caching", "optimization"], "intermediate"),
    ("Building AI-Powered Recommendations", "Behavioral signals, embeddings, RAG retrieval, and persuasive narrative generation.", "agentic-ai", 119, ["recommendations", "rag", "agents"], "intermediate"),
]


async def main() -> None:
    await init_db()
    async with SessionLocal() as db:
        # Fresh start: clear SQL products (vector store is upserted by id).
        await db.execute(delete(Product))
        await db.commit()

        products = []
        cat_counts: dict[str, int] = {}
        for title, desc, cat, price, tags, level in COURSES:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            p = Product(
                title=title, description=desc, category=cat,
                price=float(price), tags=tags, level=level,
                thumbnail_url=_thumb(title, cat, cat_counts[cat] - 1),
            )
            db.add(p)
            products.append(p)
        await db.commit()
        for p in products:
            await db.refresh(p)

        payload = [
            {
                "id": p.id, "title": p.title, "description": p.description,
                "category": p.category, "price": p.price, "tags": p.tags, "level": p.level,
            }
            for p in products
        ]

    print(f"Inserted {len(payload)} products into SQL. Embedding to vector store…")
    vectorstore.upsert_many(payload)  # one batched Mesh embedding call
    print(f"Vector store now holds {vectorstore.count()} products. Done.")


if __name__ == "__main__":
    asyncio.run(main())
