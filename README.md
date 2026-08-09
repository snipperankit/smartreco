SmartReco — Behavioral AI Recommendation Agent
An agentic recommendation system that observes user behavior in real-time, reasons with an 8-node LangGraph workflow, retrieves via hybrid RAG (vector + BM25 + RRF fusion), and proactively delivers personalized recommendations through multi-channel push (in-app + Telegram + SSE). All model calls are routed through Mesh API.

Quick Start
pip install -r requirements.txt

# Set MESH_API_KEY, SECRET_KEY, ADMIN_EMAIL, and ADMIN_PASSWORD
cp .env.example .env

python run.py          # seeds the course catalog on first run
Open http://localhost:8000 — browse courses, and watch recommendations appear automatically.

Accounts
The admin account is created from ADMIN_EMAIL and ADMIN_PASSWORD; no credentials are stored in the repository. Public registration creates learner accounts. Optional demo personas are disabled by default and can be enabled only in a disposable local environment with SEED_DEMO_USERS=true.

What It Does
Observes — Tracks views, clicks, searches, scroll depth, and dwell time via a non-blocking JS SDK with localStorage buffering and sendBeacon on unload
Buffers — Events are batched in-memory (asyncio.Queue, 10k capacity) and bulk-flushed to SQLite every 1s with dedup within a 3s window and backpressure tracking
Triggers — A multi-gate system decides whether to invoke the agent: meaningful signal check → 45s per-user cooldown → behavior-signature dedupe (SHA-1 of recent events). Admin users are excluded
Profiles — Builds weighted interest profiles using exponential temporal decay (half-life=10 events), dwell-time bonuses (30s+ doubles weight), and event-type weighting (search=3x, click=2x, view=1x)
Strategizes — Computes engagement concentration (top-category score / total score) and selects a retrieval strategy: cold_start | progression | broadening | balanced | explorer_*. Explorer strategies use the fast model for intent classification (cross_domain | undecided | variety_seeker)
Retrieves — Three-stage hybrid search: (1a) cosine vector retrieval from ChromaDB, (1b) BM25 keyword scoring, (2) Reciprocal Rank Fusion (k=60), (3) multi-signal reranking with category/sub-topic/level boosts. Explorer strategies use multi-query retrieval (one per top category, merge best-of-each)
Grades — If retrieval score < 0.30, broadens the query and retries retrieval (max 1 refine loop)
Generates — Reasoning model (tencent/hy3) writes a persuasive 2-4 sentence narrative grounded strictly in the shortlisted courses. Shortlist is diversity-constrained (max 2 per category). Falls back to template narrative on rate-limit
Reflects — Quality gate with deterministic checks (generic-phrase detection, strategy-alignment, diversity) + LLM-powered evaluation via fast model (minimax/m2-her). Regenerates once if quality fails
Delivers — Multi-channel: in-app notification stack (slide-in cards), SSE real-time push, in-app mailbox, and Telegram Bot API. Scheduled digest with content-fingerprint deduplication
Multi-Model Architecture
Models are routed by task through one Mesh client:

Model	Role	Used For
tencent/hy3 (262K, Thinking)	Creative reasoning	Narrative generation (1 call per agent run)
minimax/m2-her (32K, Text)	Fast utility	Intent classification, reflection quality check
openai/text-embedding-3-small	Embeddings	Course catalog vectorization when the Mesh account has model access
Resilience: Primary model 429 → fast model → template narrative. Embeddings are cached by SHA-256 content hash. If the configured Mesh embedding model is unavailable, retrieval uses deterministic feature-hashed vectors in Chroma; this fallback performs no external AI call and keeps vector + BM25 retrieval available without downloading a large local model.

Agent Workflow (LangGraph)
analyze_behavior → plan → retrieve → grade ──(good)──→ generate → reflect ──(pass)──→ END
                                ↑        │                  ↑          │
                                └─refine─┘                  └─regen────┘
8-node StateGraph with MemorySaver checkpointer (persists state after each node for debugging/replay):

Node	What It Does
analyze_behavior	Exponential decay weighting, dwell-time bonus, TF-IDF query construction, engagement concentration scoring
plan	Strategy selection based on concentration + event count + category spread; fast-model intent classification for explorers
retrieve	Single-query or multi-query (per-category for explorers), category + sub-topic + level boosting, hybrid vector+BM25
grade	Conditional edge: retrieval score ≥ 0.30 → generate, else → refine (max 1 retry)
refine	Broadens query to category names and retries retrieval
generate	Persuasive narrative via reasoning model; diversity-constrained shortlist (max 2 per category)
reflect	Deterministic checks (generic phrases, strategy alignment, diversity) + LLM quality check via fast model
regenerate	Re-runs generate once with the same shortlist if reflection fails
Strategy Selection Logic
Condition	Strategy	Retrieval Mode
< 3 events	cold_start	Single query
Concentration > 0.65, 1 hot category	progression	Level=advanced
Concentration < 0.4, 3+ hot categories	explorer_* (classified)	Multi-query
Concentration < 0.4, < 3 categories	broadening	Single query
Everything else	balanced	Level=intermediate
Explorer Sub-Strategies (Fast-Model Intent Classification)
When browsing is spread across 3+ categories, the fast model classifies intent:

explorer_cross_domain — interested in intersection of topics (e.g. AI + business)
explorer_undecided — hasn't found their niche, needs gateway courses
explorer_variety_seeker — deliberately wants breadth, one from each area
Each sub-strategy produces a different persuasion angle for the narrative generator.

End-to-End Code Flow
1. Event Ingestion
Browser interaction
  → tracker.js buffers in localStorage (survives navigation)
  → Flush: batch ≥ 3 events OR 3s timer OR page-hide (sendBeacon)
  → POST /api/events/track (EventBatch schema validation)
  → event_buffer.push() — asyncio.Queue (max 10k, backpressure drops)
    → Dedup: same (user_id, event_type, product_id) within 3s → skip
  → Background flush loop (every 1s): bulk INSERT to SQLite
  → Speculative pre-warm: search/click events kick off background retrieval
  → Background task: maybe_generate(user_id) (only if user.role != "admin")
Key files: app/static/tracker.js → app/routers/events.py → app/event_buffer.py → app/speculative.py

2. Recommendation Generation
maybe_generate(db, user_id)
  → Load last 20 events
  → _is_meaningful(events)?
    → Search event? ✓ | Click with product_id? ✓
    → ≥2 views in one category? ✓ | ≥4 total engagement events? ✓
  → Cooldown: last rec < 45s ago? → skip
  → Behavior signature: SHA-1(event_types + categories + product_ids + queries)
    → Same as last rec's signature? → skip (no LLM call burned)
  → asyncio.to_thread(run_recommendation, user_id, events)
    → LangGraph: analyze → plan → retrieve → grade → generate → reflect → END
  → Save Recommendation(is_sent=False, behavior_signature=sig)
  → rec_cache.invalidate(user_id) — next /latest poll reads fresh
Key files: app/agent/triggers.py → app/agent/graph.py → app/rec_cache.py

3. Recommendation Delivery
GET /api/recommendations/latest (polled by frontend)
  → rec_cache.get(user_id) — LRU cache, 900s TTL, 50k entries max
    → Hit: return cached (absorbs ~10k reads/sec at 100k users)
    → Miss: query DB, cache result, return

GET /api/recommendations/stream (SSE EventSource on homepage)
  → Polls DB every 3s, pushes full rec JSON on change
  → Frontend renders notification stack (slide-in cards, staggered 300ms, auto-dismiss 8s)

Scheduled digest (APScheduler):
  → Cron (daily at DIGEST_HOUR:DIGEST_MINUTE UTC) or Interval (every N minutes)
  → For each active user (events in last 24h, delivery enabled, not admin):
    → Find latest unsent Recommendation
    → Content fingerprint: SHA-256(sorted(recommended_product_ids))[:16]
    → Compare to user.last_digest_hash:
      → Same → mark is_sent=True, skip delivery (no duplicate sends)
      → Different → deliver() → update last_digest_hash
  → deliver(): mailbox (always) + Telegram (if enabled in DELIVERY_CHANNELS)
Key files: app/routers/recommendations.py → app/rec_cache.py → app/scheduler.py

4. Hybrid Search (Vector + BM25 + RRF)
vectorstore.search(query, k=8, exclude_ids, category_boost, level_pref)
  → Stage 1a: ChromaDB cosine retrieval (top 24)
  → Stage 1b: BM25 keyword scoring (Python implementation, zero deps)
    → Tokenize query + all indexed docs
    → IDF weighting, TF saturation (k1=1.5, b=0.75)
  → Stage 2: Reciprocal Rank Fusion (k=60)
    → RRF_score(pid) = 1/(60+vec_rank) + 1/(60+bm25_rank)
    → Surfaces docs ranked highly by EITHER signal
  → Stage 3: Multi-signal rerank
    → Category boost: +0.15 if in hot_categories
    → Sub-topic boost: +0.05 per matching keyword in title/description
    → Level preference: +0.10 if matches level_pref
  → Return top-k with scores, metadata, and doc text
Key file: app/vectorstore.py

5. Dual-Write Consistency
Product CRUD (admin):
  → POST/PUT/DELETE /api/products/{id}
    → SQLAlchemy write → vectorstore.upsert_product() / delete_product()
    → upsert_product: SHA-1 content hash → re-embed only if content changed
      → Metadata-only update (price change) skips Mesh embedding call
    → BM25 index updated in-process on every upsert/delete

Admin sync-repair:
  → POST /api/admin/sync-repair
    → Reads all products from SQL → re-upserts into ChromaDB
    → Returns {repaired, vector_count, sql_count, in_sync}

Health check:
  → GET /health → { vectors: count, in_sync: sql==vector }
Key files: app/routers/products.py → app/vectorstore.py → app/routers/admin_analytics.py

Proactive Delivery System
Scheduler Modes
Mode	Config	Behavior
Cron	DIGEST_HOUR=15, DIGEST_MINUTE=0	Runs once daily at 15:00 UTC
Interval	DIGEST_INTERVAL_MINUTES=1	Runs every N minutes (for demo/testing)
Digest Pipeline
Query active users: had events in last 24h, proactive_delivery_enabled=True, role != "admin"
For each user, find latest Recommendation where is_sent=False
Content fingerprint dedup: compute SHA-256(sorted(product_ids))[:16], compare to user.last_digest_hash. If identical, mark sent and skip (no duplicate notification for unchanged recommendations)
Hydrate product details from SQL
Deliver via all enabled channels, update last_digest_hash
Delivery Channels
Channel	Config	Implementation
Mailbox	Always on	In-memory deque (50 entries), viewable at admin dashboard
Telegram	DELIVERY_CHANNELS=mailbox,telegram	Bot API via httpx.post, HTML-formatted cards with level icons (🌱/⚡/🔥)
WhatsApp	DELIVERY_CHANNELS=mailbox,whatsapp	CallMeBot API (HTTP GET), Markdown-formatted cards
SSE	Always on	GET /api/recommendations/stream — EventSource on homepage
Telegram Card Format
🧠 SmartReco

Hey Maya, new picks just for you:

"Based on your deep interest in agentic AI..."

🌱 Building AI-Powered Recommendations
     Agentic Ai • $49 • Beginner

⚡ Advanced RAG Patterns
     Agentic Ai • $89 • Intermediate

────────────────────────────
📊 Personalized from your activity
🤖 LangGraph + Mesh API
User Opt-Out
PUT /api/auth/settings/delivery?enabled=false — sets proactive_delivery_enabled=False, scheduler skips user.

Auto-Triggering Pipeline
Browser click → tracker.js (flush every 3s or 3 events)
  → POST /api/events/track → event_buffer (flush to DB every 1s)
  → speculative_cache.speculate() on search/click events
  → background task: maybe_generate()
    → meaningful signal? → cooldown check → behavior dedupe
    → run_recommendation() in worker thread (asyncio.to_thread)
    → save to DB → SSE pushes to browser → notification stack appears
Meaningful signals (any one triggers the agent):

Any search event (highest-intent signal)
Any click with a product_id (deliberate engagement)
≥2 views in one category (focused interest)
≥4 total engagement events across any categories (handles random browsing)
Guards (prevent wasted LLM calls):

45-second per-user cooldown
Behavior signature dedupe: SHA-1 hash of (event_types, categories, product_ids, queries) — identical browsing pattern reuses existing recommendation
Admin role exclusion: user.role != "admin" skips agent entirely
Speculative Pre-Computation
CPU-style speculative execution for near-zero-latency recommendations:

On every search or click event, speculative_cache.speculate() kicks off ChromaDB retrieval in a background thread
When the user crosses the trigger threshold and maybe_generate() fires, the retrieval is already cached
Hit: latency drops from ~1500ms to ~400ms (just the LLM call). Miss: one wasted Chroma query (cheap)
Per-user cache, 30s TTL, LRU-bounded (10k entries)
Push Notifications
When the agent produces new recommendations, stacked notification cards slide in from the bottom-right:

One card per recommended course (thumbnail, title, category, price)
Staggered entrance (300ms apart)
Auto-dismiss after 8 seconds (+ 800ms offset per card)
Manual dismiss via ✕ button
Click "View course →" navigates to product page
Live Activity Feed
Real-time sidebar shows tracked events as they happen:

👁 Page views
🖱 Clicks
🔍 Searches
Polls every 3 seconds + instant display on client-side events
API Endpoints
Auth
Method	Path	Description
POST	/api/auth/register	Create account
POST	/api/auth/login	Login (sets JWT cookie)
POST	/api/auth/logout	Clear session
GET	/api/auth/me	Current user
PUT	/api/auth/settings/delivery	Opt in/out of proactive digest
Products (admin-only mutations, dual-write to SQL + ChromaDB)
Method	Path	Description
GET	/api/products	List all
GET	/api/products/{id}	Get one
POST	/api/products	Create + embed
PUT	/api/products/{id}	Update (re-embeds only if content changed)
DELETE	/api/products/{id}	Remove from SQL + vector DB
Events
Method	Path	Description
POST	/api/events/track	Batch ingestion (triggers agent in background)
POST	/api/events/track-beacon	sendBeacon endpoint for page unload
GET	/api/events/recent	Last 10 events for activity feed
Recommendations
Method	Path	Description
GET	/api/recommendations/latest	Cached latest (LRU, 900s TTL)
POST	/api/recommendations/refresh	Force regeneration (respects dedupe)
GET	/api/recommendations/stream	SSE — pushes full rec on change
Admin
Method	Path	Description
GET	/api/admin/stats	System metrics + infrastructure health
GET	/api/admin/recent-events	Global event stream (last 30, all users)
GET	/api/admin/user-profiles	Per-user behavioral profiles
POST	/api/admin/sync-repair	Fix SQL↔vector drift
GET	/api/admin/mailbox	View delivered digest messages
POST	/api/admin/trigger-digest	Run digest immediately (demo)
Pages
Path	Description
/	Course catalog + recommendations + activity feed
/login	Auth form
/product/{id}	Course detail + related courses
/journey	User activity dashboard (interest bars, event timeline)
/admin	Admin dashboard (stats, profiles, sync status)
/agent	LangGraph workflow visualization
/health	System health (vectors, event buffer, rec cache)
Project Structure
smartreco/
├── run.py                     # Single entry point (seeds + starts)
├── seed.py                    # 50 courses across 10 categories
├── seed_demo.py               # 4 demo personas with behavioral history
├── requirements.txt           # 18 dependencies
├── Procfile                   # Render/Railway deployment
├── render.yaml                # Render.com service definition
├── .env.example               # All config vars documented
├── .github/workflows/
│   └── smartreco-checks.yml   # CI: lint + test on push/PR
├── app/
│   ├── main.py                # FastAPI app + lifespan (init DB, bootstrap admin, start scheduler + buffer)
│   ├── config.py              # Pydantic Settings (45 config vars, all .env-driven)
│   ├── database.py            # SQLAlchemy async engine + session factory + init_db
│   ├── models.py              # User, Product, BehavioralEvent, Recommendation (4 tables)
│   ├── schemas.py             # Pydantic request/response models (EventBatch, RecommendationOut, etc.)
│   ├── security.py            # JWT (HS256) + password hashing (pbkdf2_sha256)
│   ├── deps.py                # FastAPI dependency injection (auth from Bearer header or cookie)
│   ├── mesh.py                # All Mesh API calls: chat(), chat_fast(), chat_stream(), embed(), embed_many()
│   ├── vectorstore.py         # ChromaDB + BM25 + RRF hybrid search + content-hash embedding cache
│   ├── event_buffer.py        # Async event buffer (asyncio.Queue, 1s flush, 3s dedup, 10k backpressure)
│   ├── rec_cache.py           # In-memory LRU recommendation cache (900s TTL, 50k entries)
│   ├── speculative.py         # Speculative pre-computation cache (30s TTL, 10k entries)
│   ├── scheduler.py           # APScheduler digest + multi-channel delivery (mailbox/Telegram/WhatsApp)
│   ├── routers/
│   │   ├── auth.py            # Register, login, logout, me, delivery settings
│   │   ├── products.py        # CRUD with dual-write (SQL + ChromaDB)
│   │   ├── events.py          # Track (buffered), track-beacon (sendBeacon), recent
│   │   ├── recommendations.py # Latest (cached), refresh (force), SSE stream
│   │   ├── admin_analytics.py # Stats, profiles, sync-repair, mailbox, trigger-digest
│   │   └── pages.py           # Server-rendered HTML: index, login, product, journey, admin, agent
│   ├── agent/
│   │   ├── graph.py           # 8-node LangGraph StateGraph with MemorySaver checkpointer
│   │   └── triggers.py        # Signal gate + cooldown + behavior-signature dedupe
│   ├── screens/               # Jinja2 templates (Tailwind CSS)
│   │   ├── base.html          # Layout with nav, notification stack, activity feed
│   │   ├── index.html         # Course catalog grid + recommendation section + SSE
│   │   ├── login.html         # Auth form
│   │   ├── product.html       # Course detail + related courses
│   │   ├── journey.html       # User dashboard: interest bars, event timeline
│   │   ├── admin.html         # Admin: system stats, user profiles, sync status
│   │   └── agent.html         # LangGraph workflow visualization
│   └── static/
│       └── tracker.js         # Non-blocking behavioral tracker (localStorage + sendBeacon)
└── tests/
    └── test_prd.py            # 77 tests covering all PRD requirements
Tech Stack
Layer	Technology
Backend	FastAPI 0.115.6 (async, lifespan context manager)
Database	SQLite + SQLAlchemy 2.0.36 (async via aiosqlite)
Vector DB	ChromaDB 0.5.23 (embedded, persistent, cosine similarity)
LLM/Embeddings	Mesh API (OpenAI-compatible gateway, 3 models)
Agent	LangGraph 0.2.60 (StateGraph, conditional edges, MemorySaver)
Scheduler	APScheduler 3.11.0 (in-process, cron or interval mode)
Frontend	Jinja2 + Tailwind CDN + vanilla JS
Tracking	Custom JS SDK (batch, throttle, sendBeacon, localStorage)
Auth	python-jose (JWT HS256) + passlib (pbkdf2_sha256)
Observability	LangSmith tracing (opt-in), /health endpoint, admin dashboard
Testing	pytest (77 tests)
Configuration
Variable	Required	Default	Purpose
MESH_API_KEY	Yes	—	Mesh API key (rsk_...)
MESH_BASE_URL	No	https://api.meshapi.ai/v1	Mesh endpoint
MESH_CHAT_MODEL	No	tencent/hy3	Primary reasoning model
MESH_FAST_MODEL	No	minimax/m2-her	Fast utility model
MESH_EMBED_MODEL	No	openai/text-embedding-3-small	Embedding model
APP_ENV	Yes (production)	development	Enables strict production validation
SECRET_KEY	Yes (production)	Ephemeral in development	JWT signing key, minimum 32 characters
COOKIE_SECURE	No	false	Forced to true in production
ENABLE_API_DOCS	No	true	Forced to false in production
ENABLE_RATE_LIMITING	No	true	Forced to true in production
ACCESS_TOKEN_EXPIRE_MINUTES	No	1440	JWT token expiry (24h)
DATABASE_URL	No	sqlite+aiosqlite:///./smartreco.db	SQLAlchemy connection string
CHROMA_DIR	No	./chroma_store	ChromaDB persistence directory
REC_COOLDOWN_SECONDS	No	45	Min seconds between agent runs per user
REC_CATEGORY_VIEW_THRESHOLD	No	2	Views in one category to trigger
REC_TOTAL_ENGAGEMENT_THRESHOLD	No	4	Total events across categories to trigger
ENABLE_SCHEDULER	No	true	Enable APScheduler digest
DIGEST_HOUR	No	15	Hour (UTC) for daily cron digest
DIGEST_MINUTE	No	0	Minute for daily cron digest
DIGEST_INTERVAL_MINUTES	No	0	>0: interval mode (every N min)
DELIVERY_CHANNELS	No	mailbox	Comma-separated: mailbox,telegram
TELEGRAM_BOT_TOKEN	No	—	Telegram Bot API token (@BotFather)
TELEGRAM_CHAT_ID	No	—	Target chat for digest delivery
TELEGRAM_RECIPIENT_EMAIL	With Telegram	—	Only this learner may receive that chat's digest
LANGSMITH_TRACING	No	false	Enable LangSmith traces
LANGCHAIN_API_KEY	No	—	LangSmith API key
LANGCHAIN_PROJECT	No	smartreco	LangSmith project name
ADMIN_EMAIL	Yes (production)	—	Auto-bootstrapped admin account
ADMIN_PASSWORD	Yes (production)	—	Admin password, minimum 12 characters
SEED_DEMO_USERS	No	false	Local-only disposable demo personas
Running Tests
# Start the server first
python run.py &

# Run all 77 tests
python -m pytest tests/ -q
Tests validate: auth flows, product CRUD + dual-write consistency, event tracking + batched ingestion, recommendation generation + caching + grounding in real catalog, admin analytics endpoints, scheduler delivery + content dedup, and SSE streaming.

Catalog
50 courses across 10 categories with curated Unsplash thumbnails:

agentic-ai (7): LangGraph, agents, RAG, MCP, tool-calling, evaluation, AI-powered recommendations
machine-learning (13): PyTorch, transformers, MLOps, recsys, CV, NLP, RL
llm (4): Prompt engineering, fine-tuning, architecture, vector databases
python (4): Data, async, FastAPI, testing
web-dev (4): HTML/CSS, React, full-stack, APIs
data-engineering (4): Fundamentals, Kafka, ETL, SQL
cloud-devops (5): Azure, Docker, Kubernetes, CI/CD, observability
security (2): Web security, OAuth/JWT
product-design (2): UX, data visualization
career (3): System design interviews, tech lead, negotiation
Each course has a unique photo from its category's Unsplash pool (deterministic assignment via CRC32 hash).

Deployment
Deploy anywhere that supports a persistent process (not serverless — needs SQLite + ChromaDB + APScheduler on disk).

Render (recommended, free)
Already configured via render.yaml. Push to GitHub → connect in Render dashboard → set MESH_API_KEY env var → deploy.

# Or manually:
pip install -r requirements.txt
python run.py  # seeds data + starts uvicorn on $PORT
Railway / Fly.io
Uses the Procfile:

web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
Efficiency Patterns
Pattern	Implementation	Impact at Scale
Event buffering	event_buffer.py — asyncio.Queue, 1s bulk flush, 3s dedup window	4 INSERTs/sec vs 8k commits/sec
Recommendation cache	rec_cache.py — LRU with 900s TTL, 50k entries	Absorbs ~10k reads/sec
Speculative pre-compute	speculative.py — background retrieval on search/click signals	1500ms → 400ms trigger latency
Trigger gating	triggers.py — cooldown + signature dedupe + threshold filtering	~90% of events skip agent entirely
Multi-model routing	mesh.py — cheap model for classification, expensive for narrative	1 creative LLM call per agent run
Dual-write + content hashing	vectorstore.py — SHA-256 content hash, re-embed only on change	Price edits skip Mesh call
Hybrid search (vector + BM25)	vectorstore.py — RRF fusion of cosine + BM25 ranked lists	Better recall than vector-only
Diversity constraint	graph.py — max 2 per category in shortlist	Prevents filter-bubble recs
Content fingerprint dedup	scheduler.py — SHA-256 of sorted product IDs per user	No duplicate digests on re-run
Non-blocking tracking	tracker.js — localStorage buffer, sendBeacon on unload	Zero data loss on navigation
Streaming generation	mesh.py — chat_stream() yields tokens as they arrive	200ms first-token vs 1500ms wait
Backpressure tracking	event_buffer.py — drops + logs on queue full (10k cap)	OOM protection at traffic spikes
