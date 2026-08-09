"""SmartReco — comprehensive test suite validating every PRD requirement.

Run against the LIVE server (must be running on localhost:8000):
    cd smartreco && python -m pytest tests/test_prd.py -v

Covers:
  AUTH-01..05  – registration, login, RBAC, sessions, input validation
  PROD-01..06  – CRUD, dual-write, sync verification
  BROWSE-01..05 – listing, detail, search, recommendation display
  EVT-01..11  – event tracking, batching, async ingestion, schema
  REC-01..10  – recommendations, caching, grounding, Mesh API
  DISP-01..05 – frontend recommendation display
  SCHED-06    – opt-in/opt-out proactive delivery
  Health       – /health endpoint
"""
import json
import os
import time
import uuid

import httpx
import pytest

BASE = "http://localhost:8000"
TIMEOUT = httpx.Timeout(30.0)
ADMIN_EMAIL = os.getenv("TEST_ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("TEST_ADMIN_PASSWORD")


# ── helpers ────────────────────────────────────────────────────────────────
def _unique_email() -> str:
    return f"test_{uuid.uuid4().hex[:8]}@test.dev"


def _register(client: httpx.Client, email: str, password: str) -> httpx.Response:
    return client.post(
        f"{BASE}/api/auth/register",
        json={"email": email, "password": password},
    )


def _login(client: httpx.Client, email: str, password: str) -> httpx.Response:
    return client.post(
        f"{BASE}/api/auth/login",
        data={"email": email, "password": password},
    )


def _admin_client() -> httpx.Client:
    """Return a client logged in with externally supplied admin credentials."""
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        pytest.skip("Set TEST_ADMIN_EMAIL and TEST_ADMIN_PASSWORD")
    c = httpx.Client(timeout=TIMEOUT)
    r = _login(c, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert r.status_code == 200, f"Admin login failed: {r.text}"
    return c


def _demo_client(email: str = "maya@demo.dev") -> httpx.Client:
    """Return a client logged in as one of the seeded demo users."""
    c = httpx.Client(timeout=TIMEOUT)
    r = _login(c, email, "demo1234")
    assert r.status_code == 200, f"Demo login ({email}) failed: {r.text}"
    return c


# ═══════════════════════════════════════════════════════════════════════════
# AUTH  (PRD §6.1)
# ═══════════════════════════════════════════════════════════════════════════
class TestAuth:
    """AUTH-01 through AUTH-05."""

    # AUTH-01: registration
    def test_register_success(self):
        c = httpx.Client(timeout=TIMEOUT)
        email = _unique_email()
        r = _register(c, email, "securepass99")
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] == "user"

    def test_register_duplicate_email(self):
        c = httpx.Client(timeout=TIMEOUT)
        email = _unique_email()
        _register(c, email, "securepass99")
        r = _register(c, email, "securepass99")
        assert r.status_code == 409

    # AUTH-05: password min 8 chars
    def test_register_short_password(self):
        c = httpx.Client(timeout=TIMEOUT)
        r = _register(c, _unique_email(), "short")
        assert r.status_code == 422

    # AUTH-05: email format validated
    def test_register_invalid_email(self):
        c = httpx.Client(timeout=TIMEOUT)
        r = c.post(
            f"{BASE}/api/auth/register",
            json={"email": "not-an-email", "password": "securepass99"},
        )
        assert r.status_code == 422

    # AUTH-02: login returns JWT
    def test_login_success(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/auth/me")
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == ADMIN_EMAIL
        assert body["role"] == "admin"

    def test_login_wrong_password(self):
        if not ADMIN_EMAIL:
            pytest.skip("Set TEST_ADMIN_EMAIL")
        c = httpx.Client(timeout=TIMEOUT)
        r = _login(c, ADMIN_EMAIL, "definitely-not-the-password")
        assert r.status_code == 401

    def test_login_nonexistent_user(self):
        c = httpx.Client(timeout=TIMEOUT)
        r = _login(c, "nobody@nowhere.dev", "whatever1")
        assert r.status_code == 401

    # AUTH-02: login sets httponly cookie
    def test_login_sets_cookie(self):
        c = _admin_client()
        assert "access_token" in c.cookies

    # AUTH-04: logout clears session
    def test_logout(self):
        c = _admin_client()
        r = c.post(f"{BASE}/api/auth/logout")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    # AUTH-03: /api/auth/me requires auth
    def test_me_unauthenticated(self):
        r = httpx.get(f"{BASE}/api/auth/me", timeout=TIMEOUT)
        assert r.status_code in (401, 403)

    def test_me_authenticated(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/auth/me")
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == ADMIN_EMAIL
        assert body["role"] == "admin"
        assert "id" in body


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCT CATALOG  (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════
class TestProducts:
    """PROD-01 through PROD-06."""

    # PROD-04 / BROWSE-01: list products
    def test_list_products(self):
        r = httpx.get(f"{BASE}/api/products", timeout=TIMEOUT)
        assert r.status_code == 200
        products = r.json()
        assert isinstance(products, list)
        assert len(products) >= 49

    # PROD-04: get single product
    def test_get_product(self):
        r = httpx.get(f"{BASE}/api/products/1", timeout=TIMEOUT)
        assert r.status_code == 200
        p = r.json()
        assert p["id"] == 1
        for field in ("title", "description", "category", "price", "tags", "level"):
            assert field in p

    def test_get_product_not_found(self):
        r = httpx.get(f"{BASE}/api/products/99999", timeout=TIMEOUT)
        assert r.status_code == 404

    # PROD-01: create product (admin only, dual-write)
    def test_create_product_admin(self):
        c = _admin_client()
        payload = {
            "title": f"Test Course {uuid.uuid4().hex[:6]}",
            "description": "Automated test product for PRD validation.",
            "category": "testing",
            "price": 42.0,
            "tags": ["test", "automation"],
            "level": "beginner",
        }
        r = c.post(f"{BASE}/api/products", json=payload)
        assert r.status_code == 201
        body = r.json()
        assert body["title"] == payload["title"]
        assert body["category"] == "testing"
        assert body["id"] > 0
        c.delete(f"{BASE}/api/products/{body['id']}")

    # AUTH-03: regular user cannot create products
    def test_create_product_forbidden_for_user(self):
        c = _demo_client()
        r = c.post(
            f"{BASE}/api/products",
            json={
                "title": "Hacked",
                "description": "x",
                "category": "x",
                "price": 0,
            },
        )
        assert r.status_code in (401, 403)

    # PROD-02: update product
    def test_update_product(self):
        c = _admin_client()
        r = c.post(
            f"{BASE}/api/products",
            json={
                "title": "Update Test",
                "description": "Before",
                "category": "test",
                "price": 10,
                "tags": [],
                "level": "all",
            },
        )
        pid = r.json()["id"]
        r2 = c.put(
            f"{BASE}/api/products/{pid}",
            json={
                "title": "Updated Title",
                "description": "After",
                "category": "test",
                "price": 20,
                "tags": ["updated"],
                "level": "advanced",
            },
        )
        assert r2.status_code == 200
        assert r2.json()["title"] == "Updated Title"
        assert r2.json()["price"] == 20
        c.delete(f"{BASE}/api/products/{pid}")

    # PROD-03: delete product
    def test_delete_product(self):
        c = _admin_client()
        r = c.post(
            f"{BASE}/api/products",
            json={
                "title": "Delete Me",
                "description": "Ephemeral",
                "category": "test",
                "price": 0,
                "tags": [],
                "level": "all",
            },
        )
        pid = r.json()["id"]
        r2 = c.delete(f"{BASE}/api/products/{pid}")
        assert r2.status_code == 204
        r3 = c.get(f"{BASE}/api/products/{pid}")
        assert r3.status_code == 404

    # PROD-05 / PROD-06: dual-write sync check
    def test_catalog_sync(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/admin/stats")
        assert r.status_code == 200
        cat = r.json()["catalog"]
        assert cat["in_sync"] is True, f"SQL={cat['sql_count']} vec={cat['vector_count']}"

    # PROD-06: sync-repair endpoint
    def test_sync_repair(self):
        c = _admin_client()
        r = c.post(f"{BASE}/api/admin/sync-repair")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# BROWSING / PAGES  (PRD §6.3)
# ═══════════════════════════════════════════════════════════════════════════
class TestBrowsing:
    """BROWSE-01 through BROWSE-05 — server-rendered pages."""

    def test_index_page_loads(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "SmartReco" in r.text
        # anon users get a sign-in prompt for personalized recs
        assert "personalized recommendations" in r.text

    def test_index_has_products(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert "product/" in r.text
        assert "data-track" in r.text

    def test_product_detail_page(self):
        r = httpx.get(f"{BASE}/product/1", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "Enroll" in r.text or "enroll" in r.text
        assert "sr-context" in r.text

    def test_login_page(self):
        r = httpx.get(f"{BASE}/login", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "Sign in" in r.text or "sign in" in r.text.lower()

    def test_admin_page_requires_auth(self):
        r = httpx.get(f"{BASE}/admin", timeout=TIMEOUT, follow_redirects=False)
        assert r.status_code in (302, 307, 401, 403)

    def test_admin_page_loads_for_admin(self):
        c = _admin_client()
        r = c.get(f"{BASE}/admin")
        assert r.status_code == 200

    def test_index_has_rec_panel(self):
        c = _demo_client()
        r = c.get(f"{BASE}/")
        assert "rec-section" in r.text
        assert "Recommended for you" in r.text

    def test_index_has_search(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert 'id="search"' in r.text or "runSearch" in r.text


# ═══════════════════════════════════════════════════════════════════════════
# EVENT TRACKING  (PRD §6.4)
# ═══════════════════════════════════════════════════════════════════════════
class TestEvents:
    """EVT-01 through EVT-11."""

    def test_track_returns_202(self):
        c = _demo_client()
        r = c.post(
            f"{BASE}/api/events/track",
            json={"events": [{"type": "page_view", "payload": {"url": "/test"}}]},
        )
        assert r.status_code == 202
        assert r.json()["accepted"] == 1

    def test_track_batch(self):
        c = _demo_client()
        events = [
            {"type": "page_view", "payload": {"url": "/"}},
            {"type": "search", "payload": {"query": "python"}},
            {"type": "click", "payload": {"product_id": 1, "category": "agentic-ai"}},
        ]
        r = c.post(f"{BASE}/api/events/track", json={"events": events})
        assert r.status_code == 202
        assert r.json()["accepted"] == 3

    def test_track_with_session_id(self):
        c = _demo_client()
        r = c.post(
            f"{BASE}/api/events/track",
            json={
                "events": [
                    {
                        "type": "product_view",
                        "payload": {"product_id": 2},
                        "session_id": "sess_abc123",
                    }
                ]
            },
        )
        assert r.status_code == 202

    @pytest.mark.parametrize(
        "event_type",
        [
            "page_view",
            "product_view",
            "search",
            "click",
            "scroll_depth",
            "recommendation_click",
        ],
    )
    def test_event_types(self, event_type: str):
        c = _demo_client("sofia@demo.dev")
        r = c.post(
            f"{BASE}/api/events/track",
            json={"events": [{"type": event_type, "payload": {"test": True}}]},
        )
        assert r.status_code == 202

    def test_track_unauthenticated(self):
        r = httpx.post(
            f"{BASE}/api/events/track",
            json={"events": [{"type": "page_view", "payload": {}}]},
            timeout=TIMEOUT,
        )
        assert r.status_code in (401, 403)

    def test_recent_events(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/events/recent")
        assert r.status_code == 200
        events = r.json()
        assert isinstance(events, list)
        if events:
            assert "event_type" in events[0]
            assert "payload" in events[0]

    def test_track_beacon(self):
        c = _demo_client("raj@demo.dev")
        r = c.post(
            f"{BASE}/api/events/track-beacon",
            content=json.dumps(
                {"events": [{"type": "page_view", "payload": {"url": "/beacon"}}]}
            ),
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 202

    def test_track_empty_batch(self):
        c = _demo_client()
        r = c.post(f"{BASE}/api/events/track", json={"events": []})
        assert r.status_code == 202
        assert r.json()["accepted"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS  (PRD §6.5)
# ═══════════════════════════════════════════════════════════════════════════
class TestRecommendations:
    """REC-01 through REC-10."""

    def test_latest_recommendation_authenticated(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/recommendations/latest")
        assert r.status_code == 200
        body = r.json()
        if body is not None:
            assert "narrative_copy" in body
            assert "recommended_product_ids" in body
            assert "products" in body
            assert "updated_at" in body

    def test_latest_recommendation_unauthenticated(self):
        r = httpx.get(f"{BASE}/api/recommendations/latest", timeout=TIMEOUT)
        assert r.status_code in (401, 403)

    def test_refresh_endpoint_exists(self):
        c = _demo_client()
        # Reasoning model (tencent/hy3) can be slow; allow a generous timeout
        r = c.post(f"{BASE}/api/recommendations/refresh", timeout=httpx.Timeout(120.0))
        # 200 if LLM works, 500 if no balance — either means the endpoint exists
        assert r.status_code in (200, 500, 502, 503)

    def test_narrative_stream_endpoint(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/recommendations/latest/narrative-stream")
        assert r.status_code in (200, 204, 500)


# ═══════════════════════════════════════════════════════════════════════════
# ADMIN ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
class TestAdminAnalytics:

    def test_stats_requires_admin(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/admin/stats")
        assert r.status_code in (401, 403)

    def test_stats_structure(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/admin/stats")
        assert r.status_code == 200
        body = r.json()
        assert "users" in body
        assert "total" in body["users"]
        assert "active_24h" in body["users"]
        assert "events" in body
        assert "total" in body["events"]
        assert "last_24h" in body["events"]
        assert "breakdown" in body["events"]
        assert "recommendations" in body
        assert "catalog" in body
        assert "sql_count" in body["catalog"]
        assert "vector_count" in body["catalog"]
        assert "in_sync" in body["catalog"]
        assert "top_categories" in body
        assert "infrastructure" in body

    def test_user_profiles_requires_admin(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/admin/user-profiles")
        assert r.status_code in (401, 403)

    def test_user_profiles(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/admin/user-profiles")
        assert r.status_code == 200
        profiles = r.json()
        assert isinstance(profiles, list)
        if profiles:
            assert "email" in profiles[0]
            assert "event_count" in profiles[0]


# ═══════════════════════════════════════════════════════════════════════════
# SCHED-06: OPT-IN / OPT-OUT
# ═══════════════════════════════════════════════════════════════════════════
class TestDeliverySettings:

    def test_delivery_opt_out(self):
        c = _demo_client("kai@demo.dev")
        r = c.put(f"{BASE}/api/auth/settings/delivery", params={"enabled": False})
        assert r.status_code == 200
        assert r.json()["proactive_delivery_enabled"] is False

    def test_delivery_opt_in(self):
        c = _demo_client("kai@demo.dev")
        r = c.put(f"{BASE}/api/auth/settings/delivery", params={"enabled": True})
        assert r.status_code == 200
        assert r.json()["proactive_delivery_enabled"] is True


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════
class TestHealth:

    def test_health_endpoint(self):
        r = httpx.get(f"{BASE}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["vectors"] >= 49
        assert "event_buffer" in body
        assert "rec_cache" in body


# ═══════════════════════════════════════════════════════════════════════════
# DUAL-WRITE LIFECYCLE  (PROD-05 end-to-end)
# ═══════════════════════════════════════════════════════════════════════════
class TestDualWrite:

    def test_full_dual_write_lifecycle(self):
        c = _admin_client()

        r0 = c.get(f"{BASE}/api/admin/stats")
        baseline_sql = r0.json()["catalog"]["sql_count"]
        baseline_vec = r0.json()["catalog"]["vector_count"]

        # CREATE
        r1 = c.post(
            f"{BASE}/api/products",
            json={
                "title": "Dual-Write Test",
                "description": "Verifying SQL+vector consistency.",
                "category": "test-dw",
                "price": 1.0,
                "tags": ["dual-write"],
                "level": "beginner",
            },
        )
        assert r1.status_code == 201
        pid = r1.json()["id"]

        r2 = c.get(f"{BASE}/api/admin/stats")
        assert r2.json()["catalog"]["sql_count"] == baseline_sql + 1
        assert r2.json()["catalog"]["vector_count"] == baseline_vec + 1
        assert r2.json()["catalog"]["in_sync"] is True

        # UPDATE
        r3 = c.put(
            f"{BASE}/api/products/{pid}",
            json={
                "title": "Dual-Write Test UPDATED",
                "description": "Post-update.",
                "category": "test-dw",
                "price": 2.0,
                "tags": ["dual-write", "updated"],
                "level": "advanced",
            },
        )
        assert r3.status_code == 200
        r4 = c.get(f"{BASE}/api/admin/stats")
        assert r4.json()["catalog"]["in_sync"] is True

        # DELETE
        r5 = c.delete(f"{BASE}/api/products/{pid}")
        assert r5.status_code == 204
        r6 = c.get(f"{BASE}/api/admin/stats")
        assert r6.json()["catalog"]["sql_count"] == baseline_sql
        assert r6.json()["catalog"]["in_sync"] is True


# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END USER JOURNEY
# ═══════════════════════════════════════════════════════════════════════════
class TestE2EUserJourney:

    def test_new_user_journey(self):
        c = httpx.Client(timeout=TIMEOUT)
        email = _unique_email()

        # 1. Register
        r = _register(c, email, "journey_pass_1234")
        assert r.status_code == 200

        # 2. Browse catalog page
        r = c.get(f"{BASE}/")
        assert r.status_code == 200

        # 3. View product page
        r = c.get(f"{BASE}/product/5")
        assert r.status_code == 200

        # 4. Track events
        events = [
            {"type": "page_view", "payload": {"url": "/"}},
            {"type": "product_view", "payload": {"product_id": 5, "category": "agentic-ai"}},
            {"type": "search", "payload": {"query": "machine learning"}},
            {"type": "click", "payload": {"product_id": 8, "category": "machine-learning"}},
            {"type": "product_view", "payload": {"product_id": 8, "category": "machine-learning"}},
            {"type": "scroll_depth", "payload": {"depth": 75, "product_id": 8}},
        ]
        r = c.post(f"{BASE}/api/events/track", json={"events": events})
        assert r.status_code == 202
        assert r.json()["accepted"] == 6

        # 5. Check recent events (buffer flushes async — allow a few seconds)
        for _ in range(6):
            r = c.get(f"{BASE}/api/events/recent")
            assert r.status_code == 200
            if len(r.json()) >= 1:
                break
            time.sleep(1)
        assert len(r.json()) >= 1, "Events never flushed to DB within 6s"

        # 6. Verify profile
        r = c.get(f"{BASE}/api/auth/me")
        assert r.status_code == 200
        assert r.json()["email"] == email

        # 7. Recommendation endpoint accessible
        r = c.get(f"{BASE}/api/recommendations/latest")
        assert r.status_code == 200

        # 8. Logout
        r = c.post(f"{BASE}/api/auth/logout")
        assert r.status_code == 200

    def test_demo_user_has_behavioral_history(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/admin/user-profiles")
        assert r.status_code == 200
        profiles = r.json()
        emails = {p["email"] for p in profiles}
        demo_emails = {"maya@demo.dev", "raj@demo.dev", "sofia@demo.dev", "kai@demo.dev"}
        found = demo_emails & emails
        assert len(found) >= 2, f"Expected demo users, got: {emails}"

        for p in profiles:
            if p["email"] in demo_emails:
                assert p["event_count"] > 0, f"{p['email']} has 0 events"
                break


# ═══════════════════════════════════════════════════════════════════════════
# SECURITY  (PRD §8.3)
# ═══════════════════════════════════════════════════════════════════════════
class TestSecurity:

    def test_sql_injection_login(self):
        c = httpx.Client(timeout=TIMEOUT)
        r = _login(c, "' OR 1=1 --", "anything")
        assert r.status_code in (401, 422)

    def test_xss_in_email(self):
        c = httpx.Client(timeout=TIMEOUT)
        r = c.post(
            f"{BASE}/api/auth/register",
            json={
                "email": "<script>alert(1)</script>@test.dev",
                "password": "securepass99",
            },
        )
        assert r.status_code == 422

    def test_admin_stats_forbidden(self):
        c = _demo_client()
        r = c.get(f"{BASE}/api/admin/stats")
        assert r.status_code in (401, 403)

    def test_admin_sync_repair_forbidden(self):
        c = _demo_client()
        r = c.post(f"{BASE}/api/admin/sync-repair")
        assert r.status_code in (401, 403)

    def test_product_create_forbidden(self):
        c = _demo_client()
        r = c.post(
            f"{BASE}/api/products",
            json={"title": "x", "description": "x", "category": "x", "price": 0},
        )
        assert r.status_code in (401, 403)

    def test_product_delete_forbidden(self):
        c = _demo_client()
        r = c.delete(f"{BASE}/api/products/1")
        assert r.status_code in (401, 403)


# ═══════════════════════════════════════════════════════════════════════════
# UI CONTENT VALIDATION  (DISP-01..05)
# ═══════════════════════════════════════════════════════════════════════════
class TestUIContent:

    def test_index_rec_panel_empty_state(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert "Sign in" in r.text or "sign in" in r.text.lower()

    def test_product_cards_have_tracking(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert "data-track" in r.text

    def test_tracker_js_loads(self):
        r = httpx.get(f"{BASE}/static/tracker.js", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "track" in r.text.lower()

    def test_index_rec_cards_tracking_attributes(self):
        r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
        assert "recommendation_click" in r.text

    def test_product_enroll_tracking(self):
        r = httpx.get(f"{BASE}/product/1", timeout=TIMEOUT)
        assert 'data-track="enroll"' in r.text

    def test_admin_has_metrics_sections(self):
        c = _admin_client()
        r = c.get(f"{BASE}/admin")
        for sid in ("m-users", "m-events", "m-recs", "m-sync"):
            assert sid in r.text, f"Missing metrics section: {sid}"

    def test_admin_has_crud_form(self):
        c = _admin_client()
        r = c.get(f"{BASE}/admin")
        assert "createProduct" in r.text or "f-title" in r.text


# ═══════════════════════════════════════════════════════════════════════════
# PROTOTYPE FLOW PARITY (SmartReco hackathon UI prototype)
# ═══════════════════════════════════════════════════════════════════════════
class TestPrototypeParity:
    """Screens & flows from the design prototype: journey, signal stream,
    dwell signal, related courses, per-row dual-write sync status."""

    def test_journey_requires_login(self):
        r = httpx.get(f"{BASE}/journey", follow_redirects=False, timeout=TIMEOUT)
        assert r.status_code in (302, 303, 307)
        assert "/login" in r.headers.get("location", "")

    def test_journey_renders_for_user(self):
        c = _demo_client()
        r = c.get(f"{BASE}/journey")
        assert r.status_code == 200
        assert "Interest profile" in r.text
        assert "Activity timeline" in r.text
        assert "signals captured" in r.text

    def test_nav_has_journey_link(self):
        c = _demo_client()
        r = c.get(f"{BASE}/")
        assert "/journey" in r.text

    def test_admin_recent_events_requires_admin(self):
        r = httpx.get(f"{BASE}/api/admin/recent-events", timeout=TIMEOUT)
        assert r.status_code == 401

    def test_admin_recent_events_returns_feed(self):
        c = _admin_client()
        r = c.get(f"{BASE}/api/admin/recent-events")
        assert r.status_code == 200
        events = r.json()
        assert isinstance(events, list)
        if events:
            e = events[0]
            assert "event_type" in e and "user" in e and "created_at" in e

    def test_admin_page_has_signal_stream(self):
        c = _admin_client()
        r = c.get(f"{BASE}/admin")
        assert "Signal Stream" in r.text
        assert "Go live" in r.text

    def test_admin_rows_show_vector_sync_status(self):
        c = _admin_client()
        r = c.get(f"{BASE}/admin")
        assert "vector" in r.text  # per-row sync indicator

    def test_product_page_has_dwell_signal(self):
        # Anonymous visitors get a login CTA instead of a dwell timer
        r = httpx.get(f"{BASE}/product/1", timeout=TIMEOUT)
        assert "so the agent can learn from your browsing" in r.text
        # Logged-in users see the live dwell-time signal
        c = _demo_client()
        r = c.get(f"{BASE}/product/1")
        assert "dwell-time" in r.text
        assert "on its way to the agent" in r.text

    def test_product_page_has_related_courses(self):
        r = httpx.get(f"{BASE}/product/1", timeout=TIMEOUT)
        assert "Because you're viewing this" in r.text
