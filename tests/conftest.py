import pytest
from fastapi.testclient import TestClient

from lighthouse import clock, data, db, orchestrator
from lighthouse.settings import settings


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """A new SQLite file with the pilot data, for every test."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "agent_timeout_s", 20.0)
    orchestrator.bus._events.clear()
    orchestrator.set_llm_factory(lambda: None)
    clock.unfreeze()
    db.init_db()
    data.generate_pilot_data()
    yield
    clock.unfreeze()


@pytest.fixture
def client(fresh_db):
    from lighthouse.api import app

    with TestClient(app) as c:
        yield c


def as_role(role: str, actor: str | None = None) -> dict:
    return {"X-Role": role, "X-Actor": actor or role}
