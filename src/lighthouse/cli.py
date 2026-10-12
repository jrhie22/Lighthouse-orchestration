"""Command line: ``lighthouse <command>`` (or ``python -m lighthouse.cli <command>``).

    serve          start the API + web page            lighthouse serve --reload
    init           create the database + pilot data    lighthouse init
    generate       bulk-create simulated cases         lighthouse generate --n 5000 --db data/bulk.db
    verify-audit   recompute the audit hash chain      lighthouse verify-audit
    ingest-csv     load a real weekly-sales CSV        lighthouse ingest-csv file.csv --store-col Store --date-col Date --sales-col Weekly_Sales
    check-llm      is Ollama up, is the model there?   lighthouse check-llm
    eval-llm       model vs. rules on N random cases   lighthouse eval-llm --db data/eval.db --n 20
"""
from __future__ import annotations

import argparse
import json
import sys

from . import audit, data, db
from .settings import settings


def _use_db(path: str | None) -> None:
    if path:
        settings.db_path = path
    db.init_db()


def cmd_serve(a) -> int:
    import uvicorn

    if a.db:
        import os

        os.environ["LIGHTHOUSE_DB"] = a.db  # the reloader starts a new process; the env var survives it
    uvicorn.run("lighthouse.api:app", host=a.host, port=a.port, reload=a.reload)
    return 0


def cmd_init(a) -> int:
    _use_db(a.db)
    print(json.dumps(data.generate_pilot_data(force=a.force), indent=2))
    print(f"database: {settings.db_path}")
    return 0


def cmd_generate(a) -> int:
    from . import bulk

    _use_db(a.db)
    data.generate_pilot_data()
    print(f"database: {settings.db_path} - generating {a.n} simulated cases (rules agents; seed {a.seed}) ...")
    summary = bulk.generate(a.n, a.seed, progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    print(json.dumps(summary, indent=2))
    with db.read() as conn:
        v = audit.verify(conn)
    print("audit chain:", "OK" if v["ok"] else f"BROKEN at seq {v.get('first_bad_seq')}", f"({v.get('entries')} entries)")
    return 0 if v["ok"] else 1


def cmd_verify(a) -> int:
    _use_db(a.db)
    with db.read() as conn:
        v = audit.verify(conn)
    print(json.dumps(v, indent=2))
    return 0 if v["ok"] else 1


def cmd_ingest(a) -> int:
    _use_db(a.db)
    out = data.ingest_sales_csv(
        a.csv, store_col=a.store_col, date_col=a.date_col, sales_col=a.sales_col, promo_col=a.promo_col,
        category=a.category, replace_generated=a.replace,
    )
    print(json.dumps(out, indent=2))
    return 0


def cmd_eval_llm(a) -> int:
    from . import eval_llm
    from .llm import OllamaClient

    _use_db(a.db)
    data.generate_pilot_data()
    if not OllamaClient().available():
        print(f"Ollama model '{settings.ollama_model}' is not reachable at {settings.ollama_host}. Run: lighthouse check-llm")
        return 1
    print(f"database: {settings.db_path}; model: {settings.ollama_model}; {a.n} cases (slow: several model calls per agent) ...")
    print(json.dumps(eval_llm.evaluate(a.n, a.seed, progress=lambda i, n: print(f"  {i}/{n}", flush=True)), indent=2))
    return 0


def cmd_check_llm(a) -> int:
    try:
        from .llm import OllamaClient

        c = OllamaClient()
        ok = c.available()
    except Exception as e:
        print(f"ollama python package or server problem: {type(e).__name__}: {e}")
        return 1
    print(f"host={settings.ollama_host} model={c.model} available={ok}")
    if not ok:
        print(f"Start Ollama, then:  ollama pull {c.model}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lighthouse", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help=f"SQLite file (default {settings.db_path})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(parents=[common], name="serve", help="run the API and web page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser(parents=[common], name="init", help="create the database and generated pilot data")
    s.add_argument("--force", action="store_true", help="regenerate sales/cost/inventory data")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser(parents=[common], name="generate", help="bulk-create simulated cases (OBJ-05)")
    s.add_argument("--n", type=int, default=5000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser(parents=[common], name="verify-audit", help="recompute the audit hash chain")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser(parents=[common], name="ingest-csv", help="load a weekly-sales CSV into the pilot data")
    s.add_argument("csv")
    s.add_argument("--store-col", required=True)
    s.add_argument("--date-col", required=True)
    s.add_argument("--sales-col", required=True)
    s.add_argument("--promo-col", help="optional 0/1 column used as the 'promotion' marker (e.g. a holiday flag - say so in the report)")
    s.add_argument("--category", default="Grocery", help="category label to store these rows under (must match a cost_assumptions category)")
    s.add_argument("--replace", action="store_true", help="delete the generated sales rows first")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser(parents=[common], name="eval-llm", help="compare the local model with the rules baseline on N random cases")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=cmd_eval_llm)

    s = sub.add_parser(parents=[common], name="check-llm", help="is Ollama reachable and is the model pulled?")
    s.set_defaults(fn=cmd_check_llm)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
