"""Browser walkthrough of the whole demo, scripted with Playwright (not part of the pytest run: needs a running server).

    lighthouse serve --port 8765 &
    python tests/e2e_browser.py http://127.0.0.1:8765 [screenshot_dir]

It clicks through the Decision Case #1 scenario exactly as a presenter would and asserts what is on screen.
"""
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if SHOTS:
    SHOTS.mkdir(parents=True, exist_ok=True)


def shot(page, name):
    if SHOTS:
        page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)


def pick_role(page, value):
    page.select_option("#role", value)
    page.wait_for_timeout(300)


with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome") if Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome").exists() else pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    page.goto(BASE)
    expect(page.locator("h1")).to_contain_text("Lighthouse")
    expect(page.locator(".scn")).to_have_count(11)
    shot(page, "01_home")

    # 1. run the Decision Case #1 scenario as Decision Owner
    page.locator(".scn", has_text="Decision Case #1").get_by_role("button", name="Run").click()
    expect(page.locator(".banner.legal")).to_contain_text("Approval locked", timeout=10000)
    expect(page.locator(".chip.big")).to_have_text("PROCEED WITH CONDITIONS")
    expect(page.locator(".agent")).to_have_count(4)
    expect(page.get_by_role("button", name="Approve (locked)")).to_be_disabled()
    expect(page.locator("#view")).to_contain_text("Why a written reason is required")
    expect(page.locator("#view")).to_contain_text("CONDITION TENSION")
    assert "null" not in page.inner_text("#view"), "a literal 'null' leaked into the page"
    shot(page, "02_case1_locked")

    # 2. evidence drill-down
    page.locator(".evid").first.click()
    expect(page.locator("#dlg")).to_be_visible()
    expect(page.locator("#dlg-body")).to_contain_text("data as of")
    shot(page, "03_evidence")
    page.click("#dlg-close")

    # 3. the owner cannot act as Legal
    expect(page.locator("#view")).to_contain_text("Waiting for the Legal Reviewer")

    # 4. Legal reviewer clears with a comment
    pick_role(page, "legal_reviewer")
    page.locator("#view textarea").fill("Remove disease-prevention wording; 'supports immune health' is acceptable.")
    page.get_by_role("button", name="Record disposition").click()
    expect(page.locator(".banner.ok")).to_contain_text("Legal cleared", timeout=10000)

    # 5. owner: approve without reason -> server says no; with reason -> approved
    pick_role(page, "decision_owner")
    page.get_by_role("button", name="Approve", exact=True).click()
    expect(page.locator(".toast.err", has_text="REASON_REQUIRED")).to_be_visible()
    page.locator("#view textarea").fill("Approved with phased shipment and the $315K ceiling.")
    page.get_by_role("button", name="Approve", exact=True).click()
    expect(page.locator("#view > .card").first).to_contain_text("APPROVED", timeout=10000)
    expect(page.locator("#view")).to_contain_text("DECISION_RECORDED")
    shot(page, "04_approved")

    # 6. audit page + verify
    page.click("nav a[data-h='#/audit']")
    page.get_by_role("button", name="Verify chain").click()
    expect(page.locator("#verify")).to_contain_text("Chain intact")
    shot(page, "05_audit")

    # 7. admin page
    pick_role(page, "admin")
    page.click("nav a[data-h='#/admin']")
    expect(page.locator("#view")).to_contain_text("Thresholds")
    expect(page.locator("#view")).to_contain_text("Agent registry")
    shot(page, "06_admin")

    # 8. revision loop (real rules agents, live SSE progress)
    pick_role(page, "decision_owner")
    page.click("nav a[data-h='#/']")
    page.locator(".scn", has_text="Revision loop").get_by_role("button", name="Run").click()
    expect(page.locator(".chip.big")).to_have_text("HOLD", timeout=10000)
    page.locator("#view textarea").fill("Reduce the discount, shorten the promotion and raise the ceiling.")
    page.get_by_role("button", name="Request revision").click()
    for name, val in (("discount_pct", "1"), ("weeks", "4"), ("budget_usd", "5000000")):
        page.locator(f"input[name={name}]").fill(val)
    page.get_by_role("button", name="Send revision request").click()
    expect(page.locator("#view h2").first).to_contain_text(".r1", timeout=10000)
    expect(page.locator(".chip.big")).not_to_have_text("HOLD", timeout=10000)
    shot(page, "07_revision")

    # 9. an injected title must render as text, not run
    page.click("#tab-new")
    page.fill("input[name=title]", "<img src=x onerror=window.__xss=1> pwn")
    page.fill("input[name=category]", "Grocery")
    page.fill("textarea[name=claims]", "")
    page.get_by_role("button", name="Open case").click()
    expect(page.locator("#view h2").first).to_contain_text("<img src=x", timeout=10000)
    assert page.evaluate("window.__xss") is None, "XSS: injected markup executed"

    # 10. wrong role gets a readable error
    pick_role(page, "legal_reviewer")
    page.click("#tab-scn")
    page.locator(".scn").first.get_by_role("button", name="Run").click()
    expect(page.locator(".toast.err", has_text="may not do this")).to_be_visible()

    # 11. mobile width still usable
    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(BASE + "/#/case/DC-0001")
    page.wait_for_timeout(600)
    shot(page, "08_mobile")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), "horizontal scroll on mobile"

    # 403/409/422 responses are provoked on purpose (role / lock / reason checks) and Chromium logs them as console errors
    bad = [e for e in errors if not (e.startswith("console.error: Failed to load resource") and any(c in e for c in ("403", "409", "422")))]
    print("console/page errors:", bad or "none")
    browser.close()
    sys.exit(1 if bad else 0)
