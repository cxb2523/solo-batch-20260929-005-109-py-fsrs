"""
tools.trace_report
------------------

Generates trace_report.html: an observable trace of how a Rating flows through
Scheduler.review_card() / Scheduler.next_states() into Card and ReviewLog objects.

Run from the repository root:

    python tools/trace_report.py

For each of the four ratings (Again/Hard/Good/Easy) the report lists every state
field value and the source function that produced it.
"""

import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsrs import Card, Rating, Scheduler, State
from fsrs.scheduler import Trace, TraceRecord

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "trace_report.html"

REVIEW_DATETIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

FIELD_LABELS = {
    "card": "Card (input)",
    "days_since_last_review": "days_since_last_review",
    "retrievability": "retrievability",
    "stability": "stability",
    "difficulty": "difficulty",
    "state": "state",
    "step": "step",
    "next_interval": "next_interval",
    "due": "due",
    "last_review": "last_review",
    "review_log": "ReviewLog",
}


def _render_records_table(records: list[TraceRecord]) -> str:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{escape(FIELD_LABELS.get(record.field, record.field))}</td>"
            f"<td><code>{escape(record.source)}</code></td>"
            f"<td>{escape(str(record.before))}</td>"
            f"<td>{escape(str(record.after))}</td>"
            f"<td>{escape(record.note)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Field</th><th>Source function</th><th>Before</th><th>After</th><th>Note</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_card_summary(next_states) -> str:
    rows = []
    for rating, card in next_states.items():
        rows.append(
            "<tr>"
            f"<td>{escape(rating.name)}</td>"
            f"<td>{escape(card.state.name)}</td>"
            f"<td>{escape(str(card.step))}</td>"
            f"<td>{card.stability:.6f}</td>"
            f"<td>{card.difficulty:.6f}</td>"
            f"<td>{escape(card.due.isoformat())}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Rating</th><th>state</th><th>step</th><th>stability</th>"
        "<th>difficulty</th><th>due</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def main() -> None:
    scheduler = Scheduler(enable_fuzzing=False)

    # Scenario 1: a single review_card() call, traced end to end
    review_trace = Trace()
    new_card = Card(card_id=1, due=REVIEW_DATETIME)
    reviewed_card, review_log = scheduler.review_card(
        card=new_card,
        rating=Rating.Good,
        review_datetime=REVIEW_DATETIME,
        review_duration=4200,
        trace=review_trace,
    )

    # Scenario 2: next_states() on a Review-state card, traced per rating
    review_state_card = Card(
        card_id=2,
        state=State.Review,
        step=None,
        stability=10.0,
        difficulty=5.0,
        due=REVIEW_DATETIME,
        last_review=REVIEW_DATETIME - timedelta(days=10),
    )
    preview_trace = Trace()
    next_states = scheduler.next_states(
        card=review_state_card,
        review_datetime=REVIEW_DATETIME,
        trace=preview_trace,
    )

    sections = [
        "<h1>FSRS trace report</h1>",
        (
            f"<p>Generated for review_datetime={escape(REVIEW_DATETIME.isoformat())}, "
            f"scheduler defaults, fuzzing disabled (deterministic preview).</p>"
        ),
        "<h2>Scenario 1: review_card(new card, Good)</h2>",
        (
            "<p>Rating flows through update_state() into the returned Card copy and the "
            "ReviewLog. The input Card is never mutated in place.</p>"
        ),
        _render_records_table(review_trace.records),
        (
            f"<p>Result: state={escape(reviewed_card.state.name)}, "
            f"stability={reviewed_card.stability:.6f}, "
            f"difficulty={reviewed_card.difficulty:.6f}, "
            f"due={escape(reviewed_card.due.isoformat())}; "
            f"ReviewLog={escape(str(review_log.to_dict()))}</p>"
        ),
        "<h2>Scenario 2: next_states(Review-state card)</h2>",
        (
            "<p>Each rating is recomputed from a fresh copy of the input card, so the "
            "four next Cards cannot pollute each other. Summary of the four next Cards:</p>"
        ),
        _render_card_summary(next_states),
    ]

    for rating in Rating:
        rating_records = [
            record
            for record in preview_trace.records
            if record.context == f"next_states:{rating.name}"
        ]
        sections.append(f"<h3>Rating: {escape(rating.name)}</h3>")
        sections.append(_render_records_table(rating_records))

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>FSRS trace report</title>"
        "<style>"
        "body{font-family:sans-serif;margin:2em;}"
        "table{border-collapse:collapse;margin-bottom:1.5em;}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
        "th{background:#f0f0f0;}"
        "</style></head><body>" + "".join(sections) + "</body></html>"
    )

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
