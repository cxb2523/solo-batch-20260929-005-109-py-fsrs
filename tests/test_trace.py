"""
Tests for the observable trace pipeline: Scheduler.review_card, Scheduler.next_states,
Scheduler.update_state and the Trace/TraceRecord/NextStates helpers.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from fsrs.card import Card
from fsrs.rating import Rating
from fsrs.scheduler import Scheduler, Trace
from fsrs.state import State

REVIEW_DATETIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _review_state_card() -> Card:
    return Card(
        card_id=42,
        state=State.Review,
        step=None,
        stability=10.0,
        difficulty=5.0,
        due=REVIEW_DATETIME,
        last_review=REVIEW_DATETIME - timedelta(days=10),
    )


class TestNextStates:
    def test_ratings_do_not_pollute_each_other(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()

        next_states = scheduler.next_states(card=card, review_datetime=REVIEW_DATETIME)

        # each rating's next Card equals an independent review_card() of the same input
        for rating, next_card in next_states.items():
            reviewed_card, _ = scheduler.review_card(
                card=card, rating=rating, review_datetime=REVIEW_DATETIME
            )
            assert next_card.to_dict() == reviewed_card.to_dict()

        # the four next Cards are four independent objects
        next_cards = [card for _, card in next_states.items()]
        assert len({id(card) for card in next_cards}) == 4

        # the four ratings diverge where the algorithm says they must
        assert next_states.again.state == State.Relearning
        assert next_states.again.step == 0
        assert next_states.hard.state == State.Review
        assert next_states.good.state == State.Review
        assert next_states.easy.state == State.Review
        assert (
            next_states.again.stability
            < next_states.hard.stability
            < next_states.good.stability
            < next_states.easy.stability
        )
        assert next_states.good.due < next_states.easy.due

        # mutating one next Card leaves the others untouched
        snapshot = [card.to_dict() for card in next_cards]
        next_cards[0].stability = 999.0
        next_cards[0].state = State.Relearning
        for mutated, original in zip(next_cards[1:], snapshot[1:]):
            assert mutated.to_dict() == original

    def test_input_card_is_not_mutated(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()
        before = card.to_dict()

        _ = scheduler.next_states(card=card, review_datetime=REVIEW_DATETIME)

        assert card.to_dict() == before

    def test_new_card_learning_steps(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = Card(due=REVIEW_DATETIME)

        next_states = scheduler.next_states(card=card, review_datetime=REVIEW_DATETIME)

        assert next_states.again.state == State.Learning
        assert next_states.again.step == 0
        assert next_states.hard.state == State.Learning
        assert next_states.good.state == State.Learning
        assert next_states.good.step == 1
        assert next_states.easy.state == State.Review
        assert next_states.easy.step is None

    def test_trace_records_per_rating_context(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()
        trace = Trace()

        _ = scheduler.next_states(
            card=card, review_datetime=REVIEW_DATETIME, trace=trace
        )

        contexts = {record.context for record in trace.records}
        for rating in Rating:
            assert f"next_states:{rating.name}" in contexts

        sources = {record.source for record in trace.records}
        assert "update_state" in sources
        assert "_next_stability" in sources
        assert "_next_difficulty" in sources
        assert "get_card_retrievability" in sources


class TestReviewCardTrace:
    def test_review_log_fields_complete(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = Card(card_id=7, due=REVIEW_DATETIME)
        review_duration = 5000

        reviewed_card, review_log = scheduler.review_card(
            card=card,
            rating=Rating.Good,
            review_datetime=REVIEW_DATETIME,
            review_duration=review_duration,
        )

        # every public ReviewLog field is populated from the review inputs
        assert review_log.card_id == card.card_id
        assert review_log.rating == Rating.Good
        assert review_log.review_datetime == REVIEW_DATETIME
        assert review_log.review_duration == review_duration
        assert set(review_log.to_dict()) == {
            "card_id",
            "rating",
            "review_datetime",
            "review_duration",
        }

        # the returned Card carries the review results
        assert reviewed_card.last_review == REVIEW_DATETIME
        assert reviewed_card.due > REVIEW_DATETIME
        assert reviewed_card.stability is not None
        assert reviewed_card.difficulty is not None

    def test_input_card_is_not_mutated(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()
        before = deepcopy(card)

        _ = scheduler.review_card(
            card=card, rating=Rating.Again, review_datetime=REVIEW_DATETIME
        )

        assert card.to_dict() == before.to_dict()

    def test_trace_records_full_pipeline(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = Card(card_id=7, due=REVIEW_DATETIME)
        trace = Trace()

        reviewed_card, review_log = scheduler.review_card(
            card=card,
            rating=Rating.Easy,
            review_datetime=REVIEW_DATETIME,
            trace=trace,
        )

        assert len(trace.records) > 0
        assert all(record.context == "review_card" for record in trace.records)

        sources = {record.source for record in trace.records}
        # memory-state update
        assert "_initial_stability" in sources
        assert "_initial_difficulty" in sources
        # state machine + interval
        assert "update_state" in sources
        assert "_next_interval" in sources
        # Card write-back and ReviewLog creation
        assert "review_card" in sources

        fields = {record.field for record in trace.records}
        assert {
            "card",
            "days_since_last_review",
            "stability",
            "difficulty",
            "state",
            "next_interval",
            "due",
            "last_review",
            "review_log",
        } <= fields

        # the traced write-back matches the returned objects
        due_records = [r for r in trace.records if r.field == "due"]
        assert due_records[-1].after == reviewed_card.due
        log_records = [r for r in trace.records if r.field == "review_log"]
        assert log_records[-1].after == review_log.to_dict()

    def test_trace_records_relearning_branch(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()
        trace = Trace()

        reviewed_card, _ = scheduler.review_card(
            card=card, rating=Rating.Again, review_datetime=REVIEW_DATETIME, trace=trace
        )

        assert reviewed_card.state == State.Relearning
        state_records = [r for r in trace.records if r.field == "state"]
        assert any(r.after == State.Relearning for r in state_records)

    def test_no_trace_by_default(self):
        scheduler = Scheduler(enable_fuzzing=False)
        card = _review_state_card()

        # review_card/next_states work without a trace attached
        _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=REVIEW_DATETIME
        )
        _ = scheduler.next_states(card=card, review_datetime=REVIEW_DATETIME)
