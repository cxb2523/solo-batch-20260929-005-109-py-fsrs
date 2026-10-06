"""
fsrs.scheduler
---------

This module defines the Scheduler class as well as the various constants used in its calculations.

Classes:
    Scheduler: The FSRS spaced-repetition scheduler.
    Trace: Collects TraceRecord entries describing each step of a review/preview.
    TraceRecord: One observed step in the Rating -> Card/ReviewLog pipeline.
    NextStates: The four possible next Cards (Again/Hard/Good/Easy) of a review.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from random import random
from typing import TYPE_CHECKING, TypedDict, overload

from fsrs.card import Card
from fsrs.rating import Rating
from fsrs.review_log import ReviewLog
from fsrs.state import State

if TYPE_CHECKING:
    from torch import Tensor  # torch is optional; import only for type checking
from typing_extensions import Self

FSRS_DEFAULT_DECAY = 0.1542
DEFAULT_PARAMETERS = (
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    FSRS_DEFAULT_DECAY,
)

STABILITY_MIN = 0.001
LOWER_BOUNDS_PARAMETERS = (
    STABILITY_MIN,
    STABILITY_MIN,
    STABILITY_MIN,
    STABILITY_MIN,
    1.0,
    0.001,
    0.001,
    0.001,
    0.0,
    0.0,
    0.001,
    0.001,
    0.001,
    0.001,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.1,
)

INITIAL_STABILITY_MAX = 100.0
UPPER_BOUNDS_PARAMETERS = (
    INITIAL_STABILITY_MAX,
    INITIAL_STABILITY_MAX,
    INITIAL_STABILITY_MAX,
    INITIAL_STABILITY_MAX,
    10.0,
    4.0,
    4.0,
    0.75,
    4.5,
    0.8,
    3.5,
    5.0,
    0.25,
    0.9,
    4.0,
    1.0,
    6.0,
    2.0,
    2.0,
    0.8,
    0.8,
)

MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0

FUZZ_RANGES = [
    {
        "start": 2.5,
        "end": 7.0,
        "factor": 0.15,
    },
    {
        "start": 7.0,
        "end": 20.0,
        "factor": 0.1,
    },
    {
        "start": 20.0,
        "end": math.inf,
        "factor": 0.05,
    },
]


@dataclass(frozen=True)
class TraceRecord:
    """
    One observed step in the Rating -> Card/ReviewLog pipeline.

    Attributes:
        context: Which entry point produced the record, e.g. "review_card" or "next_states:Good".
        source: The name of the function that produced the value, e.g. "_next_stability".
        field: The Card field or intermediate quantity the record describes.
        before: The value before the step (None if not applicable).
        after: The value after the step.
        note: Extra detail, e.g. which clamp bound or learning-step branch applied.
    """

    context: str
    source: str
    field: str
    before: object
    after: object
    note: str


class Trace:
    """
    Observability sink for Scheduler.review_card() and Scheduler.next_states().

    Pass an instance via the `trace` argument and every step of the computation
    (state machine branch, memory-state update, clamp boundary, interval choice,
    Card write-back, ReviewLog creation) is appended as a TraceRecord.
    """

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self.context: str = ""

    def record(
        self,
        *,
        source: str,
        field: str,
        before: object,
        after: object,
        note: str = "",
    ) -> None:
        self.records.append(
            TraceRecord(
                context=self.context,
                source=source,
                field=field,
                before=before,
                after=after,
                note=note,
            )
        )


def _record(
    trace: Trace | None,
    *,
    source: str,
    field: str,
    before: object,
    after: object,
    note: str = "",
) -> None:
    """
    Appends a TraceRecord when a Trace is attached; a no-op otherwise.
    """

    if trace is not None:
        trace.record(source=source, field=field, before=before, after=after, note=note)


@dataclass(frozen=True)
class NextStates:
    """
    The four possible next Cards of a review, one per Rating.

    Attributes:
        again: The Card as it would look after an Again rating.
        hard: The Card as it would look after a Hard rating.
        good: The Card as it would look after a Good rating.
        easy: The Card as it would look after an Easy rating.
    """

    again: Card
    hard: Card
    good: Card
    easy: Card

    def items(self) -> tuple[tuple[Rating, Card], ...]:
        """
        Returns the (Rating, next Card) pairs in Again/Hard/Good/Easy order.
        """

        return (
            (Rating.Again, self.again),
            (Rating.Hard, self.hard),
            (Rating.Good, self.good),
            (Rating.Easy, self.easy),
        )


class SchedulerDict(TypedDict):
    """
    JSON-serializable dictionary representation of a Scheduler object.
    """

    parameters: list[float]
    desired_retention: float
    learning_steps: list[int]
    relearning_steps: list[int]
    maximum_interval: int
    enable_fuzzing: bool


@dataclass(init=False)
class Scheduler:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        parameters: The model weights of the FSRS scheduler.
        desired_retention: The desired retention rate of cards scheduled with the scheduler.
        learning_steps: Small time intervals that schedule cards in the Learning state.
        relearning_steps: Small time intervals that schedule cards in the Relearning state.
        maximum_interval: The maximum number of days a Review-state card can be scheduled into the future.
        enable_fuzzing: Whether to apply a small amount of random 'fuzz' to calculated intervals.
    """

    parameters: tuple[float, ...]
    desired_retention: float
    learning_steps: tuple[timedelta, ...]
    relearning_steps: tuple[timedelta, ...]
    maximum_interval: int
    enable_fuzzing: bool

    def __init__(
        self,
        parameters: Sequence[float] = DEFAULT_PARAMETERS,
        desired_retention: float = 0.9,
        learning_steps: tuple[timedelta, ...] | list[timedelta] = (
            timedelta(minutes=1),
            timedelta(minutes=10),
        ),
        relearning_steps: tuple[timedelta, ...] | list[timedelta] = (
            timedelta(minutes=10),
        ),
        maximum_interval: int = 36500,
        enable_fuzzing: bool = True,
    ) -> None:
        self._validate_parameters(parameters=parameters)

        self.parameters = tuple(parameters)
        self.desired_retention = desired_retention
        self.learning_steps = tuple(learning_steps)
        self.relearning_steps = tuple(relearning_steps)
        self.maximum_interval = maximum_interval
        self.enable_fuzzing = enable_fuzzing

        self._DECAY = -self.parameters[20]
        self._FACTOR = 0.9 ** (1 / self._DECAY) - 1

    def _validate_parameters(self, *, parameters: Sequence[float]) -> None:
        if len(parameters) != len(LOWER_BOUNDS_PARAMETERS):
            raise ValueError(
                f"Expected {len(LOWER_BOUNDS_PARAMETERS)} parameters, got {len(parameters)}."
            )

        error_messages = []
        for index, (parameter, lower_bound, upper_bound) in enumerate(
            zip(parameters, LOWER_BOUNDS_PARAMETERS, UPPER_BOUNDS_PARAMETERS)
        ):
            if not lower_bound <= parameter <= upper_bound:
                error_message = f"parameters[{index}] = {parameter} is out of bounds: ({lower_bound}, {upper_bound})"
                error_messages.append(error_message)

        if len(error_messages) > 0:
            raise ValueError(
                "One or more parameters are out of bounds:\n"
                + "\n".join(error_messages)
            )

    def get_card_retrievability(
        self, card: Card, current_datetime: datetime | None = None
    ) -> float:
        """
        Calculates a Card object's current retrievability for a given date and time.

        The retrievability of a card is the predicted probability that the card is correctly recalled at the provided datetime.

        Args:
            card: The card whose retrievability is to be calculated
            current_datetime: The current date and time

        Returns:
            float: The retrievability of the Card object.
        """

        if card.last_review is None or card.stability is None:
            return 0

        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc)

        elapsed_days = max(0, (current_datetime - card.last_review).days)

        return (1 + self._FACTOR * elapsed_days / card.stability) ** self._DECAY

    def review_card(
        self,
        card: Card,
        rating: Rating,
        review_datetime: datetime | None = None,
        review_duration: int | None = None,
        trace: Trace | None = None,
    ) -> tuple[Card, ReviewLog]:
        """
        Reviews a card with a given rating at a given time for a specified duration.

        Args:
            card: The card being reviewed. It is never mutated in place.
            rating: The chosen rating for the card being reviewed.
            review_datetime: The date and time of the review.
            review_duration: The number of miliseconds it took to review the card or None if unspecified.
            trace: An optional Trace that collects a TraceRecord for every step of the review.

        Returns:
            tuple[Card,ReviewLog]: A tuple containing the updated, reviewed card and its corresponding review log.

        Raises:
            ValueError: If the `review_datetime` argument is not timezone-aware and set to UTC.

        Design note (API surface): the returned ReviewLog intentionally exposes only the
        four public fields (card_id, rating, review_datetime, review_duration) that the
        Optimizer consumes. Every intermediate value of the scheduling computation is
        observable through the opt-in `trace` argument instead, so the scheduling API is
        not inflated with internal structure.
        """

        self._validate_review_datetime(review_datetime=review_datetime)

        if review_datetime is None:
            review_datetime = datetime.now(timezone.utc)

        previous_context = trace.context if trace is not None else ""
        if trace is not None:
            trace.context = "review_card"
        try:
            # the caller's Card is never mutated in place: update_state() writes to a private copy
            _record(
                trace,
                source="review_card",
                field="card",
                before=card.to_dict(),
                after="copied",
                note="input Card is left untouched; update_state() mutates a private copy",
            )
            card = copy(card)

            next_interval = self.update_state(
                card=card,
                rating=rating,
                review_datetime=review_datetime,
                trace=trace,
            )

            if self.enable_fuzzing and card.state == State.Review:
                unfuzzed_interval = next_interval
                next_interval = self._get_fuzzed_interval(interval=next_interval)
                _record(
                    trace,
                    source="_get_fuzzed_interval",
                    field="next_interval",
                    before=unfuzzed_interval,
                    after=next_interval,
                    note="random fuzz applied to Review-state intervals",
                )

            card.due = review_datetime + next_interval
            card.last_review = review_datetime
            _record(
                trace,
                source="review_card",
                field="due",
                before=None,
                after=card.due,
                note="due = review_datetime + next_interval",
            )
            _record(
                trace,
                source="review_card",
                field="last_review",
                before=None,
                after=card.last_review,
                note="write-back to the returned Card copy",
            )

            review_log = ReviewLog(
                card_id=card.card_id,
                rating=rating,
                review_datetime=review_datetime,
                review_duration=review_duration,
            )
            _record(
                trace,
                source="review_card",
                field="review_log",
                before=None,
                after=review_log.to_dict(),
                note="only the four public ReviewLog fields consumed by the Optimizer leave the scheduler",
            )
        finally:
            if trace is not None:
                trace.context = previous_context

        return card, review_log

    def update_state(
        self,
        *,
        card: Card,
        rating: Rating,
        review_datetime: datetime,
        trace: Trace | None = None,
    ) -> timedelta:
        """
        Advances a card one review forward and returns its next interval.

        This is the low-level state machine shared by review_card() and next_states().

        Args:
            card: The card to update. Mutated in place; pass a copy if the original must be preserved
                (review_card() and next_states() already copy for you).
            rating: The chosen rating for the card.
            review_datetime: The date and time of the review (timezone-aware, UTC).
            trace: An optional Trace that collects a TraceRecord for every step.

        Returns:
            timedelta: The interval until the card's next review.
        """

        days_since_last_review = (
            (review_datetime - card.last_review).days if card.last_review else None
        )
        _record(
            trace,
            source="update_state",
            field="days_since_last_review",
            before=card.last_review,
            after=days_since_last_review,
            note="timedelta.days floors to whole days; values < 1 select _short_term_stability, otherwise _next_stability",
        )

        match card.state:
            case State.Learning:
                assert card.step is not None

                # update the card's stability and difficulty
                if card.stability is None or card.difficulty is None:
                    new_stability = self._initial_stability(rating=rating)
                    _record(
                        trace,
                        source="_initial_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"first review of the card; lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                    new_difficulty = self._initial_difficulty(rating=rating, clamp=True)
                    _record(
                        trace,
                        source="_initial_difficulty",
                        field="difficulty",
                        before=card.difficulty,
                        after=new_difficulty,
                        note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                    )
                    card.difficulty = new_difficulty

                elif days_since_last_review is not None and days_since_last_review < 1:
                    new_stability = self._short_term_stability(
                        stability=card.stability, rating=rating
                    )
                    _record(
                        trace,
                        source="_short_term_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"same-day review (days_since_last_review < 1); lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                    new_difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )
                    _record(
                        trace,
                        source="_next_difficulty",
                        field="difficulty",
                        before=card.difficulty,
                        after=new_difficulty,
                        note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                    )
                    card.difficulty = new_difficulty

                else:
                    retrievability = self.get_card_retrievability(
                        card,
                        current_datetime=review_datetime,
                    )
                    _record(
                        trace,
                        source="get_card_retrievability",
                        field="retrievability",
                        before=None,
                        after=retrievability,
                        note="elapsed_days = max(0, (review_datetime - last_review).days)",
                    )
                    new_stability = self._next_stability(
                        difficulty=card.difficulty,
                        stability=card.stability,
                        retrievability=retrievability,
                        rating=rating,
                    )
                    _record(
                        trace,
                        source="_next_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                    new_difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )
                    _record(
                        trace,
                        source="_next_difficulty",
                        field="difficulty",
                        before=card.difficulty,
                        after=new_difficulty,
                        note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                    )
                    card.difficulty = new_difficulty

                # calculate the card's next interval
                ## first if-clause handles edge case where the Card in the Learning state was previously
                ## scheduled with a Scheduler with more learning_steps than the current Scheduler
                if len(self.learning_steps) == 0 or (
                    card.step >= len(self.learning_steps)
                    and rating in (Rating.Hard, Rating.Good, Rating.Easy)
                ):
                    _record(
                        trace,
                        source="update_state",
                        field="state",
                        before=card.state,
                        after=State.Review,
                        note="learning steps exhausted; card graduates to Review",
                    )
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)
                    _record(
                        trace,
                        source="_next_interval",
                        field="next_interval",
                        before=None,
                        after=next_interval,
                        note="rounded to whole days; clamped to [1, maximum_interval]",
                    )

                else:
                    match rating:
                        case Rating.Again:
                            card.step = 0
                            next_interval = self.learning_steps[card.step]
                            _record(
                                trace,
                                source="update_state",
                                field="step",
                                before=None,
                                after=card.step,
                                note="Again resets to the first learning step",
                            )
                            _record(
                                trace,
                                source="update_state",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="learning_steps[0]",
                            )

                        case Rating.Hard:
                            # card step stays the same

                            if card.step == 0 and len(self.learning_steps) == 1:
                                next_interval = self.learning_steps[0] * 1.5
                                hard_note = "learning_steps[0] * 1.5"
                            elif card.step == 0 and len(self.learning_steps) >= 2:
                                next_interval = (
                                    self.learning_steps[0] + self.learning_steps[1]
                                ) / 2.0
                                hard_note = (
                                    "average of learning_steps[0] and learning_steps[1]"
                                )
                            else:
                                next_interval = self.learning_steps[card.step]
                                hard_note = (
                                    f"learning_steps[{card.step}]; step unchanged"
                                )
                            _record(
                                trace,
                                source="update_state",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note=hard_note,
                            )

                        case Rating.Good:
                            if card.step + 1 == len(
                                self.learning_steps
                            ):  # the last step
                                _record(
                                    trace,
                                    source="update_state",
                                    field="state",
                                    before=card.state,
                                    after=State.Review,
                                    note="Good on the last learning step; card graduates to Review",
                                )
                                card.state = State.Review
                                card.step = None

                                next_interval_days = self._next_interval(
                                    stability=card.stability
                                )
                                next_interval = timedelta(days=next_interval_days)
                                _record(
                                    trace,
                                    source="_next_interval",
                                    field="next_interval",
                                    before=None,
                                    after=next_interval,
                                    note="rounded to whole days; clamped to [1, maximum_interval]",
                                )

                            else:
                                card.step += 1
                                next_interval = self.learning_steps[card.step]
                                _record(
                                    trace,
                                    source="update_state",
                                    field="step",
                                    before=card.step - 1,
                                    after=card.step,
                                    note="Good advances one learning step",
                                )
                                _record(
                                    trace,
                                    source="update_state",
                                    field="next_interval",
                                    before=None,
                                    after=next_interval,
                                    note=f"learning_steps[{card.step}]",
                                )

                        case Rating.Easy:
                            _record(
                                trace,
                                source="update_state",
                                field="state",
                                before=card.state,
                                after=State.Review,
                                note="Easy graduates the card to Review immediately",
                            )
                            card.state = State.Review
                            card.step = None

                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)
                            _record(
                                trace,
                                source="_next_interval",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="rounded to whole days; clamped to [1, maximum_interval]",
                            )

                        case _:
                            raise ValueError(f"Unknown rating: {rating}")

            case State.Review:
                assert card.stability is not None
                assert card.difficulty is not None

                # update the card's stability and difficulty
                if days_since_last_review is not None and days_since_last_review < 1:
                    new_stability = self._short_term_stability(
                        stability=card.stability, rating=rating
                    )
                    _record(
                        trace,
                        source="_short_term_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"same-day review (days_since_last_review < 1); lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                else:
                    retrievability = self.get_card_retrievability(
                        card,
                        current_datetime=review_datetime,
                    )
                    _record(
                        trace,
                        source="get_card_retrievability",
                        field="retrievability",
                        before=None,
                        after=retrievability,
                        note="elapsed_days = max(0, (review_datetime - last_review).days)",
                    )
                    new_stability = self._next_stability(
                        difficulty=card.difficulty,
                        stability=card.stability,
                        retrievability=retrievability,
                        rating=rating,
                    )
                    _record(
                        trace,
                        source="_next_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability

                new_difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )
                _record(
                    trace,
                    source="_next_difficulty",
                    field="difficulty",
                    before=card.difficulty,
                    after=new_difficulty,
                    note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                )
                card.difficulty = new_difficulty

                # calculate the card's next interval
                match rating:
                    case Rating.Again:
                        # if there are no relearning steps (they were left blank)
                        if len(self.relearning_steps) == 0:
                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)
                            _record(
                                trace,
                                source="_next_interval",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="no relearning steps configured; rounded to whole days, clamped to [1, maximum_interval]",
                            )

                        else:
                            _record(
                                trace,
                                source="update_state",
                                field="state",
                                before=card.state,
                                after=State.Relearning,
                                note="Again lapses the card into Relearning at step 0",
                            )
                            card.state = State.Relearning
                            card.step = 0

                            next_interval = self.relearning_steps[card.step]
                            _record(
                                trace,
                                source="update_state",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="relearning_steps[0]",
                            )

                    case Rating.Hard | Rating.Good | Rating.Easy:
                        next_interval_days = self._next_interval(
                            stability=card.stability
                        )
                        next_interval = timedelta(days=next_interval_days)
                        _record(
                            trace,
                            source="_next_interval",
                            field="next_interval",
                            before=None,
                            after=next_interval,
                            note="rounded to whole days; clamped to [1, maximum_interval]",
                        )

                    case _:
                        raise ValueError(f"Unknown rating: {rating}")

            case State.Relearning:
                assert card.stability is not None
                assert card.difficulty is not None
                assert card.step is not None

                # update the card's stability and difficulty
                if days_since_last_review is not None and days_since_last_review < 1:
                    new_stability = self._short_term_stability(
                        stability=card.stability, rating=rating
                    )
                    _record(
                        trace,
                        source="_short_term_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"same-day review (days_since_last_review < 1); lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                    new_difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )
                    _record(
                        trace,
                        source="_next_difficulty",
                        field="difficulty",
                        before=card.difficulty,
                        after=new_difficulty,
                        note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                    )
                    card.difficulty = new_difficulty

                else:
                    retrievability = self.get_card_retrievability(
                        card,
                        current_datetime=review_datetime,
                    )
                    _record(
                        trace,
                        source="get_card_retrievability",
                        field="retrievability",
                        before=None,
                        after=retrievability,
                        note="elapsed_days = max(0, (review_datetime - last_review).days)",
                    )
                    new_stability = self._next_stability(
                        difficulty=card.difficulty,
                        stability=card.stability,
                        retrievability=retrievability,
                        rating=rating,
                    )
                    _record(
                        trace,
                        source="_next_stability",
                        field="stability",
                        before=card.stability,
                        after=new_stability,
                        note=f"lower-bounded at STABILITY_MIN={STABILITY_MIN}",
                    )
                    card.stability = new_stability
                    new_difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )
                    _record(
                        trace,
                        source="_next_difficulty",
                        field="difficulty",
                        before=card.difficulty,
                        after=new_difficulty,
                        note=f"clamped to [{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]",
                    )
                    card.difficulty = new_difficulty

                # calculate the card's next interval
                ## first if-clause handles edge case where the Card in the Relearning state was previously
                ## scheduled with a Scheduler with more relearning_steps than the current Scheduler
                if len(self.relearning_steps) == 0 or (
                    card.step >= len(self.relearning_steps)
                    and rating in (Rating.Hard, Rating.Good, Rating.Easy)
                ):
                    _record(
                        trace,
                        source="update_state",
                        field="state",
                        before=card.state,
                        after=State.Review,
                        note="relearning steps exhausted; card graduates to Review",
                    )
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)
                    _record(
                        trace,
                        source="_next_interval",
                        field="next_interval",
                        before=None,
                        after=next_interval,
                        note="rounded to whole days; clamped to [1, maximum_interval]",
                    )

                else:
                    match rating:
                        case Rating.Again:
                            card.step = 0
                            next_interval = self.relearning_steps[card.step]
                            _record(
                                trace,
                                source="update_state",
                                field="step",
                                before=None,
                                after=card.step,
                                note="Again resets to the first relearning step",
                            )
                            _record(
                                trace,
                                source="update_state",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="relearning_steps[0]",
                            )

                        case Rating.Hard:
                            # card step stays the same

                            if card.step == 0 and len(self.relearning_steps) == 1:
                                next_interval = self.relearning_steps[0] * 1.5
                                hard_note = "relearning_steps[0] * 1.5"
                            elif card.step == 0 and len(self.relearning_steps) >= 2:
                                next_interval = (
                                    self.relearning_steps[0] + self.relearning_steps[1]
                                ) / 2.0
                                hard_note = "average of relearning_steps[0] and relearning_steps[1]"
                            else:
                                next_interval = self.relearning_steps[card.step]
                                hard_note = (
                                    f"relearning_steps[{card.step}]; step unchanged"
                                )
                            _record(
                                trace,
                                source="update_state",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note=hard_note,
                            )

                        case Rating.Good:
                            if card.step + 1 == len(
                                self.relearning_steps
                            ):  # the last step
                                _record(
                                    trace,
                                    source="update_state",
                                    field="state",
                                    before=card.state,
                                    after=State.Review,
                                    note="Good on the last relearning step; card graduates to Review",
                                )
                                card.state = State.Review
                                card.step = None

                                next_interval_days = self._next_interval(
                                    stability=card.stability
                                )
                                next_interval = timedelta(days=next_interval_days)
                                _record(
                                    trace,
                                    source="_next_interval",
                                    field="next_interval",
                                    before=None,
                                    after=next_interval,
                                    note="rounded to whole days; clamped to [1, maximum_interval]",
                                )

                            else:
                                card.step += 1
                                next_interval = self.relearning_steps[card.step]
                                _record(
                                    trace,
                                    source="update_state",
                                    field="step",
                                    before=card.step - 1,
                                    after=card.step,
                                    note="Good advances one relearning step",
                                )
                                _record(
                                    trace,
                                    source="update_state",
                                    field="next_interval",
                                    before=None,
                                    after=next_interval,
                                    note=f"relearning_steps[{card.step}]",
                                )

                        case Rating.Easy:
                            _record(
                                trace,
                                source="update_state",
                                field="state",
                                before=card.state,
                                after=State.Review,
                                note="Easy graduates the card to Review immediately",
                            )
                            card.state = State.Review
                            card.step = None

                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)
                            _record(
                                trace,
                                source="_next_interval",
                                field="next_interval",
                                before=None,
                                after=next_interval,
                                note="rounded to whole days; clamped to [1, maximum_interval]",
                            )

                        case _:
                            raise ValueError(f"Unknown rating: {rating}")

            case _:
                raise ValueError(f"Unknown card state: {card.state}")

        return next_interval

    def next_states(
        self,
        card: Card,
        review_datetime: datetime | None = None,
        trace: Trace | None = None,
    ) -> NextStates:
        """
        Computes, in one call, the next Card for each of the four ratings.

        Args:
            card: The card to preview. It is never mutated in place.
            review_datetime: The date and time of the hypothetical review.
            trace: An optional Trace that collects a TraceRecord per step and rating.

        Returns:
            NextStates: The four possible next Cards (Again/Hard/Good/Easy).

        Raises:
            ValueError: If the `review_datetime` argument is not timezone-aware and set to UTC.

        Design note (shared vs. recomputed memory state): each rating is recomputed
        from a fresh copy of the input card instead of reusing one shared,
        already-updated memory state. Reusing a single copy would be cheaper, but
        update_state() mutates its card in place, so the four ratings would
        contaminate each other (and the caller's card). The 4x recompute cost buys
        guaranteed isolation between the ratings. Fuzzing is skipped so the preview
        is deterministic.
        """

        self._validate_review_datetime(review_datetime=review_datetime)

        if review_datetime is None:
            review_datetime = datetime.now(timezone.utc)

        next_cards: dict[Rating, Card] = {}
        previous_context = trace.context if trace is not None else ""
        try:
            for rating in Rating:
                # fresh copy per rating: update_state() mutates in place, so a shared
                # copy would let earlier ratings pollute later ones
                rating_card = copy(card)

                if trace is not None:
                    trace.context = f"next_states:{rating.name}"

                next_interval = self.update_state(
                    card=rating_card,
                    rating=rating,
                    review_datetime=review_datetime,
                    trace=trace,
                )

                rating_card.due = review_datetime + next_interval
                rating_card.last_review = review_datetime
                _record(
                    trace,
                    source="next_states",
                    field="due",
                    before=None,
                    after=rating_card.due,
                    note="preview only: no fuzzing applied, no ReviewLog created",
                )

                next_cards[rating] = rating_card
        finally:
            if trace is not None:
                trace.context = previous_context

        return NextStates(
            again=next_cards[Rating.Again],
            hard=next_cards[Rating.Hard],
            good=next_cards[Rating.Good],
            easy=next_cards[Rating.Easy],
        )

    @staticmethod
    def _validate_review_datetime(review_datetime: datetime | None) -> None:
        if review_datetime is not None and (
            (review_datetime.tzinfo is None) or (review_datetime.tzinfo != timezone.utc)
        ):
            raise ValueError("datetime must be timezone-aware and set to UTC")

    def reschedule_card(self, card: Card, review_logs: list[ReviewLog]) -> Card:
        """
        Reschedules/updates the given card with the current scheduler provided that card's review logs.

        If the current card was previously scheduled with a different scheduler, you may want to reschedule/update
        it as if it had always been scheduled with this current scheduler. For example, you may want to reschedule
        each of your cards with a new scheduler after computing the optimal parameters with the Optimizer.

        Args:
            card: The card to be rescheduled/updated.
            review_logs: A list of that card's review logs (order doesn't matter).

        Returns:
            Card: A new card that has been rescheduled/updated with this current scheduler.

        Raises:
            ValueError: If any of the review logs are for a card other than the one specified, this will raise an error.

        """

        for review_log in review_logs:
            if review_log.card_id != card.card_id:
                raise ValueError(
                    f"ReviewLog card_id {review_log.card_id} does not match Card card_id {card.card_id}"
                )

        review_logs = sorted(review_logs, key=lambda log: log.review_datetime)

        rescheduled_card = Card(card_id=card.card_id, due=card.due)

        for review_log in review_logs:
            rescheduled_card, _ = self.review_card(
                card=rescheduled_card,
                rating=review_log.rating,
                review_datetime=review_log.review_datetime,
            )

        return rescheduled_card

    def to_dict(
        self,
    ) -> SchedulerDict:
        """
        Returns a dictionary representation of the Scheduler object.

        Returns:
            SchedulerDict: A dictionary representation of the Scheduler object.
        """

        return {
            "parameters": list(self.parameters),
            "desired_retention": self.desired_retention,
            "learning_steps": [
                int(learning_step.total_seconds())
                for learning_step in self.learning_steps
            ],
            "relearning_steps": [
                int(relearning_step.total_seconds())
                for relearning_step in self.relearning_steps
            ],
            "maximum_interval": self.maximum_interval,
            "enable_fuzzing": self.enable_fuzzing,
        }

    @classmethod
    def from_dict(cls, source_dict: SchedulerDict) -> Self:
        """
        Creates a Scheduler object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing Scheduler object.

        Returns:
            Self: A Scheduler object created from the provided dictionary.
        """

        return cls(
            parameters=source_dict["parameters"],
            desired_retention=source_dict["desired_retention"],
            learning_steps=[
                timedelta(seconds=learning_step)
                for learning_step in source_dict["learning_steps"]
            ],
            relearning_steps=[
                timedelta(seconds=relearning_step)
                for relearning_step in source_dict["relearning_steps"]
            ],
            maximum_interval=source_dict["maximum_interval"],
            enable_fuzzing=source_dict["enable_fuzzing"],
        )

    def to_json(self, indent: int | str | None = None) -> str:
        """
        Returns a JSON-serialized string of the Scheduler object.

        Args:
            indent: Equivalent argument to the indent in json.dumps()

        Returns:
            str: A JSON-serialized string of the Scheduler object.
        """

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, source_json: str) -> Self:
        """
        Creates a Scheduler object from a JSON-serialized string.

        Args:
            source_json: A JSON-serialized string of an existing Scheduler object.

        Returns:
            Self: A Scheduler object created from the JSON string.
        """

        source_dict: SchedulerDict = json.loads(source_json)
        return cls.from_dict(source_dict=source_dict)

    @overload
    def _clamp_difficulty(self, *, difficulty: float) -> float: ...
    @overload
    def _clamp_difficulty(self, *, difficulty: Tensor) -> Tensor: ...
    def _clamp_difficulty(self, *, difficulty: float | Tensor) -> float | Tensor:
        if isinstance(difficulty, (int, float)):
            difficulty = min(max(difficulty, MIN_DIFFICULTY), MAX_DIFFICULTY)
        else:
            difficulty = difficulty.clamp(min=MIN_DIFFICULTY, max=MAX_DIFFICULTY)

        return difficulty

    @overload
    def _clamp_stability(self, *, stability: float) -> float: ...
    @overload
    def _clamp_stability(self, *, stability: Tensor) -> Tensor: ...
    def _clamp_stability(self, *, stability: float | Tensor) -> float | Tensor:
        if isinstance(stability, (int, float)):
            stability = max(stability, STABILITY_MIN)
        else:
            stability = stability.clamp(min=STABILITY_MIN)

        return stability

    def _initial_stability(self, *, rating: Rating) -> float:
        initial_stability = self.parameters[rating - 1]

        initial_stability = self._clamp_stability(stability=initial_stability)

        return initial_stability

    def _initial_difficulty(self, *, rating: Rating, clamp: bool) -> float:
        initial_difficulty = (
            self.parameters[4] - (math.e ** (self.parameters[5] * (rating - 1))) + 1
        )

        if clamp:
            initial_difficulty = self._clamp_difficulty(difficulty=initial_difficulty)

        return initial_difficulty

    def _next_interval(self, *, stability: float) -> int:
        next_interval = (stability / self._FACTOR) * (
            (self.desired_retention ** (1 / self._DECAY)) - 1
        )

        if not isinstance(next_interval, (int, float)):
            next_interval = next_interval.detach().item()

        next_interval = round(next_interval)  # intervals are full days

        # must be at least 1 day long
        next_interval = max(next_interval, 1)

        # can not be longer than the maximum interval
        next_interval = min(next_interval, self.maximum_interval)

        return next_interval

    def _short_term_stability(self, *, stability: float, rating: Rating) -> float:
        short_term_stability_increase = (
            math.e ** (self.parameters[17] * (rating - 3 + self.parameters[18]))
        ) * (stability ** -self.parameters[19])

        if rating in (Rating.Hard, Rating.Good, Rating.Easy):
            if isinstance(short_term_stability_increase, (int, float)):
                short_term_stability_increase = max(short_term_stability_increase, 1.0)
            else:
                short_term_stability_increase = short_term_stability_increase.clamp(
                    min=1.0
                )

        short_term_stability = stability * short_term_stability_increase

        short_term_stability = self._clamp_stability(stability=short_term_stability)

        return short_term_stability

    def _next_difficulty(self, *, difficulty: float, rating: Rating) -> float:
        def _linear_damping(*, delta_difficulty: float, difficulty: float) -> float:
            return (10.0 - difficulty) * delta_difficulty / 9.0

        def _mean_reversion(*, arg_1: float, arg_2: float) -> float:
            return self.parameters[7] * arg_1 + (1 - self.parameters[7]) * arg_2

        arg_1 = self._initial_difficulty(rating=Rating.Easy, clamp=False)

        delta_difficulty = -(self.parameters[6] * (rating - 3))
        arg_2 = difficulty + _linear_damping(
            delta_difficulty=delta_difficulty, difficulty=difficulty
        )

        next_difficulty = _mean_reversion(arg_1=arg_1, arg_2=arg_2)

        next_difficulty = self._clamp_difficulty(difficulty=next_difficulty)

        return next_difficulty

    def _next_stability(
        self,
        *,
        difficulty: float,
        stability: float,
        retrievability: float,
        rating: Rating,
    ) -> float:
        if rating == Rating.Again:
            next_stability = self._next_forget_stability(
                difficulty=difficulty,
                stability=stability,
                retrievability=retrievability,
            )

        elif rating in (Rating.Hard, Rating.Good, Rating.Easy):
            next_stability = self._next_recall_stability(
                difficulty=difficulty,
                stability=stability,
                retrievability=retrievability,
                rating=rating,
            )

        else:
            raise ValueError(f"Unknown rating: {rating}")

        next_stability = self._clamp_stability(stability=next_stability)

        return next_stability

    def _next_forget_stability(
        self, *, difficulty: float, stability: float, retrievability: float
    ) -> float:
        next_forget_stability_long_term_params = (
            self.parameters[11]
            * (difficulty ** -self.parameters[12])
            * (((stability + 1) ** (self.parameters[13])) - 1)
            * (math.e ** ((1 - retrievability) * self.parameters[14]))
        )

        next_forget_stability_short_term_params = stability / (
            math.e ** (self.parameters[17] * self.parameters[18])
        )

        return min(
            next_forget_stability_long_term_params,
            next_forget_stability_short_term_params,
        )

    def _next_recall_stability(
        self,
        *,
        difficulty: float,
        stability: float,
        retrievability: float,
        rating: Rating,
    ) -> float:
        hard_penalty = self.parameters[15] if rating == Rating.Hard else 1
        easy_bonus = self.parameters[16] if rating == Rating.Easy else 1

        return stability * (
            1
            + (math.e ** (self.parameters[8]))
            * (11 - difficulty)
            * (stability ** -self.parameters[9])
            * ((math.e ** ((1 - retrievability) * self.parameters[10])) - 1)
            * hard_penalty
            * easy_bonus
        )

    def _get_fuzzed_interval(self, *, interval: timedelta) -> timedelta:
        """
        Takes the current calculated interval and adds a small amount of random fuzz to it.
        For example, a card that would've been due in 50 days, after fuzzing, might be due in 49, or 51 days.

        Args:
            interval: The calculated next interval, before fuzzing.

        Returns:
            timedelta: The new interval, after fuzzing.
        """

        interval_days = interval.days

        if interval_days < 2.5:  # fuzz is not applied to intervals less than 2.5
            return interval

        def _get_fuzz_range(*, interval_days: int) -> tuple[int, int]:
            """
            Helper function that computes the possible upper and lower bounds of the interval after fuzzing.
            """

            delta = 1.0
            for fuzz_range in FUZZ_RANGES:
                delta += fuzz_range["factor"] * max(
                    min(float(interval_days), fuzz_range["end"]) - fuzz_range["start"],
                    0.0,
                )

            min_ivl = round(interval_days - delta)
            max_ivl = round(interval_days + delta)

            # make sure the min_ivl and max_ivl fall into a valid range
            min_ivl = max(2, min_ivl)
            max_ivl = min(max_ivl, self.maximum_interval)
            min_ivl = min(min_ivl, max_ivl)

            return min_ivl, max_ivl

        min_ivl, max_ivl = _get_fuzz_range(interval_days=interval_days)

        fuzzed_interval_days = (
            random() * (max_ivl - min_ivl + 1)
        ) + min_ivl  # the next interval is a random value between min_ivl and max_ivl

        fuzzed_interval_days = min(round(fuzzed_interval_days), self.maximum_interval)

        fuzzed_interval = timedelta(days=fuzzed_interval_days)

        return fuzzed_interval


__all__ = ["NextStates", "Scheduler", "Trace", "TraceRecord"]
