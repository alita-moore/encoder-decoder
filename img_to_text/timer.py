from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel
import torch

__all__ = [
    "Timer",
]


class EventType(Enum):
    STANDARD = 1
    REFERENCE_TIME = 2
    CHILD = 3


class Timer(BaseModel):
    events: (
        list[
            tuple[
                float,
                Literal[EventType.REFERENCE_TIME],
            ]
            | tuple[Timer, Literal[EventType.CHILD]]
            | tuple[tuple[str, float], Literal[EventType.STANDARD]]
        ]
        | None
    ) = None
    depth: int = 0
    name: str
    _stop_time: float | None = None
    # NOTE: we allow for the timer to be disabled so that we can avoid non-master
    # processes from logging the timer events
    enabled: bool = True
    synchronize: Callable[[], None] | None = lambda: torch.cuda.synchronize()

    def model_post_init(self, __context: Any) -> None:
        self.reset()

    def reset(self):
        self.events = [(time.time(), EventType.REFERENCE_TIME)]

    @property
    def _events(self):
        if self.events is None:
            raise ValueError("No events found")
        return self.events

    def _get_latest_reference_time(self):
        for event in reversed(self._events):
            if event[1] == EventType.REFERENCE_TIME:
                return event[0]
        else:
            raise ValueError("No reference time found")

    def push_event(self, event_name: str, log_now: bool = True):
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return
        if not self.enabled:
            return
        if self.synchronize is not None:
            self.synchronize()
        latest_event = self._get_latest_reference_time()
        self._events.append(
            ((event_name, time.time() - latest_event), EventType.STANDARD)
        )
        if log_now:
            logging.debug(
                f"{self.indent} {event_name}: {time.time() - latest_event:.5f}s"
            )
        self.push_reference_time()

    @property
    def is_stopped(self):
        return self._stop_time is not None

    def stop(self):
        if not self.enabled:
            return
        if self._stop_time is not None:
            raise ValueError("Timer already stopped")
        if self.synchronize is not None:
            self.synchronize()
        self._stop_time = time.time()
        for event in self._events:
            if event[1] == EventType.CHILD:
                if not event[0].is_stopped:
                    raise ValueError(f"Child timer ({event[0].name}) not stopped")

    def get_total_time(self):
        if not self.enabled:
            return 0
        if self._stop_time is None:
            raise ValueError("Timer not stopped")
        assert self._events[0][1] == EventType.REFERENCE_TIME
        return self._stop_time - self._events[0][0]

    def push_reference_time(self):
        if not self.enabled:
            return
        self._events.append((time.time(), EventType.REFERENCE_TIME))

    def gen_child(self, name: str):
        child = Timer(name=name, enabled=self.enabled)
        child.depth = self.depth + 1
        self._events.append((child, EventType.CHILD))
        return child

    @property
    def indent(self):
        return "--" * self.depth

    def report(self):
        if not self.enabled:
            return
        if self.depth == 0:
            logging.debug(
                f"=== Timer {self.name}: Total time {self.get_total_time():.5f}s ==="
            )
        for event in self._events:
            match event[1]:
                case EventType.REFERENCE_TIME:
                    continue
                case EventType.CHILD:
                    child = event[0]
                    total_time = child.get_total_time()
                    # NOTE: we could also do this in the child call / in an else after checking
                    # if the depth is 0, but I choose not to do this because I generally find
                    # this easier to reason about.
                    logging.debug(f"{child.indent} {child.name} {child.indent}")
                    child.report()
                    logging.debug(
                        f"{child.indent} Total: {total_time:.5f}s ({total_time / self.get_total_time():.2%} of {self.name})"
                    )
                case EventType.STANDARD:
                    name, duration = event[0]
                    logging.debug(
                        f"{self.indent} {name}: {duration:.5f}s ({duration / self.get_total_time():.2%} of {self.name})"
                    )
