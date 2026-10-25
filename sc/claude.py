"""Claude API helper: one request, streamed (required by the SDK for long requests), final message returned."""
from __future__ import annotations


def final_message(client, params: dict):
    with client.messages.stream(**params) as stream:
        return stream.get_final_message()
