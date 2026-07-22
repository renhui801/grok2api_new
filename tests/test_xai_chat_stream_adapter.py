import json
import sys
import types
import unittest


def _clear_previous_app_stubs():
    for name, module in list(sys.modules.items()):
        if name != "app" and not name.startswith("app."):
            continue
        if getattr(module, "__file__", None) is None and not hasattr(module, "__path__"):
            sys.modules.pop(name, None)


class _DummyConfig:
    def get_bool(self, _key, default=False):
        return default


_clear_previous_app_stubs()

logger_stub = types.SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
)
sys.modules["app.platform.logging.logger"] = types.SimpleNamespace(logger=logger_stub)
sys.modules["app.platform.config.snapshot"] = types.SimpleNamespace(
    get_config=lambda: _DummyConfig()
)

from app.dataplane.reverse.protocol.xai_chat import StreamAdapter


class StreamAdapterImageCardTests(unittest.TestCase):
    def test_final_image_chunk_without_url_does_not_raise(self):
        adapter = StreamAdapter()
        card = {
            "id": "image_card",
            "image_chunk": {
                "progress": 100,
                "imageUuid": "image_1",
                "moderated": False,
            },
        }
        frame = {
            "result": {
                "response": {
                    "cardAttachment": {
                        "jsonData": json.dumps(card),
                    }
                }
            }
        }

        events = adapter.feed(json.dumps(frame))

        self.assertEqual([event.kind for event in events], ["image_progress"])
        self.assertEqual(adapter.image_urls, [])

    def test_image_chunk_accepts_image_url_alias(self):
        adapter = StreamAdapter()
        frame = {
            "result": {
                "response": {
                    "cardAttachments": [
                        {
                            "jsonData": {
                                "id": "image_card",
                                "imageChunk": {
                                    "progress": 100,
                                    "imageUuid": "image_2",
                                    "image_url": "users/user_1/generated/final/image.jpg",
                                    "moderated": False,
                                },
                            }
                        }
                    ]
                }
            }
        }

        events = adapter.feed(json.dumps(frame))

        self.assertEqual([event.kind for event in events], ["image_progress", "image"])
        self.assertEqual(
            adapter.image_urls,
            [
                (
                    "https://assets.grok.com/users/user_1/generated/final/image.jpg",
                    "image_2",
                )
            ],
        )

    def test_nested_response_error_does_not_abort_image_card(self):
        adapter = StreamAdapter()
        frame = {
            "result": {
                "response": {
                    "error": {
                        "message": "You've reached your usage limit. Please try again later."
                    },
                    "cardAttachment": {
                        "jsonData": json.dumps(
                            {
                                "id": "image_card",
                                "image_chunk": {
                                    "progress": 100,
                                    "imageUuid": "image_3",
                                    "imageUrl": "users/user_1/generated/final/image.jpg",
                                    "moderated": False,
                                },
                            }
                        )
                    },
                }
            }
        }

        events = adapter.feed(json.dumps(frame))

        self.assertEqual([event.kind for event in events], ["image_progress", "image"])
        self.assertEqual(
            adapter.image_urls,
            [
                (
                    "https://assets.grok.com/users/user_1/generated/final/image.jpg",
                    "image_3",
                )
            ],
        )

    def test_diagnostics_track_empty_image_signals(self):
        adapter = StreamAdapter()
        adapter.feed(
            json.dumps(
                {
                    "result": {
                        "response": {
                            "messageTag": "final",
                            "token": (
                                '<grok:render card_id="c1" card_type="generated_image_card" '
                                'type="render_generated_image"></grok:render>'
                            ),
                            "isThinking": False,
                        }
                    }
                }
            )
        )
        adapter.feed(
            json.dumps(
                {
                    "result": {
                        "response": {
                            "cardAttachment": {
                                "jsonData": json.dumps(
                                    {
                                        "id": "c1",
                                        "image_chunk": {
                                            "progress": 100,
                                            "imageUuid": "img",
                                            "moderated": False,
                                        },
                                    }
                                )
                            }
                        }
                    }
                }
            )
        )
        adapter.feed(
            json.dumps(
                {
                    "result": {
                        "response": {
                            "isSoftStop": True,
                        }
                    }
                }
            )
        )
        # soft_stop 之后的 modelResponse 兜底出图
        events = adapter.feed(
            json.dumps(
                {
                    "result": {
                        "response": {
                            "modelResponse": {
                                "generatedImageUrls": [],
                                "cardAttachmentsJson": [
                                    json.dumps(
                                        {
                                            "id": "c1",
                                            "image_chunk": {
                                                "progress": 100,
                                                "imageUuid": "img",
                                                "imageUrl": "users/u1/generated/final/image.jpg",
                                                "moderated": False,
                                            },
                                        }
                                    )
                                ],
                            }
                        }
                    }
                }
            )
        )

        diag = adapter.diagnostics()
        self.assertEqual(diag["final_missing_url"], 1)
        self.assertEqual(diag["soft_stop"], 1)
        self.assertEqual(diag["model_response"], 1)
        self.assertEqual(diag["model_response_card_json"], 1)
        self.assertEqual(diag["image_count"], 1)
        self.assertEqual([event.kind for event in events], ["image"])
        self.assertEqual(
            adapter.image_urls,
            [
                (
                    "https://assets.grok.com/users/u1/generated/final/image.jpg",
                    "img",
                )
            ],
        )

    def test_model_response_generated_image_urls_fallback(self):
        adapter = StreamAdapter()
        events = adapter.feed(
            json.dumps(
                {
                    "result": {
                        "response": {
                            "modelResponse": {
                                "generatedImageUrls": [
                                    "users/u1/generated/a/image.jpg"
                                ],
                                "cardAttachmentsJson": [],
                            }
                        }
                    }
                }
            )
        )
        self.assertEqual([event.kind for event in events], ["image"])
        self.assertEqual(
            adapter.image_urls,
            [
                (
                    "https://assets.grok.com/users/u1/generated/a/image.jpg",
                    "",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
