import unittest

from app.control.model import registry
from app.control.model.enums import Capability, ModeId, Tier


class GrokChatFastRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = registry.resolve("grok-chat-fast")

    def test_grok_chat_fast_is_web_fast_chat_model(self):
        self.assertEqual(self.spec.mode_id, ModeId.FAST)
        self.assertEqual(self.spec.mode_id.to_api_str(), "fast")
        self.assertEqual(self.spec.tier, Tier.BASIC)
        self.assertTrue(self.spec.capability & Capability.CHAT)
        self.assertTrue(self.spec.is_chat())
        self.assertFalse(self.spec.is_console_chat())

    def test_grok_chat_fast_uses_basic_first_pool_order(self):
        self.assertEqual(self.spec.pool_candidates(), (0, 1, 2))

    def test_webui_model_data_source_includes_grok_chat_fast(self):
        # /webui/api/models is populated from registry.list_enabled().
        enabled = {spec.model_name: spec for spec in registry.list_enabled()}
        self.assertIn("grok-chat-fast", enabled)
        self.assertEqual(enabled["grok-chat-fast"].public_name, "Grok Chat Fast")


if __name__ == "__main__":
    unittest.main()
