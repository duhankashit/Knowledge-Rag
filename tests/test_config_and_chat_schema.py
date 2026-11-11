import unittest

from pydantic import ValidationError

from app.core.config import Settings
from app.schemas import ChatRequest


class SettingsAndChatSchemaTests(unittest.TestCase):
    def test_huggingface_model_id_is_derived_from_model_url(self) -> None:
        settings = Settings(
            _env_file=None,
            huggingface_model_url="https://huggingface.co/example/model/tree/main",
        )

        self.assertEqual(settings.huggingface_model_id, "example/model")

    def test_huggingface_urls_must_use_https(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, huggingface_endpoint="http://huggingface.co")

    def test_chat_history_is_transient_and_bounded(self) -> None:
        request = ChatRequest(
            question="Follow up",
            history=[{"role": "assistant", "content": "Previous answer"}],
        )

        self.assertEqual(request.history[0].role, "assistant")
        with self.assertRaises(ValidationError):
            ChatRequest(
                question="Follow up",
                history=[
                    {"role": "user", "content": f"message {index}"}
                    for index in range(13)
                ],
            )


if __name__ == "__main__":
    unittest.main()
