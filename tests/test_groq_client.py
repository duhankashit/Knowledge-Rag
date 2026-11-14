import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from groq import APIStatusError

from app.clients.grok import GrokClient


def api_status_error(status_code: int, message: str) -> APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(message, response=response, body={"error": message})


class GrokClientTests(unittest.TestCase):
    def test_uses_fallback_when_primary_model_is_unavailable(self) -> None:
        client = GrokClient(
            "test-key",
            "https://api.groq.com/openai/v1",
            "primary-model",
            "fallback-model",
        )
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            api_status_error(404, "model not found"),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="  answer from fallback  ")
                    )
                ]
            ),
        ]
        client._client = mock_client

        answer = client.generate([{"role": "user", "content": "question"}])

        self.assertEqual(answer, "answer from fallback")
        self.assertEqual(
            [
                call.kwargs["model"]
                for call in mock_client.chat.completions.create.call_args_list
            ],
            ["primary-model", "fallback-model"],
        )

    def test_does_not_fallback_on_non_model_api_errors(self) -> None:
        client = GrokClient(
            "test-key",
            "https://api.groq.com/openai/v1",
            "primary-model",
            "fallback-model",
        )
        mock_client = Mock()
        error = api_status_error(429, "rate limit exceeded")
        mock_client.chat.completions.create.side_effect = error
        client._client = mock_client

        with self.assertRaises(APIStatusError):
            client.generate([{"role": "user", "content": "question"}])

        mock_client.chat.completions.create.assert_called_once()

    def test_uses_selected_model_before_configured_fallback(self) -> None:
        client = GrokClient(
            "test-key",
            "https://api.groq.com/openai/v1",
            "primary-model",
            "fallback-model",
        )
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        client._client = mock_client

        answer = client.generate(
            [{"role": "user", "content": "question"}], model="selected-model"
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["model"],
            "selected-model",
        )


if __name__ == "__main__":
    unittest.main()
