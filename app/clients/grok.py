import logging

from groq import APIStatusError, Groq

logger = logging.getLogger(__name__)


class GrokClient:
    def __init__(
        self, api_key: str, base_url: str, model: str, fallback_model: str = ""
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.fallback_model = fallback_model
        self._client: Groq | None = None

    @property
    def client(self) -> Groq:
        if not self.api_key.strip():
            raise RuntimeError("GROK_API_KEY is not configured.")
        if self._client is None:
            self._client = Groq(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        models = dict.fromkeys(
            candidate.strip()
            for candidate in (model or self.model, self.fallback_model)
            if candidate.strip()
        )
        last_model_error: APIStatusError | None = None
        for model in models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
            except APIStatusError as error:
                if error.status_code != 404 and not (
                    error.status_code == 400 and "model" in str(error).lower()
                ):
                    raise
                last_model_error = error
                logger.warning(
                    "Groq model %s unavailable; trying the next configured model.",
                    model,
                )
                continue

            answer = (
                response.choices[0].message.content
                if response.choices
                else None
            )
            if answer and answer.strip():
                if model != self.model:
                    logger.info("Groq response generated with fallback model %s.", model)
                return answer.strip()
            logger.warning("Groq model %s returned an empty response.", model)

        if last_model_error is not None:
            raise RuntimeError(
                f"All configured Groq models were unavailable: {', '.join(models)}."
            ) from last_model_error
        raise RuntimeError("All configured Groq models returned empty responses.")
