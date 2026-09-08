import os


class LiteLLMEngine:
    def __init__(self, llm_engine_name, api_key, api_endpoint):
        import openai
        self.llm_engine_name = llm_engine_name
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.client = openai.OpenAI(api_key=api_key, base_url=api_endpoint) 
        

    def respond(self, user_input, temperature = 0.7, top_p = 0.95, max_tokens = 40000):
        if os.environ.get('SCHOLAREVAL_OFFLINE') == '1' or os.environ.get('SCHOLAREVAL_NO_LLM') == '1':
            from ScholarEval.utils.workflow_errors import ConfigurationError
            raise ConfigurationError('Offline/no-LLM guard: API inference refused')
        response = self.client.chat.completions.create(model=self.llm_engine_name,messages=user_input, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


def LLMEngine(llm_engine_name=None, api_key=None, api_endpoint=None):
    """Keep existing imports/constructors working; LiteLLM remains the default."""
    backend = os.environ.get('SCHOLAREVAL_LLM_BACKEND', 'litellm').lower()
    if backend == 'codex':
        from .codex_session import get_codex_engine
        return get_codex_engine()
    if backend != 'litellm':
        raise ValueError('SCHOLAREVAL_LLM_BACKEND must be litellm or codex.')
    return LiteLLMEngine(llm_engine_name, api_key, api_endpoint)
