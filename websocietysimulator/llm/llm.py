from typing import Dict, List, Optional, Union
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from .infinigence_embeddings import InfinigenceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.generativeai as genai
import logging
logger = logging.getLogger("websocietysimulator")

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
class OllamaLLM(LLMBase):
    def __init__(self, model: str = "llama3"):
        """
        Initialize Ollama LLM running locally.
        
        Args:
            model: Model name (must be pulled first via 'ollama pull modelname')
                   Defaults to "llama3"
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key="ollama", # Required by client but ignored by Ollama
            base_url="http://127.0.0.1:11434/v1" # Points to your local machine
        )
        # Ollama does support embeddings, but often requires a specific model (e.g., 'nomic-embed-text').
        # Setting to None for now to match your other classes.
        self.embedding_model = None 
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Wait 2s to 60s
        stop=stop_after_attempt(5)  # Retry up to 5 times
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 2048, stop_strs: Optional[List[str]] = None, n: int = 1, response_format: Optional[Dict[str, str]] = None) -> Union[str, List[str]]:
        """
        Call local Ollama API to get response.
        
        Args:
            messages: List of input messages (role/content dicts)
            model: Optional model override
            temperature: Defaults to 0.0 for deterministic outputs
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
                response_format=response_format
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
                
        except Exception as e:
            # Basic error handling for local connection issues
            error_msg = str(e)
            if "Connection refused" in error_msg:
                logger.error("Ollama Connection Refused: Is Ollama running? (Run 'ollama serve' in terminal)")
            else:
                logger.error(f"Ollama LLM Error: {error_msg}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class GeminiLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """
        Gemini LLM through the OpenAI-compatible API
        """
        super().__init__(model)

        # IMPORTANT ↓↓↓
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.embedding_model_name = "models/embedding-001"

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        print("Gemini Messages:", messages)
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # stop=stop_strs,
            n=n
        )

        print("Gemini Response:", response)
        if n == 1:
            return response.choices[0].message.content

        return [choice.message.content for choice in response.choices]

    # -------------------------
    # Embeddings
    # -------------------------
    def get_embedding_model(self):
        return self
    
    def embed(self, text: str):
        """
        Create embeddings using Gemini's embedding model
        """
        res = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        return res.data[0].embedding
# class GeminiLLM(LLMBase):
#     """
#     Wrapper for Google Gemini models that mirrors the interface of DeepseekLLM.
#     """

#     def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
#         """
#         Initialize Gemini LLM
        
#         Args:
#             api_key: Google API key (https://aistudio.google.com/app/apikey)
#             model: Default Gemini model ("gemini-1.5-pro" recommended)
#         """
#         super().__init__(model)

#         genai.configure(api_key=api_key)
#         self.client = genai.GenerativeModel(model)
        
#         # Gemini *does* support embeddings, but with a different API.
#         # For consistency with your Deepseek wrapper, treat embeddings as optional.
#         self.embedding_model = None

#     @retry(
#         retry=retry_if_exception_type(Exception),
#         wait=wait_exponential(multiplier=1, min=10, max=300),
#         stop=stop_after_attempt(10)
#     )
#     def __call__(
#         self,
#         messages: List[Dict[str, str]],
#         model: Optional[str] = None,
#         temperature: float = 0.1,
#         max_tokens: int = 1000,
#         stop_strs: Optional[List[str]] = None,
#         n: int = 1,
#     ) -> Union[str, List[str]]:
#         """
#         Call Gemini API using the same interface as DeepSeekLLM.

#         Args:
#             messages: [{"role": "...", "content": "..."}]
#             model: Optional override of the Gemini model
#             temperature: Sampling temperature
#             max_tokens: Max output tokens
#             stop_strs: Optional stop strings
#             n: Number of completions to generate

#         Returns:
#             str or list[str]
#         """
#         try:
#             gemini_messages = []
#             system_instruction = None

#             # 1. Map OpenAI roles to Gemini roles
#             for m in messages:
#                 if m["role"] == "system":
#                     # Extract system prompt to pass separately
#                     system_instruction = m["content"]
#                 elif m["role"] == "user":
#                     gemini_messages.append({"role": "user", "parts": [{"text": m["content"]}]})
#                 elif m["role"] == "assistant":
#                     gemini_messages.append({"role": "model", "parts": [{"text": m["content"]}]})

#             # 2. Initialize Model with System Instruction (if present)
#             g_model = genai.GenerativeModel(
#                 model_name=model,
#                 system_instruction=system_instruction
#             )

#             # 3. Generate Content
#             response = g_model.generate_content(
#                 contents=gemini_messages,
#                 generation_config={
#                     "temperature": temperature,
#                     "max_output_tokens": max_tokens,
#                     "stop_sequences": stop_strs,
#                     "candidate_count": n,
#                 },
#             )

#             # 4. Extract Text safely
#             outputs = []
#             if response.candidates:
#                 for candidate in response.candidates:
#                     # Check if the response was blocked by safety filters
#                     if candidate.finish_reason != 1: # 1 = STOP (Success)
#                         # Handle safety block or other finish reasons
#                         outputs.append(f"[Blocked: {candidate.finish_reason.name}]")
#                         continue
                    
#                     if candidate.content.parts:
#                         outputs.append(candidate.content.parts[0].text)
#                     else:
#                         outputs.append("")
#             else:
#                 # Fallback if no candidates returned (rare, usually strict safety filters)
#                 outputs.append("")

#             if n == 1 and outputs:
#                 return outputs[0]
#             return outputs

#         except Exception as e:
#             msg = str(e)
#             if "429" in msg:
#                 # logger.warning(f"Gemini Rate Limit Error: {msg}")
#                 pass
#             elif "quota" in msg.lower():
#                 # logger.error("Gemini Quota Exceeded.")
#                 pass
#             else:
#                 # logger.error(f"Gemini LLM Error: {msg}")
#                 pass
#             raise e

#     def get_embedding_model(self):
#         """
#         Gemini technically supports embeddings (via genai.embed_text),
#         but to match the DeepseekLLM interface we return None unless you want it.
#         """
#         if self.embedding_model is None:
#             logger.warning("Gemini embedding model not set. Returning None.")
#         return self.embedding_model
class DeepseekLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        """
        Initialize DeepSeek LLM
        
        Args:
            api_key: DeepSeek API key (get from https://platform.deepseek.com)
            model: Model name, defaults to "deepseek-chat" (DeepSeek-V3)
                   Use "deepseek-reasoner" for DeepSeek-R1
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        # DeepSeek does not currently offer an embedding API. 
        # If your base class requires this attribute, consider using OpenAI or a local model.
        self.embedding_model = None 
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # Wait 10s to 300s
        stop=stop_after_attempt(10)  # Retry up to 10 times
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 1.0, max_tokens: int = 4096, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call DeepSeek API to get response with rate limit handling
        
        Args:
            messages: List of input messages (role/content dicts)
            model: Optional model override
            temperature: Defaults to 1.0 (DeepSeek recommends higher temp than OpenAI)
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        try:
            # DeepSeek uses 'stop' instead of 'stop_strs' in the API call
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
                stream=False
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
                
        except Exception as e:
            # DeepSeek specific error handling
            error_msg = str(e)
            if "429" in error_msg:
                logger.warning(f"DeepSeek Rate limit exceeded: {error_msg}")
            elif "402" in error_msg:
                logger.error("DeepSeek Insufficient Balance: Please top up your account.")
            else:
                logger.error(f"DeepSeek LLM Error: {error_msg}")
            raise e
    
    def get_embedding_model(self):
        if self.embedding_model is None:
            logger.warning("DeepSeek does not support embeddings. Returning None.")
        return self.embedding_model

class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://cloud.infini-ai.com/maas/v1"
        )
        self.embedding_model = InfinigenceEmbeddings(api_key=api_key)
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    
    def get_embedding_model(self):
        return self.embedding_model 
