import openai
import os
from typing import List
import asyncio


class OpenAIChat():
    # more details on: https://platform.openai.com/docs/api-reference/chat
    def __init__(
        self, 
        model_name='gpt-3.5-turbo', 
        max_tokens=2500, 
        temperature=0.5, 
        top_p=1,
        request_timeout=180, 
        stop=None, 
        response_format='text', # text or json_object
        logprobs=False, 
        top_logprobs=None,
        n=1,
    ):
        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,
            'stop': stop,
            'response_format': response_format,
            'logprobs': logprobs,
            'top_logprobs': top_logprobs,
            'sample_n': n,
            }
        
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = "https://api.ai-gaochao.cn/v1"

        self.history = []

    async def dispatch_openai_requests(
        self,
        messages_list,
    ) -> List[str]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """
        async def _request_with_retry(messages, retry=3):
            for try_i in range(retry):
                try:
                    # for text embedding models
                    if "embedding" in self.config['model_name']:
                        response = await openai.Embedding.acreate(
                            model=self.config['model_name'],
                            input=messages,
                        )
                    else:
                        # for chat models
                        response = await openai.ChatCompletion.acreate(
                            model=self.config['model_name'],
                            response_format={'type': self.config['response_format']},
                            messages=messages,
                            max_tokens=self.config['max_tokens'],
                            temperature=self.config['temperature'],
                            top_p=self.config['top_p'],
                            request_timeout=self.config['request_timeout'],
                            stop=self.config['stop'],
                            logprobs=self.config['logprobs'],
                            top_logprobs=self.config['top_logprobs'],
                            n=self.config['sample_n'],
                        )
                    
                    return response
                
                except openai.error.InvalidRequestError as e:
                    print(e)
                    print(f'Retry {try_i+1} Invalid request error, waiting for 3 second...')
                    await asyncio.sleep(1)
                except openai.error.RateLimitError:
                    print(f'Retry {try_i+1} Rate limit error, waiting for 40 second...')
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print(f'Retry {try_i+1} API error, waiting for 5 second...')
                    await asyncio.sleep(5)
                except openai.error.AuthenticationError as e:
                    print(e)
                    print(f'Retry {try_i+1} Authentication error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.Timeout:
                    print(f'Retry {try_i+1} Timeout error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.APIConnectionError as e:
                    print(e)
                    print(f'Retry {try_i+1} API connection error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.ServiceUnavailableError:
                    print(f'Retry {try_i+1} Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)
                
            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)
    
    def normalize_messages(self, messages):
        if isinstance(messages, str):
            """
            messages_list = 'How are you?'
            """
            return [[{'role': 'user', 'content': messages}]]
        
        elif isinstance(messages, list):
            if all(isinstance(item, str) for item in messages):
                """
                messages_list = ['How are you?', 'How are you?', 'How are you?',...]
                """
                return [[{'role': 'user', 'content': msg}] for msg in messages]
            elif all(isinstance(item, dict) for item in messages):
                """
                mesaages_list = [{'role': 'user', 'content': 'How are you?'}]
                """
                return [messages]
            elif all(isinstance(item, list) and all(isinstance(subitem, dict) for subitem in item) for item in messages):
                """mesaages_list = [
                    [{'role': 'user', 'content': 'How are you?'}],
                    [{'role': 'user', 'content': 'How are you?'}], ...
                ]"""
                return messages
            else:
                raise ValueError("List elements are not consistent.")
        else:
            raise TypeError("Unsupported type for messages_list.")
    
    async def async_run(self, messages_list):
        retry = 1

        responses = [None for _ in range(len(messages_list))]

        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            # print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
            
            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
            )

            if "embedding" in self.config['model_name']:
                preds = [prediction['data'][0]['embedding'] if prediction is not None else None for prediction in predictions]
            else:
                if self.config['logprobs'] == False:
                    preds = [prediction['choices'][0]['message']['content'] if prediction is not None else None for prediction in predictions]
                else:
                    preds = [
                        [
                            prediction['choices'][0]['message']['content'],
                            [d['logprob'] for d in prediction['choices'][0]['logprobs']['content']]
                        ] if prediction is not None else None for prediction in predictions
                    ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])
            
            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
            
            retry -= 1
        
        return responses
    
    def batch_run(self, messages):
        """A synchronous wrapper for the async_run method."""
        messages = self.normalize_messages(messages)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_run(messages))

    def clear_history(self):
        self.history = []
    
    def chat_run(self, messages: str, clear=False):
        if clear:
            self.clear_history()
        self.history += [{'role': 'user', 'content': messages}]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(self.async_run([self.history]))
        self.history += [{'role': 'assistant', 'content': response[0]}]
        print(self.history)
        return response[0]
    