'''
As the conversation goes on, summarize sections and add them to context?
Create long term (summarized from conversation pieces) and short term
(original unaltered bits of conversation to add as context).
'''
import re

from llm.other.conversations import Conversation
from llm.models.gpt_oss_20b import GptOss20b


if __name__ == '__main__':

    model = GptOss20b()
    model.load(location=r'/home/eric/Repos/model_cache')

    c = Conversation()
    c.set_overall_prompt(text='Your name is Seamus OFinnegan. Respond as the assistant in character.')
    c.add_context(text='Youre a real salt of the earth Irish coal miner.')
    c.add_context(text='You get straight to the point and dont waste words on small talk.')
    c.add_context(text='You are married to your wife of 10 years, Gurdy. Your favorite hobby is fighting.')

    # Loop:
    '''
    1. get user response.
    2. Ask AI to continue the conversation.
    '''

    while True:
        user_response = input('[User]: ')

        if user_response == 'stop':
            break

        c.add_response(role='user', text=user_response)

        system_response = model.ask(prompt=c)

        print(f'[Seamus]: {system_response}\n')

        c.add_response(role='assistant', text=system_response)