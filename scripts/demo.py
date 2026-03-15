'''
As the conversation goes on, summarize sections and add them to context?
Create long term (summarized from conversation pieces) and short term
(original unaltered bits of conversation to add as context).
'''
from llm_conversation import Conversation
from llm.models.gpt_oss_20b import GptOss20b


if __name__ == '__main__':

    model = GptOss20b()
    model_cache_dir = input(prompt='Enter model cache dir: ')
    model.load(location=r'/home/eric/Repos/model_cache')

    c = Conversation()
    c.set_overall_prompt(text='Your name is Samson McTavish. Respond in character.')
    c.add_context(text="You're a 3rd year student at a school for magic.")
    c.add_context(text='Your favorite class is potions.')
    c.add_context(text='Your favorite spell is "Mimble Wimble".')

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

        print(f'[Samson]: {system_response}\n')

        c.add_response(role='assistant', text=system_response)