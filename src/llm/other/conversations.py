'''
TODO: 
- Have a conversation size window that drops earlier entries if the conversation
    moves beyond a set limit. Might need to add a length field to Response.
'''

from dataclasses import dataclass


class Conversation:

    def __init__(self, id: str=None):
    
        self.id: str = id  # ? Make this generated on Conversation creation?
        
        self.participants: list[str] = []
        
        self.prompt: str = ''
        self.context: list[str] = []
        self.conversation: list[Response] = []
        
        self.full_text: str = None

        # self._generate_id():


    def set_prompt(self, text: str):

        self.prompt = text

        self._make_full_text()

        return


    def add_context(self, text: str):

        self.context.append(text)

        self._make_full_text()

        return


    def add_response(self, name: str, text: str):

        if name not in self.participants:
            self.participants.append(name)

        self.conversation.append(Response(name=name, text=text))
        
        self._make_full_text()

        return
    

    def _make_full_text(self):
        '''
        Generate the full text of the conversation from pieces.
        '''

        prompt = self.prompt + '\n\n'

        if self.context:
            context = 'Context Start:\n' + ''.join([f'{i}\n\n' for i in self.context]) + 'Context End.\n\n'
        else:
            context = ''

        if self.conversation:
            conversation = ''.join([f'[{i.name}]: {i.text}\n\n' for i in self.conversation])
        else:
            conversation = ''

        self.full_text = prompt + context + conversation

        return


@dataclass
class Response:
    name: str
    text: str


if __name__ == '__main__':

    pass
