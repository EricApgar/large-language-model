'''
TODO: 
- Have a conversation size window that drops earlier entries if the conversation
    moves beyond a set limit. Might need to add a length field to Response.
'''
from dataclasses import dataclass
from typing import List, Literal

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation as HarmonyConversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
    RenderConversationConfig)


@dataclass
class Response:
    role: Literal['user', 'assistant']
    text: str


class Conversation:

    def __init__(self, id: str=None):
    
        self.id: str = id  # ? Make this generated on Conversation creation?
        
        self.participants: list[str] = []
        
        self.overall_prompt: str = None
        self.context: list[str] = []
        self.history: list[Response] = []
        
        self.reasoning_level: ReasoningEffort = ReasoningEffort.LOW

        self.harmony_convo: HarmonyConversation = None 
        self.full_text: str = None

        # self._generate_id():


    def set_reasoning_level(self, level: str):
        
        VALID_LEVELS = ('low', 'medium', 'high')

        if level not in VALID_LEVELS:
            raise ValueError(f'Invalid level. Valid options are {VALID_LEVELS}!')

        if level == 'low':
            self.reasoning_level = ReasoningEffort.LOW
        elif level == 'medium':
            self.reasoning_level = ReasoningEffort.MEDIUM
        elif level == 'high':
            self.reasoning_level = ReasoningEffort.HIGH

        return


    def set_overall_prompt(self, text: str):

        self.overall_prompt = text

        self._make_harmony_convo()

        return


    def add_context(self, text: str):

        self.context.append(text)

        self._make_harmony_convo()

        return


    def add_response(self, role: str, text: str):

        VALID_ROLES = ('user', 'system')

        if role not in VALID_ROLES:
            raise ValueError(f'Invalid role. Valid options are {VALID_ROLES}!')

        if role not in self.participants:
            self.participants.append(role)

        self.history.append(Response(role=role, text=text))
        
        self._make_harmony_convo()

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

        if self.history:
            conversation = ''.join([f'[{i.name}]: {i.text}\n\n' for i in self.history])
        else:
            conversation = ''

        self.full_text = prompt + context + conversation

        return


    def _make_harmony_convo(self) -> HarmonyConversation:

        # System Details about the Overall Conversation.
        system_msg = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(self.reasoning_level))
        
        developer_msg = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(self.overall_prompt))

        msgs: List[Message] = [system_msg, developer_msg]

        # Background Context Information.
        if self.context:
            context_block = '\n'.join([
                "BACKGROUND CONTEXT (not part of the dialogue):",
                ''.join([f'{i}\n' for i in self.context]),
                "END BACKGROUND CONTEXT"])

            msgs.append(Message.from_role_and_content(Role.USER, context_block))

        # Conversation history between user and AI.
        for turn in self.history:
            if turn.role == "user":
                msgs.append(Message.from_role_and_content(Role.USER, turn.text))
            else:
                msgs.append(Message.from_role_and_content(Role.ASSISTANT, turn.text))

        self.harmony_convo = HarmonyConversation.from_messages(msgs)

        return


if __name__ == '__main__':

    c = Conversation()
    c.set_overall_prompt(text='Your name is Seamus OFinnegan. Youre a hard, crusty, salt of the earth Irish dockworker with a strong accent.')
    c.add_context(text='You get straight to the point and dont waste words on small talk.')
    c.add_context(text='You are married to your wife of 10 years, Gurdy. Your favorite hobby is fighting.')

    c.add_response(role='user', text='Hi, how are you?')

    pass
