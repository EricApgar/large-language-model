'''
TODO: 
- Have a conversation size window that drops earlier entries if the conversation
    moves beyond a set limit. Might need to add a length field to Response.
'''
from dataclasses import dataclass
from typing import List, Literal

from openai_harmony import (
    Conversation as HarmonyConversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort)


@dataclass
class Response:
    role: Literal['user', 'assistant']
    text: str


class Conversation:

    def __init__(self, id: str=None):

        self.participants: list[str] = []

        self.overall_prompt: str = None
        self.context: list[str] = []
        self.history: list[Response] = []

        self.reasoning_level: ReasoningEffort = ReasoningEffort.LOW

        self.harmony_convo: HarmonyConversation = None


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

        self._build_harmony()

        return


    def add_context(self, text: str):
        '''
        Add background context the system can reference.
        '''

        self.context.append(text)

        self._build_harmony()

        return


    def add_response(self, role: str, text: str):
        '''
        Add either a user or assistant (currently only these
        two supported) response to an ongoing conversation.
        '''

        VALID_ROLES = ('user', 'assistant')

        if role not in VALID_ROLES:
            raise ValueError(f'Invalid role. Valid options are {VALID_ROLES}!')

        if role not in self.participants:
            self.participants.append(role)

        self.history.append(Response(role=role, text=text))

        self._build_harmony()

        return
    

    def to_dict(self):
        '''
        Generate a dictionary equivalent to the harmony conversation.
        This let's you create an API friendly data object that can be
        easily reconstituted into a conversation.
        '''

        result = {}

        result['reasoning_level'] = self.reasoning_level
        result['overall_prompt'] = self.overall_prompt  # str.
        result['context'] = self.context  # List of str.
        result['history'] = [(h.role, h.text) for h in self.history]

        return result


    def from_dict(self, data: dict):
        '''
        Build an instance of Conversation() from a dictionary.
        '''

        NEEDED_FIELDS = (
            'reasoning_level',
            'overall_prompt',
            'context',
            'history')
        
        if not set(NEEDED_FIELDS) == set(data.keys()):
            raise ValueError('Input "data" is unaligned to needed keys!')

        self.reasoning_level = data['reasoning_level']
        self.overall_prompt = data['overall_prompt']
        self.context = data['context']
        self.history = [Response(role=h[0], text=h[1]) for h in data['history']]

        self._build_harmony()

        return


    def _build_harmony(self) -> HarmonyConversation:
        '''
        Build a Harmony-Conversation object.
        '''

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
                '\n'.join(self.context),
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
    c.add_response(role='assistant', text='None of your business.')

    pass
