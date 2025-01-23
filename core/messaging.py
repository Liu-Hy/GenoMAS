from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Set, Tuple, Optional


class Role(Enum):
    PI = "PI"
    GEO_AGENT = "GEO Agent"
    TCGA_AGENT = "TCGA Agent"
    STATISTICIAN_AGENT = "Statistician Agent"
    CODE_REVIEWER = "Code Reviewer"
    DOMAIN_EXPERT = "Domain Expert"


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    CODE_WRITING_REQUEST = "code_writing_request"
    CODE_REVIEW_REQUEST = "code_review_request"
    CODE_REVISION_REQUEST = "code_revision_request"
    PLANNING_REQUEST = "planning_request"

    TASK_RESPONSE = "task_response"
    CODE_WRITING_RESPONSE = "code_writing_response"
    CODE_REVIEW_RESPONSE = "code_review_response"
    CODE_REVISION_RESPONSE = "code_revision_response"
    PLANNING_RESPONSE = "planning_response"

    # System
    TIMEOUT = "timeout"

    @classmethod
    def get_type(cls, message: Union['MessageType', 'Message']) -> 'MessageType':
        if isinstance(message, MessageType):
            return message
        elif isinstance(message, Message):
            return message.type
        else:
            raise ValueError(f"Invalid input type: {type(message)}")

    @classmethod
    def is_request(cls, message: Union['MessageType', 'Message']) -> bool:
        message_type = cls.get_type(message)
        return message_type.value.endswith('_request')

    @classmethod
    def is_response(cls, message: Union['MessageType', 'Message']) -> bool:
        message_type = cls.get_type(message)
        return message_type.value.endswith('_response')

    @classmethod
    def is_system(cls, message: Union['MessageType', 'Message']) -> bool:
        return not (cls.is_request(message) or cls.is_response(message))

    @classmethod
    def get_response_type(cls, request: Union['MessageType', 'Message']) -> Optional['MessageType']:
        request_type = cls.get_type(request)
        if not cls.is_request(request_type):
            return None

        response_name = request_type.value.replace('_request', '_response')
        return next((t for t in cls if t.value == response_name), None)


@dataclass
class Message:
    role: Role
    type: MessageType
    content: str
    target_roles: Union[Role, List[Role], Tuple[Role, ...], Set[Role]]

    def __post_init__(self):
        # Convert target_roles to Set[Role] regardless of input type
        if isinstance(self.target_roles, Role):
            self.target_roles = {self.target_roles}
        elif isinstance(self.target_roles, (list, tuple)):
            self.target_roles = set(self.target_roles)
        elif not isinstance(self.target_roles, set):
            raise ValueError(
                f"target_roles must be Role, List[Role], Tuple[Role], or Set[Role], got {type(self.target_roles)}")
