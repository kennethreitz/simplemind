import textwrap
from typing import Literal

from pydantic.main import BaseModel

from simplemind import generate_text

MAX_WIDTH = 80


# A member of a discussion (an LLM)
class DiscussionMember(BaseModel):
    """The member of a discussion (an LLM)"""

    provider_name: str
    provider_model: str
    nickname: str
    custom_prompt: str | None = None


# A message in a conversation
class DiscussionMessage(BaseModel):
    """A message in a conversation"""

    content: str


class BotMessage(DiscussionMessage):
    """The message sent between LLMs"""

    sender: DiscussionMember

    def __str__(self):
        return f"{self.sender.nickname}: {self.content}"


class ModeratorMessage(DiscussionMessage):
    """The message sent by the moderator"""

    visible_to: list[DiscussionMember] = []
    sendor: Literal["Moderator"] = "Moderator"

    def __str__(self):
        return f"{self.sendor}: {self.content}"


# A discussion
class Discussion:
    """Make LLMs discuss something"""

    def __init__(self, topic: str | None = None, *, verbose: bool = False):
        self.topic = topic
        self.members: list[DiscussionMember] = []
        self.conversation: list[DiscussionMessage] = []
        self.verbose = verbose

    def add_member(
        self,
        provider_name: str,
        provider_model: str,
        nickname: str | None = None,
        custom_prompt: str | None = None,
    ):
        """
        add_member Adds a member to the discussion
        Parameters
        ----------
        provider_name : str
            The name of the LLM provider
        provider_model : str
            The model name of the LLM
        nickname : str | None, optional
            The nickname of the member, by default the provider_name
        custom_prompt : str | None, optional
            The custom prompt for the member (visible only to the member), by default None
        """
        member = DiscussionMember(
            provider_name=provider_name,
            provider_model=provider_model,
            nickname=nickname or provider_name,
            custom_prompt=custom_prompt,
        )
        # make sure the nickname is unique
        assert member.nickname not in [
            m.nickname for m in self.members
        ], f"Duplicate nickname: {member.nickname}"
        self.members.append(member)
        if self.verbose:
            print(f"Added {member.nickname} to the discussion.")

    def get_members(self) -> list[DiscussionMember]:
        """Get the members of the discussion"""
        return self.members

    def set_topic(self, topic: str):
        """Set the topic of the discussion"""
        self.topic = topic

    def get_topic(self) -> str | None:
        """Get the topic of the discussion"""
        return self.topic

    def _get_history_for_member(self, member: DiscussionMember) -> str:
        """
        _get_history_for_member Get the history form the POV of the given member.
        Parameters
        ----------
        member : DiscussionMember
            The member to get the history for
        Returns
        -------
        str
            The history as seen by the member
        """
        relevant_messages: list[DiscussionMessage] = []
        for message in self.conversation:
            if isinstance(message, BotMessage):
                relevant_messages.append(message)
            elif isinstance(message, ModeratorMessage) and member in message.visible_to:
                relevant_messages.append(message)
        return "\n\n".join(map(str, relevant_messages))

    @property
    def initial_moderator_message(self) -> str:
        return f"Discuss the following topic and answer during your turn only: {self.topic}"

    def _get_response(self, member: DiscussionMember) -> BotMessage:
        """
        _get_response Returns the BotMessage from the given member
        Parameters
        ----------
        member : DiscussionMember
            The member to get the response from
        Returns
        -------
        BotMessage
            The BotMessage
        """

        history = self._get_history_for_member(member)
        prompt = f"{history}\n\n{member.nickname}: "
        content = generate_text(
            prompt=prompt,
            llm_provider=member.provider_name,
            llm_model=member.provider_model,
        )
        message = BotMessage(
            content=content,
            sender=member,
        )
        self.conversation.append(message)
        if self.verbose:
            print(message.sender.nickname)
            print("\n".join(textwrap.wrap(message.content, MAX_WIDTH)))
            print()
        return message

    def add_moderator_message(
        self, content: str, visible_to: list[DiscussionMember] | None = None
    ):
        """
        add_moderator_message adds a message to the conversation as the moderator
        Parameters
        ----------
        content : str
            The content of the message
        visible_to : list[DiscussionMember], optional
            The list of members that the message is visible to, defaults to all members
        """
        if visible_to is None:
            visible_to = self.members
        message = ModeratorMessage(
            content=content,
            visible_to=self.members,
        )
        self.conversation.append(message)

    def _initialize_discussion(self):
        """Initialize the discussion"""
        assert self.topic is not None, "Topic must be set"
        assert len(self.members) >= 2, "There must be at least 2 members"
        self.add_moderator_message(self.initial_moderator_message)

        for member in self.members:
            if member.custom_prompt is not None:
                self.add_moderator_message(member.custom_prompt, visible_to=[member])

        if self.verbose:
            print(f"Topic: {self.topic}")
            print(f"Members: {', '.join(member.nickname for member in self.members)}")

    def discuss(self, no_of_rounds: int = 1):
        """
        discuss returns the responses of the members at the end of the discussion.
        Parameters
        ----------
        no_of_rounds : int, optional
            The number of rounds, by default 1.
            Round is the number of turns each LLM gets.
        verbose : bool, optional
            Whether to print the conversation, by default False
        Returns
        -------
        list[DiscussionMessage]
            The conversation between the LLMs
        """

        self._initialize_discussion()
        for i in range(no_of_rounds):
            for member in self.members:
                try:
                    self._get_response(member)
                except Exception as e:
                    if self.verbose:
                        print(f"Error: {e}")
                    continue
            if self.verbose:
                print(f"Round {i + 1} completed.")
                print("=" * MAX_WIDTH)
        return self.conversation

    def discuss_yield(self, no_of_rounds: int = 1):
        """
        discuss yields the responses of the members during the discussion.
        Parameters
        ----------
        no_of_rounds : int, optional
            The number of rounds, by default 1.
            Round is the number of turns each LLM gets.
        verbose : bool, optional
            Whether to print the conversation, by default False
        Returns
        -------
        list[DiscussionMessage]
            The conversation between the LLMs
        """

        self._initialize_discussion()
        for i in range(no_of_rounds):
            for member in self.members:
                try:
                    message = self._get_response(member)
                    yield message
                except Exception as e:
                    if self.verbose:
                        print(f"Error: {e}")
                    continue
            if self.verbose:
                print(f"Round {i + 1} completed.")
                print("=" * MAX_WIDTH)


if __name__ == "__main__":
    discussion = Discussion(verbose=True)
    discussion.set_topic("The future of human-AI collaboration in creative fields")
    discussion.add_member(
        provider_name="openai",
        provider_model="gpt-4o-mini",
        nickname="Alice",
        custom_prompt="You are an AI expert.",
    )
    discussion.add_member(
        provider_name="openai",
        provider_model="gpt-4o",
        nickname="Bob",
        custom_prompt="You are an Artist.",
    )
    discussion.add_member(
        provider_name="ollama",
        provider_model="llama3.2",
        nickname="Charlie",
        custom_prompt="You are an Programmer.",
    )
    discussion.discuss(3)
