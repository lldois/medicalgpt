# -*- coding: utf-8 -*-
"""
Chat prompt templates for different model families.
"""

from dataclasses import dataclass
from typing import Optional, List, Sequence, Dict

__all__ = ['Conversation', 'register_conv_template', 'get_conv_template']


@dataclass
class Conversation:
    name: str
    system_prompt: str
    messages: Optional[List[Sequence[str]]]
    roles: Optional[Sequence[str]]
    prompt: str
    sep: str
    stop_str: Optional[str] = "</s>"

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""
        messages = messages or self.messages
        convs = []
        if not messages:
            messages = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        self.messages.append([query, answer])


conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    conv_templates[template.name] = template


register_conv_template(
    Conversation(
        name="qwen",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT:",
        sep="</s>",
    )
)

register_conv_template(
    Conversation(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="llama3",
        system_prompt=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful, excellent and smart assistant."
        ),
        messages=[],
        roles=("user", "assistant"),
        prompt=(
            "<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        sep="<|eot_id|>",
        stop_str="<|eot_id|>",
    )
)


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name]
