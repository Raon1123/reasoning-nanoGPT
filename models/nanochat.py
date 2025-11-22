from .nanogpt import GPT, GPTConfig

class NanoChatConfig(GPTConfig):
    pass

class NanoChat(GPT):
    """
    NanoChat model. 
    Essentially the same as NanoGPT but potentially with different default configs or methods 
    specific to chat (e.g. specific generation handling).
    For now, it's a direct subclass.
    """
    pass
