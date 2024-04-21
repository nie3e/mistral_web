from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, BitsAndBytesConfig)
from config import logger, get_model_checkpoint
from accelerate.utils import release_memory
import torch
import gc
from threading import Thread
from itertools import cycle


class MyMistralChat:
    """MyMistralChat is a class designed for loading different types of Mistral
     models.

    Attributes:
        device (str): Device to run the model on (CUDA or CPU).
        checkpoint (str): Path to the directory containing the model.
        model (AutoModelForCausalLM): The initialized model.
        tokenizer (AutoTokenizer): The initialized tokenizer.

    Example:
        chat = MyMistralChat()
        chat.load_model("/path/to/model", "cuda", device_map_auto=True)

        result = ""
        for txt in chat.stream_msg_history(message="Hello!", history=[]):
            result += txt
        print(result)

    """
    def __init__(
            self,
            checkpoint: str = get_model_checkpoint(),
            device: str = "cuda",
    ) -> None:
        """Initializes MyMistralChat object with a given checkpoint and device.

        Parameters:
            checkpoint (str): The path to the model checkpoint.
            device (str): CUDA or CPU.
        """
        self.device = device
        self.checkpoint = checkpoint
        self.model = None
        self.tokenizer = None
        self.current_conf = None

    def load_model(
        self,
        checkpoint: str = get_model_checkpoint(),
        device: str = "cuda",
        configuration_name: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention_2: bool = False,
        device_map_auto: bool = False
    ) -> None:
        """Loads Mistral model with given parameters.

        Parameters:
            checkpoint (str): The path to the model checkpoint.
            device (str): CUDA or CPU.
            configuration_name (str): Model configuration name.
            load_in_4bit (bool): Whether to load the model with 4-bit
                quantization
            load_in_8bit (bool): Whether to load the model with 8-bit
                quantization
            use_flash_attention_2 (bool): Whether to use flash attention
            device_map_auto (bool): Whether to automatically assign the model
                to multiple devices if possible.
        """
        self.device = device
        self.checkpoint = checkpoint

        if self.device == "cuda":
            torch.cuda.empty_cache()

        model_args = {
            "pretrained_model_name_or_path": checkpoint,
            "force_download": False,
        }
        if load_in_4bit:
            model_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            model_args["load_in_8bit"] = True
        else:
            model_args["torch_dtype"] = torch.bfloat16

        if use_flash_attention_2:
            model_args["use_flash_attention_2"] = True

        if device_map_auto:
            model_args["device_map"] = "auto"
        logger.info(f"Loading model using {model_args}")
        self.model = AutoModelForCausalLM.from_pretrained(
            **model_args
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint
        )
        if not (device_map_auto or load_in_4bit or load_in_8bit):
            self.model.to(self.device)
        self.current_conf = configuration_name
        logger.info("Model loaded!")

    def unload_model(self):
        """Unloads the loaded model and tokenizer, and frees GPU memory if the
        model was on CUDA."""
        if self.model:
            release_memory(self.model)
            self.model = None
            gc.collect()
        if self.tokenizer:
            self.tokenizer = None
            gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        self.current_conf = None

    def stream_msg_history(
            self, message: str, history: list[str], repetition_penalty: float,
            temperature: float, top_k: int, top_p: float
    ) -> str:
        """Performs inference on each message in the given message history,
        appending the response for each message to the output string.

        Parameters:
            message (str): The new message to send for inference.
            history (list): A list containing all previous messages in the
                conversation.
            # https://huggingface.co/docs/transformers/v4.40.0/en/main_classes/text_generation#transformers.GenerationConfig
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty.
            temperature (float): The value used to modulate the next token
                probabilities.
            top_k (int): The number of highest probability vocabulary tokens to
                keep for top-k-filtering.
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.

        Returns:
            str: A concatenated string of each message response in order,
                including the new message.
        """
        flat_history = [
            h for hs in history for h in hs
        ]
        messages = flat_history + [message]
        messages_dict = [
            {"role": u, "content": msg}
            for u, msg in zip(cycle(["user", "assistant"]), messages)
        ]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        encodeds = self.tokenizer.apply_chat_template(
            messages_dict, return_tensors="pt", add_generation_prompt=True
        )
        logger.info(f"Input tokens: {encodeds.shape[1]}, settings: "
                    f"{repetition_penalty=} {temperature=} {top_k=} {top_p=}")
        model_inputs = encodeds.to(self.device)
        generation_kwargs = dict(
            inputs=model_inputs, streamer=streamer, do_sample=True,
            max_new_tokens=8000, repetition_penalty=repetition_penalty,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ""
        for new_text in streamer:
            response += new_text
            yield response
