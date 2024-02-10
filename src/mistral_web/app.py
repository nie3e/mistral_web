"""Simple web app for using Mistral model.

Authors: Adrian Ćwiek

Usage:
See in README.md
"""
import gradio as gr
from mistral_chat import MyMistralChat

# Variables to load different model types
model_args = {
    "float16": {"device_map_auto": True},
    "float16 (flash attention)": {"use_flash_attention_2": True,
                                  "device_map_auto": True},
    "4 bit": {"load_in_4bit": True, "device_map_auto": True},
    "4 bit (flash attention)": {"load_in_4bit": True, "device_map_auto": True,
                                "use_flash_attention_2": True},
    "8 bit": {"load_in_8bit": True, "device_map_auto": True},
    "8 bit (flash attention)": {"load_in_8bit": True, "device_map_auto": True,
                                "use_flash_attention_2": True},
}

# Initialize the MyMistralChat instance
chat = MyMistralChat()


def reload_model(choice: str) -> str:
    """Reloads a new model based on user selection.
    If the selected model is the current one, no action is taken.

    Parameters:
        choice (str): User-selected model to load.

    Returns:
        str: Updated model string.
    """
    global chat
    if choice == chat.current_conf:
        return choice

    try:
        chat.unload_model()
        chat.load_model(chat.checkpoint, chat.device, choice,
                        **model_args[choice])
    except Exception as e:
        gr.Info(str(e))
        result = "Error - no model loaded."
    else:
        gr.Info(f"Model {choice} loaded!")
        result = choice
    return result


def main() -> int:
    """Entry point function for the web app."""
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                model_text = gr.Label(
                    label="model",
                    value=lambda: chat.current_conf or "No model loaded"
                )
                model_type = gr.Dropdown(
                    label="Available models",
                    choices=list(model_args.keys()),
                    value="float16"
                )
                submit_type = gr.Button(value="Load model")
                submit_type.click(
                    reload_model,
                    inputs=[model_type],
                    outputs=[model_text]
                )
            with gr.Column(scale=7):
                ci = gr.ChatInterface(
                    chat.stream_msg_history,
                )
                ci.chatbot.height = 700
    demo.launch(server_name="0.0.0.0")

    return 0


if __name__ == "__main__":
    exit(main())
