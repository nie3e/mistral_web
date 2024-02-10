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


def load_model(choice: str) -> str:
    """Loads a new model based on user selection.
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


def unload_model() -> None:
    """Unloads the currently loaded model.
    If no model is loaded, no action is taken.
    """
    global chat
    if not chat.model:
        gr.Info("No model loaded")
        return
    chat.unload_model()


def main() -> int:
    """Entry point function for the web app."""
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    model_text = gr.Label(
                        label="model",
                        value=lambda: chat.current_conf or "No model loaded"
                    )
                    model_type = gr.Dropdown(
                        label="Available models",
                        choices=list(model_args.keys()),
                        value="float16"
                    )
                    with gr.Row():
                        load_model_btn = gr.Button(value="Load model",
                                                   variant="primary")
                        unload_model_btn = gr.Button(value="Unload model")
                    load_model_btn.click(
                        load_model,
                        inputs=[model_type],
                        outputs=[model_text]
                    )
                    unload_model_btn.click(unload_model)
            with gr.Column(scale=7):
                ci = gr.ChatInterface(
                    chat.stream_msg_history,
                )
                ci.chatbot.height = 700
    demo.launch(server_name="0.0.0.0")

    return 0


if __name__ == "__main__":
    exit(main())
