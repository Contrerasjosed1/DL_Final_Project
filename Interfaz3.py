import gradio as gr
import tensorflow_hub as hub
import numpy as np

# Cargar el modelo Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Conjunto simple de preguntas/respuestas para simular el chatbot
faq = {
    "hello": "Hi there! How can I help you?",
    "who are you?": "I'm a chatbot trained on movie dialogs!",
    "what's your favorite movie?": "I love The Matrix. It's a classic!",
    "bye": "Goodbye! Come back soon.",
    "how are you?": "I'm just code, but thanks for asking!"
}

# Embeddings precalculados de las preguntas
questions = list(faq.keys())
responses = list(faq.values())
question_embeddings = embed(questions)

# Función para obtener la respuesta más parecida
def generar_respuesta(user_input, history):
    input_embedding = embed([user_input])
    similarities = np.inner(input_embedding, question_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    response = responses[best_match_idx]
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, ""

# Interfaz en Gradio
with gr.Blocks(css="#chatbox .message {font-size: 16px;}") as demo:
    gr.Markdown("## 🎬 Chatbot de Películas (Demo con USE)")
    chatbot = gr.Chatbot(elem_id="chatbox", label="Conversación", type="messages")
    with gr.Row():
        txt = gr.Textbox(placeholder="Escribe tu mensaje aquí...", scale=8)
        send_btn = gr.Button("Enviar", scale=1)

    send_btn.click(fn=generar_respuesta, inputs=[txt, chatbot], outputs=[chatbot, txt])
    txt.submit(fn=generar_respuesta, inputs=[txt, chatbot], outputs=[chatbot, txt])

demo.launch()