import os
import uuid
import gradio as gr
from pydub import AudioSegment
from sparkai.core.messages import ChatMessage
from dwspark.config import Config
from dwspark.models import ChatModel, Audio2Text

# Configuration
SPARKAI_APP_ID = '36'
SPARKAI_API_SECRET = 'N2I'
SPARKAI_API_KEY = 'a9b7b6'
TEMP_AUDIO_DIR = "./"
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)

# Initialize models
a2t = Audio2Text(config)
model = ChatModel(config)

def process_audio_file(audio_path):
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    unique_filename = 'audio' + ".mp3"
    temp_filepath = os.path.join(TEMP_AUDIO_DIR, unique_filename)
    audio_segment.export(temp_filepath, format="mp3")
    return temp_filepath

# def process_audio(audio,history):
# # def process_audio(audio):
#     print(f"接收到的音频路径: {audio}")  # Debugging information
#     if os.path.isfile(audio):
#         audio_path = process_audio_file(audio)
#         print(audio_path)
#         audio_text = a2t.gen_text(audio_path)
#         # os.remove(audio_path)
        
#         if not audio_text.strip():
#             return "未识别到语音，请重试。", history
        
#         response = model.generate([ChatMessage(role="user", content=audio_text)])
#         print(response)

        
#         history.append((audio_text, response))
#         #print(history[0])
#         # exit()
#         return "", history 
#         # return response
#      # Make sure the return is correct
#     elif os.path.isdir(audio):
#         return "输入的是目录，请上传一个音频文件。", history
#     return "无效的音频文件，请上传有效的音频。", history

def clear_history(history):
    history.clear()
    return history

def clear_chat(chat_history):
    return clear_history(chat_history)
# os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Create Gradio interface
# with gr.Blocks() as demo:
#     #gr.Markdown("### 语音识别对话\n请录入音频或上传文件以进行对话。")
    
#     audio_input = gr.Audio(type="filepath",label="Input Audio", sources=["upload", "microphone"])
#     # audio_input = '/root/temp_audios/audio.wav'
#     chatbot = gr.Chatbot(label="聊天记录")
#     # result_network = gr.Textbox(label="搜索结果", lines=10)
#     submit_btn = gr.Button("提交")
#     clear_btn = gr.Button("清空历史")

#     # submit_btn_network = gr.Button("联网搜索")

#     submit_btn.click(process_audio, inputs=[audio_input,chatbot], outputs=[audio_input,chatbot])
#     # submit_btn_network.click(process_audio, inputs=[audio_input], outputs=[result_network])
#     clear_btn.click(clear_history, outputs=chatbot)


# # Launch the application
# demo.launch()


def process_audio(audio, history):
    print(f"接收到的音频: {audio}, 类型: {type(audio)}")  # Debugging information

    if audio is None:
        return "没有接收到音频文件，请上传一个音频文件。", history

    if isinstance(audio, str) and os.path.isfile(audio):
        audio_path = process_audio_file(audio)
        print(f"处理的音频文件路径: {audio_path}")

        try:
            audio_text = a2t.gen_text(audio_path)
            print(f"语音识别结果：{audio_text}")

            if not audio_text.strip():
                return "未识别到语音，请重试。", history
            
            response = model.generate([ChatMessage(role="user", content=audio_text)])
            print(f"生成的响应: {response}")

            # 确保历史记录更新为元组格式
            history.append((audio_text, response))
            return history  # 确保返回空字符串和更新后的历史记录

        except Exception as e:
            return f"处理音频时发生错误: {str(e)}", history

    return "无效的音频文件，请上传有效的音频。", history

# Gradio 接口定义
with gr.Blocks() as demo:
    audio_input = gr.Audio(type="filepath")
    chatbot = gr.Chatbot(label="聊天记录",type="tuples", height= 800)
    submit_btn = gr.Button("提交")
    clear_btn = gr.Button("清空历史")

    submit_btn.click(process_audio, inputs=[audio_input, chatbot], outputs=[chatbot])
    clear_btn.click(clear_chat, chatbot, chatbot)

# 启动应用
demo.launch()

