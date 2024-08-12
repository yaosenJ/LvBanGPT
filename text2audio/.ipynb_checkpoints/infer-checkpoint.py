from text2audio import *

def text2audio_url(text):

    audio_url = text_to_speech(text)
    return audio_url
def a2lip(audio_url,video_path='./demo.mp4'):
    #输入的基础视频1秒即可,文件链接需为公共可见的地址
    video_url = "https://synchlabs-public.s3.us-west-2.amazonaws.com//lip-sync-video-input/expressionSet-942c4cee-be63-4173-ba08-9409b708092c.mp4"
    audio_url = audio_url

    task_id = submit_lipsync_task(video_url, audio_url)
    # task_id = '79b98102-cdd5-4448-95ab-6af84b7582bf'
    if task_id:
        print(f"任务提交成功, ID: {task_id}")

        # 等待任务完成，轮询状态
        while True:
            task_result = check_lipsync_task(task_id)
            if task_result and task_result.get("status") == "COMPLETED":
                print("任务处理成功:", task_result)
                download_video(task_result["videoUrl"],video_path)
                break
            elif task_result and task_result.get("status") == "FAILED":
                print("任务处理失败:", task_result)
                break
            else:
                print("任务正在处理,8秒后再次检查...")
                time.sleep(8)
    else:
        print("任务提交失败!")

def audio2lip(text, video_path):

    audio_url = text2audio_url(text)
    print(audio_url)  # 音频URL  # 请将此URL 转为您的音频播放器播放
    a2lip(audio_url,video_path)
    return video_path
