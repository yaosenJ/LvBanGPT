import requests
import time
import os


# api_key = os.environ.get("x_api_key")
api_key = "283b1d46-33e7-47fc-8cee-2a50f9ab38f7"

def submit_lipsync_task(video_url, audio_url, model="wav2lip++", synergize=True):
    url = "https://api.synclabs.so/lipsync"

    payload = {
        "model": model,
        "videoUrl": video_url,
        "audioUrl": audio_url,
        "synergize": synergize
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        result = response.json()
        return result.get("id")
    else:
        print("提交失败:", response.text)
        return None


def check_lipsync_task(task_id):
    url = f"https://api.synclabs.so/lipsync/{task_id}"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        # print("未能检测到任务,请勿退出...", response.text)
        # print("未能检测到任务,请勿退出...")

        return None


def download_video(video_url,video_path='./demo.mp4'):
    response = requests.get(video_url)
    if response.status_code == 200:
        # filename = video_url.split('/')[-1]
        with open(video_path, 'wb') as file:
            file.write(response.content)
        print(f"视频保存为 : {video_path}")
    else:
        print("下载视频失败:", response.status_code)


def main():
    #输入的基础视频1秒即可,文件链接需为公共可见的地址
    video_url = "https://synchlabs-public.s3.us-west-2.amazonaws.com//lip-sync-video-input/expressionSet-942c4cee-be63-4173-ba08-9409b708092c.mp4"
    audio_url = "https://pub-8a6c901f26754c4bbd4f79e70e61d104.r2.dev/luvvoice.com-20240714-o1vd.mp3"

    task_id = submit_lipsync_task(video_url, audio_url)
    # task_id = '79b98102-cdd5-4448-95ab-6af84b7582bf'
    if task_id:
        print(f"任务提交成功, ID: {task_id}")

        # 等待任务完成，轮询状态
        while True:
            task_result = check_lipsync_task(task_id)
            if task_result and task_result.get("status") == "COMPLETED":
                print("任务处理成功:", task_result)
                download_video(task_result["videoUrl"])
                break
            elif task_result and task_result.get("status") == "FAILED":
                print("任务处理失败:", task_result)
                break
            else:
                print("任务正在处理,8秒后再次检查...")
                time.sleep(8)
    else:
        print("任务提交失败!")

#
# if __name__ == "__main__":
#     main()
