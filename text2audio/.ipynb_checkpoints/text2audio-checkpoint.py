import requests
from urllib.parse import urlparse
# 请求信息
url = 'https://luvvoice.xyz/text_to_speech'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Origin': 'https://luvvoice.com',
    'Referer': 'https://luvvoice.com/'
}
def text_to_speech(text):
    data = {
        'text': text,
        'language_code': 'zh-CN-YunyangNeural'
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=data)

    # 处理响应
    if response.status_code == 200:
        json_response = response.json()
        audio_url = json_response.get('result_audio_url')
        result_text = json_response.get('result_text')
        return audio_url
        # print(f"音频URL: {audio_url}")
        # print(f"合成文本: {result_text}")
        # res = requests.get(audio_url, headers=headers)
        #
        # if audio_url:
        #     parsed_url = urlparse(audio_url)
        #     filename = parsed_url.path.split('/')[-1]  # 最后一个部分作为文件名
        #     audio_response = requests.get(audio_url)
        #     if audio_response.status_code == 200:
        #         with open(filename, 'wb') as f:
        #             f.write(audio_response.content)
        #         print(f"音频文件 {filename} 保存成功！")
        #     else:
        #         print(f"下载音频失败，状态码: {audio_response.status_code}")
        # else:
        #     print("未找到音频URL")

    else:
        print(f"请求失败，状态码: {response.status_code}")

def main():
    text = input("输入要转语音的文本:\n")
    text_to_speech(text)

if  __name__ == '__main__':
    main()
