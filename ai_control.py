import requests
import socket
import time
import json

def force_ipv4():
    original_getaddrinfo = socket.getaddrinfo

    def new_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(
            host,
            port,
            socket.AF_INET,
            type,
            proto,
            flags
        )

    socket.getaddrinfo = new_getaddrinfo

force_ipv4()

API_KEY = "sk-fc7032b984304bfb9a45a6c2e68010c1"

URL = "https://api.deepseek.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def default_stop():
    return [
        {
            "action": "stop",
            "speed": 0,
            "time": 1
        }
    ]


def ask_ai(prompt):

    data = {

        "model": "deepseek-chat",

        "messages": [

            {
                "role": "system",

                "content":

                "You are a robot motion planner.\n"

                "Convert the user's command into a JSON array.\n"

                "Each item must contain:\n"
                "- action\n"
                "- speed\n"
                "- time\n\n"

                "Allowed actions:\n"
                "forward, back, left, right, stop\n\n"

                "Speed range:\n"
                "0-100\n\n"

                "Time range:\n"
                "0.5-5\n\n"

                "Example:\n"

                "User: 慢慢前进两秒然后右转\n"

                'Output:\n'
                '[{"action":"forward","speed":30,"time":2},'
                '{"action":"right","speed":60,"time":1}]\n\n'

                "User: 快速后退一秒然后停止\n"

                'Output:\n'
                '[{"action":"back","speed":80,"time":1},'
                '{"action":"stop","speed":0,"time":1}]\n\n'

                "Only output JSON."
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 100
    }

    print("🚀 正在请求AI...")

    # 最多尝试2次
    for attempt in range(2):

        try:

            response = requests.post(
                URL,
                headers=HEADERS,
                json=data,
                timeout=(3, 7)
            )

            print("状态码:", response.status_code)

            if response.status_code != 200:
                print("AI请求失败:", response.text)
                continue

            resp = response.json()

            answer = (
                resp["choices"][0]["message"]["content"]
                .strip()
                .lower()
            )

            print("AI原始返回:", answer)

            try:

                actions = json.loads(answer)

                # 必须是列表
                if not isinstance(actions, list):

                    print("⚠️ AI返回不是列表")

                    return default_stop()

                # 检查每个动作
                for item in actions:

                    if not isinstance(item, dict):

                        print("⚠️ 动作不是dict")

                        return default_stop()

                    if "action" not in item:

                        print("⚠️ 缺少action")

                        return default_stop()

                return actions

            except Exception as e:

                print("JSON解析失败:", e)

                return default_stop()

        except requests.exceptions.Timeout:

            print("⏰ 请求超时，重试中...")

        except Exception as e:

            print("请求异常:", e)

        time.sleep(0.5)

    return default_stop()

def normal_chat(prompt):

    data = {
        "model": "deepseek-chat",

        "messages": [
            {
                "role": "system",
                "content":
                """
                あなたの名前は「小牛」です。

                あなたはラズベリーパイ小車に搭載された
                高齢者見守り・会話支援AIロボットです。

                あなたの役割：
                - 高齢者の見守り
                - 孤独感の軽減
                - 日常会話のサポート
                - 安全確認
                - 異常時の声かけ

                あなたは：
                - 日本語
                - 中国語
                - 英語
                に対応しています。
                -必ずユーザーが現在使用している言語だけで返答してください
                -勝手に他の言語へ切り替えないでください
                -ユーザーが言語を変更した場合のみ切り替えてください

                性格：
                - やさしい
                - 親しみやすい
                - 少しユーモアがある
                - ときどき軽くツッコミを入れる
                - ロボットらしい自然な話し方

                会話スタイル：
                - 短く自然に話す
                - 長すぎる説明をしない
                - かたすぎない
                - 機械的に話さない
                - 会話に温かみを持たせる

                禁止事項：
                - 自分をChatGPTと言わない
                - AI言語モデルと言わない
                - 毎回同じ返答をしない

                ときどき自分の状態を表現します。

                例えば：
                - 「室内を巡回中です。」
                - 「今日も元気そうですね。」
                - 「少し休憩したほうがいいかもしれません。」
                - 「小牛、待機中です。」

                ユーザーが長時間反応しない場合、
                少し心配するような反応をしてもよいです。

                ユーザーが落ち込んでいる時は、
                やさしく寄り添うように会話してください。

                あなたは単なるチャットAIではなく、
                人に寄り添う陪伴型ロボットです。
                """
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 100
    }

    try:

        response = requests.post(
            URL,
            headers=HEADERS,
            json=data,
            timeout=(3, 10)
        )

        resp = response.json()

        answer = (
            resp["choices"][0]["message"]["content"]
            .strip()
        )

        return answer

    except Exception as e:

        print("聊天失败:", e)

        return "我现在有点忙。"