#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LINE emergency notification helper.

This uses the LINE Messaging API push endpoint. Configure credentials with:
- LINE_CHANNEL_ACCESS_TOKEN
- LINE_USER_ID
"""

import os
from datetime import datetime
from pathlib import Path

import requests


CURRENT_DIR = Path(__file__).resolve().parent
ENV_FILE = CURRENT_DIR / ".env"
PARENT_ENV_FILE = CURRENT_DIR.parent / ".env"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
TOKEN_ENV = "LINE_CHANNEL_ACCESS_TOKEN"
USER_ID_ENV = "LINE_USER_ID"
DEFAULT_TIMEOUT = 10


def load_env_file(path=ENV_FILE):
    if not path.exists() and PARENT_ENV_FILE.exists():
        path = PARENT_ENV_FILE

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def describe_response_status(response_status):
    if response_status == "help":
        return "老人明确表示需要帮助。"
    if response_status == "unknown":
        return "未能确认老人是否安全，可能无人回答或语音识别失败。"
    if response_status == "safe":
        return "老人已回答安全。"
    return "尚未进行安全确认。"


def build_fall_alert_message(response_status=None, response_text=None):
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        "[紧急通知]\n"
        "陪护机器人检测到老人可能发生跌倒。\n"
        f"时间: {now_text}\n"
        f"安全确认: {describe_response_status(response_status)}\n"
    )

    if response_text:
        message += f"识别到的回答: {response_text}\n"

    message += "请尽快联系或前往现场确认老人安全。"
    return message


def send_line_message(message, token=None, user_id=None, timeout=DEFAULT_TIMEOUT):
    load_env_file()

    token = token or os.environ.get(TOKEN_ENV)
    user_id = user_id or os.environ.get(USER_ID_ENV)

    if not token or not user_id:
        print(
            "LINE alert skipped: set LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID "
            "to enable emergency notifications."
        )
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message,
            }
        ],
    }

    try:
        response = requests.post(
            LINE_PUSH_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        print(f"LINE alert failed: {exc}")
        return False

    if response.status_code != 200:
        print(f"LINE alert failed: {response.status_code} {response.text}")
        return False

    print("LINE alert sent.")
    return True


def send_fall_alert(response_status=None, response_text=None):
    return send_line_message(build_fall_alert_message(response_status, response_text))


if __name__ == "__main__":
    send_fall_alert()
