#!/usr/bin/env python3
"""
Qwen3-VL 多轮对话 Demo
- 纯文本多轮对话使用 MLCEngine
- 图片理解使用 Qwen3VLEngine
"""

import os
import sys

# 环境设置
os.environ['TVM_HOME'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['TVM_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['LD_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/lib:/root/autodl-tmp/mlc-llm/build/tvm:' + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.insert(0, '/root/autodl-tmp/mlc-llm/python')

from mlc_llm import MLCEngine
from mlc_llm.model.qwen3_vl import Qwen3VLEngine


class ChatSession:
    """多轮对话会话管理"""
    
    def __init__(self, model_path: str):
        print("=" * 60)
        print("🤖 Qwen3-VL 多轮对话 Demo")
        print("=" * 60)
        print("正在加载模型...")
        
        model_lib = os.path.join(model_path, "lib.so")
        
        # 文本引擎（多轮对话）
        print("  - 加载文本引擎 (MLCEngine)...")
        self.text_engine = MLCEngine(
            model=model_path,
            model_lib=model_lib
        )
        
        # 视觉引擎（图片理解）
        print("  - 加载视觉引擎 (Qwen3VLEngine)...")
        self.vision_engine = Qwen3VLEngine(model_path)
        
        # 对话历史
        self.messages = []
        self.system_prompt = "You are Qwen, a helpful assistant."
        
        print("✅ 模型加载完成！")
        self.print_help()
    
    def print_help(self):
        print("\n" + "=" * 60)
        print("📖 使用说明:")
        print("-" * 60)
        print("  直接输入文字      - 纯文本多轮对话")
        print("  /image <路径> <问题> - 图片理解（单次）")
        print("  /system <提示>    - 设置系统提示词")
        print("  /clear            - 清空对话历史")
        print("  /history          - 查看对话历史")
        print("  /help             - 显示帮助信息")
        print("  /quit             - 退出程序")
        print("-" * 60)
        print("💡 纯文本支持多轮对话，图片理解为单次问答")
        print("=" * 60 + "\n")
    
    def text_chat(self, user_input: str) -> str:
        """纯文本多轮对话"""
        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 构建完整消息
        full_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.messages
        
        try:
            response = self.text_engine.chat.completions.create(
                messages=full_messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.8
            )
            
            assistant_reply = response.choices[0].message.content
            
            # 记录回复
            self.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })
            
            return assistant_reply
            
        except Exception as e:
            self.messages.pop()  # 移除失败的消息
            return f"❌ 错误: {e}"
    
    def image_chat(self, image_path: str, question: str) -> str:
        """图片理解（使用视觉引擎）"""
        if not os.path.exists(image_path):
            return f"❌ 图片不存在: {image_path}"
        
        try:
            response = self.vision_engine.chat(image_path, question)
            
            # 将图片对话也记录到历史（作为上下文）
            self.messages.append({
                "role": "user",
                "content": f"[📷 图片: {os.path.basename(image_path)}] {question}"
            })
            self.messages.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        except Exception as e:
            return f"❌ 图片分析错误: {e}"
    
    def clear_history(self):
        self.messages = []
        print("✅ 对话历史已清空")
    
    def show_history(self):
        if not self.messages:
            print("📝 对话历史为空")
            return
        
        print("\n" + "=" * 60)
        print(f"📝 对话历史 ({len(self.messages)} 条):")
        print("-" * 60)
        for i, msg in enumerate(self.messages):
            role = "👤" if msg["role"] == "user" else "🤖"
            content = msg["content"]
            if len(content) > 60:
                content = content[:60] + "..."
            print(f"{i+1}. {role} {content}")
        print("=" * 60 + "\n")
    
    def run(self):
        while True:
            try:
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                # 命令处理
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=2)
                    cmd = parts[0].lower()
                    
                    if cmd in ["/quit", "/exit", "/q"]:
                        print("👋 再见！")
                        break
                    elif cmd == "/help":
                        self.print_help()
                    elif cmd == "/clear":
                        self.clear_history()
                    elif cmd == "/history":
                        self.show_history()
                    elif cmd == "/system":
                        if len(parts) > 1:
                            self.system_prompt = parts[1]
                            print(f"✅ 系统提示已设置")
                        else:
                            print(f"当前系统提示: {self.system_prompt}")
                    elif cmd == "/image":
                        if len(parts) >= 3:
                            image_path = parts[1]
                            question = parts[2]
                            print(f"\n📷 分析图片: {image_path}")
                            print(f"❓ 问题: {question}")
                            print("\n🤖 助手: ", end="", flush=True)
                            response = self.image_chat(image_path, question)
                            print(response + "\n")
                        else:
                            print("用法: /image <图片路径> <问题>")
                            print("示例: /image /root/OCR_BENCH.jpg 这张图片讲了什么？")
                    else:
                        print(f"❌ 未知命令，输入 /help 查看帮助")
                else:
                    # 纯文本对话
                    print("\n🤖 助手: ", end="", flush=True)
                    response = self.text_chat(user_input)
                    print(response + "\n")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except EOFError:
                break


def main():
    model_path = "/root/autodl-tmp/Qwen3-VL-4B-mlc"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)
    
    session = ChatSession(model_path)
    session.run()


if __name__ == "__main__":
    main()
