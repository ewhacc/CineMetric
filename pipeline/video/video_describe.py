import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2
import argparse
import os

def main():
    # 1. 인자값 설정
    parser = argparse.ArgumentParser(description="VideoLLaMA3 Inference 스크립트")
    parser.add_argument("--video_path", type=str, required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--save_path", type=str, required=True, help="결과를 저장할 디렉토리 경로")
    args = parser.parse_args()

    device = "cuda"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

    print(f"🔄 모델 로딩 중: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    response = infer(args.video_path, model, processor, device)

    
    os.makedirs(args.save_path, exist_ok=True)
    
    base_name = os.path.basename(args.video_path).split('.')[0]
    output_file = os.path.join(args.save_path, f"{base_name}_desc.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"✅ 결과가 저장되었습니다: {output_file}")

def infer(filename, model, processor, device):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release() # 리소스 해제

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": filename, "fps": fps, "max_frames": frame_count}},
                {"type": "text", "text": "describe the video"},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

if __name__ == "__main__":
    main()
