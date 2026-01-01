import json
import os

def calculate_bias_ratio(file_path):
    """
    JSONL 파일에서 hypothesis_only_bias가 1인 데이터의 비율을 계산합니다.
    """
    total_count = 0
    bias_count = 0
    
    print(f"Reading file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 빈 줄 건너뛰기
                
                try:
                    data = json.loads(line)
                    total_count += 1
                    
                    # 'hypothesis_only_bias' 키 값을 확인 (없으면 0 취급)
                    bias_value = data.get("hypothesis_only_bias", 0)
                    
                    if bias_value == 1:
                        bias_count += 1
                        
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_num} is not valid JSON. Skipped.")
                    continue

        if total_count == 0:
            print("No valid data found.")
            return 0.0

        ratio = (bias_count / total_count) * 100
        
        print("-" * 30)
        print(f"Total Examples: {total_count}")
        print(f"Biased Examples (Value 1): {bias_count}")
        print(f"Bias Ratio: {ratio:.2f}%")
        print("-" * 30)
        
        return ratio

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# --- 사용 예시 ---

# 테스트를 위한 가상의 파일 생성 (실제 사용 시에는 기존 파일 경로를 넣으세요)
dummy_filename = "./hypothesis_bias_tags.jsonl"

# 함수 실행
calculate_bias_ratio(dummy_filename)

# 테스트 파일 삭제 (필요시)
# os.remove(dummy_filename)