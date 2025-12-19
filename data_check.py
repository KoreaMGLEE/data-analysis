"""
JSON 파일에서 데이터 개수를 확인하는 스크립트

Usage:
    python data_check.py <json_file_path>
    python data_check.py --json_file /path/to/file.json
    
Example:
    python data_check.py /home/user3/data-analysis/easy_examples_confidence_0.8_1_5e-05.json
"""

import json
import argparse
import os
from pathlib import Path


def check_json_data(json_file_path):
    """
    JSON 파일을 읽어서 데이터 개수와 기본 정보를 출력합니다.
    
    Args:
        json_file_path: JSON 파일 경로
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return
    
    print(f"Loading JSON file: {json_file_path}")
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 데이터 타입 확인
        if isinstance(data, list):
            count = len(data)
            print(f"\nData type: List")
            print(f"Total examples: {count}")
            
            # 추가 정보 (첫 번째 항목 구조 확인)
            if count > 0:
                print(f"\nFirst example keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                if isinstance(data[0], dict) and "confidence" in data[0]:
                    confidences = [item.get("confidence", 0) for item in data if isinstance(item, dict)]
                    if confidences:
                        print(f"Confidence range: {min(confidences):.4f} ~ {max(confidences):.4f}")
                        print(f"Average confidence: {sum(confidences) / len(confidences):.4f}")
            
        elif isinstance(data, dict):
            print(f"\nData type: Dictionary")
            print(f"Dictionary keys: {list(data.keys())}")
            # 딕셔너리 안에 리스트가 있는 경우
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  '{key}': {len(value)} items")
        else:
            print(f"\nData type: {type(data)}")
            print(f"Content: {str(data)[:100]}...")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Check data count in JSON file")
    parser.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="Path to JSON file (positional argument)"
    )
    
    args = parser.parse_args()
    
    # json_file 경로 결정
    json_file_path = args.json_file
    
    if json_file_path is None:
        # 기본 경로 시도
        default_path = "/home/user3/data-analysis/easy_examples_confidence_0.8_1_5e-05.json"
        if os.path.exists(default_path):
            json_file_path = default_path
            print(f"Using default path: {default_path}")
        else:
            parser.print_help()
            print(f"\nError: No JSON file specified and default file not found.")
            return
    
    check_json_data(json_file_path)


if __name__ == "__main__":
    main()

