"""
Test script untuk API TB Detector
Mengirim sample request dan mengevaluasi response
"""

import requests
import os
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_model_info():
    """Test model info endpoint"""
    print("Testing /model_info...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_prediction_with_file():
    """Test prediction endpoint dengan file upload"""
    print("Testing /predict with file upload...")
    
    # Buat dummy audio file (atau gunakan file yang ada)
    # Untuk test ini kita akan skip jika tidak ada file
    
    # Prepare form data
    data = {
        'age': 35,
        'gender': 'L',
        'has_fever': 'true',
        'has_cough': 'true',
        'cough_duration_days': 14,
        'has_night_sweats': 'true',
        'has_weight_loss': 'true',
        'has_chest_pain': 'false',
        'has_shortness_breath': 'true',
        'previous_tb_history': 'false'
    }
    
    # Cek apakah ada file audio di data/audio
    audio_dir = Path('data/audio')
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        if audio_files:
            with open(audio_files[0], 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{BASE_URL}/predict", data=data, files=files)
                print(f"Status: {response.status_code}")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print("No audio files found in data/audio")
    else:
        print("Note: No audio directory found. Create data/audio/ and add audio files for full testing.")
        print("This test requires actual cough audio files.")
    print()


def test_prediction_without_audio():
    """Test error handling saat tidak ada audio"""
    print("Testing /predict without audio (should fail)...")
    
    data = {
        'age': 35,
        'gender': 'L',
        'has_fever': 'true',
        'has_cough': 'true',
        'cough_duration_days': 14,
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_frontend():
    """Test frontend loading"""
    print("Testing frontend...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Content length: {len(response.text)} chars")
    print()


def run_all_tests():
    """Run all tests"""
    print("="*50)
    print("TB Detector API Test Suite")
    print("="*50)
    print()
    
    tests = [
        test_health,
        test_model_info,
        test_frontend,
        test_prediction_without_audio,
        test_prediction_with_file,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test failed: {e}")
            print()
    
    print("="*50)
    print("Tests completed!")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()
