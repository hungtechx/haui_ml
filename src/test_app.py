#!/usr/bin/env python3
"""
Fake Reviews Detection - System Test
Test all components of the application to ensure they work correctly.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    try:
        from models.model_manager import ModelManager
        print("✅ ModelManager imported successfully")
        
        from utils.text_preprocessing import TextPreprocessor
        print("✅ Text preprocessing utilities imported successfully")
        
        from config.config import AVAILABLE_MODELS
        print("✅ Configuration imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_preprocessing():
    """Test text preprocessing functionality"""
    print("\n🧪 Testing text preprocessing...")
    try:
        from utils.text_preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        test_text = "This is a GREAT product! I love it so much!!!"
        processed_text = preprocessor.preprocess_for_bow(test_text)
        
        print(f"✅ Text preprocessing: '{test_text}' -> '{processed_text}'")
        
        # Test validation
        from utils.text_preprocessing import validate_text_input
        is_valid, error = validate_text_input(test_text)
        print(f"✅ Text validation: {is_valid} (error: {error})")
        
        # Test feature extraction
        from utils.text_preprocessing import extract_text_features
        features = extract_text_features(test_text)
        print(f"✅ Feature extraction: {len(features)} features extracted")
        
        return True
    except Exception as e:
        print(f"❌ Text preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_manager():
    """Test model manager functionality"""
    print("\n🧪 Testing model manager...")
    try:
        from models.model_manager import ModelManager
        
        model_manager = ModelManager()
        available_models = model_manager.get_available_models()
        print(f"✅ Model manager initialized, {len(available_models)} model categories available")
        
        # Test model info
        model_info = model_manager.get_model_info("Bag of Words Models", "Logistic Regression")
        print(f"✅ Model info retrieved: {model_info.get('description', 'No description')}")
        
        return True
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False

def test_model_files():
    """Test if all model files exist"""
    print("\n🧪 Testing model files...")
    try:
        model_dir = Path(__file__).parent.parent / "model_weights"
        model_files = list(model_dir.glob("*.pkl"))
        
        print(f"✅ Found {len(model_files)} model files in {model_dir}")
        for model_file in sorted(model_files):
            print(f"  📁 {model_file.name}")
        
        return len(model_files) > 0
    except Exception as e:
        print(f"❌ Model files test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n🧪 Testing prediction functionality...")
    try:
        from models.model_manager import ModelManager
        
        model_manager = ModelManager()
        test_text = "This product is absolutely amazing! I love it so much and would definitely recommend!"
        
        start_time = time.time()
        result = model_manager.predict(test_text, "Bag of Words Models", "Naive Bayes")
        prediction_time = time.time() - start_time
        
        print("✅ Prediction successful:")
        print(f"  📊 Result: {result.get('prediction_label', 'Error')}")
        print(f"  🎯 Confidence: {result.get('confidence', 0.0):.2f}%")
        print(f"  ⏱️ Time: {prediction_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FAKE REVIEWS DETECTION - SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Text Preprocessing", test_text_preprocessing),
        ("Model Manager", test_model_manager),
        ("Model Files", test_model_files),
        ("Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The application should work correctly.")
        print("💡 Run 'streamlit run app.py' to start the web interface.")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
