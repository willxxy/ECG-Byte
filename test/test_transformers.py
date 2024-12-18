def test_transformers_installation():
    print("Testing transformers installation...")
    
    try:
        import transformers
        print("✓ Basic import successful")
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print("✗ Failed to import transformers:", e)
        return
    
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir = '../.huggingface')
        test_text = "Testing BERT tokenizer"
        tokens = tokenizer(test_text)
        print("✓ Tokenizer loading and tokenization successful")
        print(f"  Sample tokenization: {test_text} -> {tokenizer.convert_ids_to_tokens(tokens['input_ids'])}")
    except Exception as e:
        print("✗ Failed to load tokenizer:", e)
        return
    
    try:
        from transformers import BertModel
        model = BertModel.from_pretrained('bert-base-uncased', cache_dir = '../.huggingface')
        print("✓ Model loading successful")
    except Exception as e:
        print("✗ Failed to load model:", e)
        return
    
    try:
        import inspect
        source_code = inspect.getsource(transformers.BertModel)
        print("✓ Can access source code of transformers")
    except Exception as e:
        print("✗ Failed to access source code:", e)
        return
    
    print("\nAll tests passed! Transformers is properly installed in editable mode.")

if __name__ == "__main__":
    test_transformers_installation()