#!/usr/bin/env python3
"""Quick test to verify log level configuration works"""

from src.eliza_trainer.sft.run_config import load_sft_run_config
from src.eliza_trainer.common.runtime import configure_logging
import logging
import sys

def test_log_level_parsing():
    """Test that log level is parsed correctly from config"""
    
    # Test with the updated config
    config_path = "configs/sft_hass_qwen3_5_0_8b.yaml"
    
    try:
        config = load_sft_run_config(config_path)
        print(f"✅ Config loaded successfully")
        print(f"   Log level from config: {config.runtime.log_level}")
        
        # Test that logging configuration works
        configure_logging(config.runtime.log_level)
        
        # Test that debug messages will show
        logger = logging.getLogger(__name__)
        current_level = logging.getLogger().getEffectiveLevel()
        
        print(f"✅ Logging configured with level: {logging.getLevelName(current_level)}")
        
        if current_level == logging.DEBUG:
            print(f"✅ DEBUG logging is ENABLED")
            logger.debug("This is a test debug message")
        else:
            print(f"⚠️  Current level is {logging.getLevelName(current_level)}, not DEBUG")
        
        logger.info("This is a test info message")
        logger.warning("This is a test warning message")
        logger.error("This is a test error message")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_log_level_parsing()
    sys.exit(0 if success else 1)
