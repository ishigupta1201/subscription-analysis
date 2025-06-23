# client/config_manager.py

import json
import os
import sys

CONFIG_FILE_NAME = "config.json"
DEFAULT_SERVER_URL = "https://subscription-analysis-production.up.railway.app"

class ConfigManager:
    """Handles loading, saving, and prompting for user-specific configuration."""

    def __init__(self):
        # Place config.json in the same directory as this script.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, CONFIG_FILE_NAME)
        self.config = {}

    def _load_config(self) -> bool:
        """Tries to load config from config.json. Returns True on success."""
        if not os.path.exists(self.config_path):
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            required_keys = ["GOOGLE_API_KEY", "API_KEY_1", "SUBSCRIPTION_API_URL"]
            if all(key in self.config and self.config[key] for key in required_keys):
                print(f"✅ Configuration loaded successfully from {self.config_path}")
                return True
            else:
                print("⚠️ Config file is missing required keys. Re-running setup...")
                return False
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not read or parse config file. Re-running setup. Error: {e}")
            return False

    def _save_config(self):
        """Saves the current config dictionary to the file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"✅ Configuration saved to {self.config_path}")
        except IOError as e:
            print(f"❌ CRITICAL ERROR: Could not write config file to {self.config_path}. Error: {e}")
            print("   Please check your folder permissions.")
            sys.exit(1)

    def _prompt_for_config(self):
        """Runs the interactive first-time setup wizard for the user."""
        print("\n--- First-Time Setup: Subscription Analytics Client ---")
        print("You only need to do this once. Your keys will be saved locally.")
        
        try:
            # 1. Get Google API Key
            google_key = input("➡️ Enter your Google AI API Key (from ai.google.dev, starts with 'AIza...'): ").strip()
            if not google_key:
                print("❌ Error: Google API Key cannot be empty. Exiting.")
                sys.exit(1)
            self.config["GOOGLE_API_KEY"] = google_key
            
            # 2. Get Subscription Analytics API Key
            app_key = input("➡️ Enter the Subscription Analytics API Key (from the project owner, starts with 'sub_analytics...'): ").strip()
            if not app_key:
                print("❌ Error: Subscription Analytics API Key cannot be empty. Exiting.")
                sys.exit(1)
            self.config["API_KEY_1"] = app_key

            # 3. Get Server URL (with a default)
            server_url_prompt = f"➡️ Enter the Server URL [press Enter for default: {DEFAULT_SERVER_URL}]: "
            server_url = input(server_url_prompt).strip()
            self.config["SUBSCRIPTION_API_URL"] = server_url or DEFAULT_SERVER_URL

            self._save_config()

        except (KeyboardInterrupt, EOFError):
            print("\nSetup cancelled. Exiting.")
            sys.exit(1)

    def get_config(self) -> dict:
        """The main method to get configuration.
        
        It tries to load from the file first. If that fails for any reason,
        it triggers the interactive setup wizard to get the keys from the user.
        """
        if not self._load_config():
            self._prompt_for_config()
        
        return self.config