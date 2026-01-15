import json
import os

import requests
from dotenv import load_dotenv

from app.utils.logging import logger

# ==================== Main Function (Original) ====================


def load_config() -> None:
    """
    Load configuration from Vault injected JSON file.

    Vault Agent automatically injects secrets as JSON files at:
    /vault/secrets/configuration.{PYTHON_ENVIRONMENT}.json

    For local development without Vault, gracefully skip if file doesn't exist.
    """
    service_env = os.getenv("PYTHON_ENVIRONMENT", "development").lower()
    vault_config_path = f"/vault/secrets/configuration.{service_env}.json"

    logger.info(f"Loading configuration from Vault file: {vault_config_path}")
    _load_from_vault_file(vault_config_path)


def _load_from_vault_file(vault_config_path: str) -> None:
    """
    Load secrets from Vault injected JSON file and set environment variables.

    Args:
        vault_config_path: Path to the Vault configuration JSON file
    """
    if not os.path.exists(vault_config_path):
        logger.warning(f"⚠ Vault configuration file not found: {vault_config_path}")
        logger.info("💡 For local development, ensure .env file exists in project root")
        logger.info("💡 For Docker deployment, ensure Vault Agent injects configuration")
        return

    try:
        with open(vault_config_path) as config_file:
            config_data = json.load(config_file)

        if not isinstance(config_data, dict):
            logger.error("✗ Vault configuration is not a dictionary")
            return

        # Log all config keys (without values for security)
        secret_keys = list(config_data.keys())
        logger.info(f"✓ Loaded {len(secret_keys)} configuration keys from Vault")

        # Set environment variables from config
        for key, value in config_data.items():
            os.environ[key] = str(value)
            logger.debug(f"Set environment variable: {key}")

        logger.info(f"✓ Successfully loaded configuration from Vault ({vault_config_path})")

    except json.JSONDecodeError as e:
        logger.error(f"✗ Failed to parse Vault configuration JSON: {str(e)}")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration from Vault: {str(e)}")


# ==================== NEW VERSION 2: API-based (Direct Vault API) ====================


def load_config_from_api_v2() -> None:
    """
    [V2] Load configuration from Vault API.

    Connects to Vault server using:
    - VAULT_ADDR: Vault server address (e.g., http://vault:8200)
    - VAULT_TOKEN: Vault authentication token
    - PYTHON_ENVIRONMENT: Environment name for secret path

    For local development without Vault, gracefully skip if unavailable.
    """
    print("🔐 [V2] Loading configuration from Vault API...")
    load_dotenv()  # Load environment variables from .env file
    vault_addr = os.getenv("VAULT_ADDR")
    logger.info(f"[V2] VAULT_ADDR: {vault_addr}")
    vault_token = os.getenv("VAULT_TOKEN")
    service_env = os.getenv("PYTHON_ENVIRONMENT", "development").lower()
    logger.info(f"[V2] PYTHON_ENVIRONMENT: {service_env}")

    if not vault_addr or not vault_token:
        logger.warning("[V2] VAULT_ADDR or VAULT_TOKEN not set, skipping Vault API connection")
        print("⚠ [V2] Vault credentials not found, skipping loading from Vault.")
        logger.info("💡 For local development, ensure .env file exists in project root")
        logger.info("💡 For Docker deployment, ensure VAULT_ADDR and VAULT_TOKEN are set")
        return

    _load_from_vault_api_v2(vault_addr, vault_token, service_env)


def _load_from_vault_api_v2(vault_addr: str, vault_token: str, service_env: str) -> None:
    """
    [V2] Load secrets from Vault API and set environment variables.

    Args:
        vault_addr: Vault server address (e.g., http://vault:8200)
        vault_token: Vault authentication token
        service_env: Environment name (e.g., development, staging, production)
    """
    secret_path = f"kv/data/Meobeo/{service_env}"
    vault_url = f"{vault_addr.rstrip('/')}/v1/{secret_path}"

    try:
        headers = {
            "X-Vault-Token": vault_token,
            "Content-Type": "application/json",
        }

        logger.info(f"[V2] Requesting secrets from Vault API: {vault_url}")
        response = requests.get(vault_url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        print("✓ [V2] Successfully retrieved configuration from Vault API.")

        # Extract secrets from Vault KV v2 response format
        if "data" not in data or "data" not in data["data"]:
            logger.error("✗ [V2] Unexpected Vault API response format")
            return

        config_data = data["data"]["data"]

        if not isinstance(config_data, dict):
            logger.error("✗ [V2] Vault configuration is not a dictionary")
            return

        # Log all config keys (without values for security)
        secret_keys = list(config_data.keys())
        logger.info(f"✓ [V2] Loaded {len(secret_keys)} configuration keys from Vault")

        # Set environment variables from config
        for key, value in config_data.items():
            os.environ[key] = str(value)
            logger.debug(f"[V2] Set environment variable: {key}")
            print(f"✓ [V2] Set environment variable: {key}")

        logger.info(f"✓ [V2] Successfully loaded configuration from Vault API ({secret_path})")

    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ [V2] Failed to connect to Vault API: {str(e)}")
        print(f"✗ [V2] Failed to connect to Vault API at {vault_addr}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ [V2] Vault API request failed with status {response.status_code}: {str(e)}")
        print(f"✗ [V2] Vault API returned error: {response.status_code}")
    except requests.exceptions.Timeout as e:
        logger.error(f"✗ [V2] Vault API request timed out: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"✗ [V2] Failed to parse Vault API response: {str(e)}")
    except Exception as e:
        logger.error(f"✗ [V2] Failed to load configuration from Vault API: {str(e)}")
