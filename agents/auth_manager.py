import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import keyring
import jwt
import aiohttp

logger = logging.getLogger(__name__)

class AuthenticationManager:
    """
    Maneja la autenticación y el manejo de sesiones para múltiples servicios.
    Encripta credenciales y renueva tokens cuando es necesario.
    """
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.sessions = {}
        self.refresh_tasks = {}
        self.service_configs = {}
        self.auth_cache = {}
        self.session_validators: Dict[str, Callable[[Dict[str, Any]], Coroutine[Any, Any, bool]]] = {}

    async def initialize(self, service_configs: Dict[str, Any]) -> None:
        """Inicializa con configuraciones específicas de cada servicio."""
        self.service_configs = service_configs
        asyncio.create_task(self._monitor_sessions())

    async def get_session(self, service: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retorna una sesión activa para un servicio. Si no existe o expiró,
        la renueva o la crea.
        """
        try:
            if service in self.sessions and await self._is_session_valid(service):
                return self.sessions[service]

            if service in self.sessions:
                refreshed_session = await self._refresh_session(service)
                if refreshed_session:
                    return refreshed_session

            if credentials:
                return await self._create_session(service, credentials)
            elif service in self.auth_cache:
                return await self._create_session(service, self._decrypt_credentials(service))
            else:
                raise ValueError(f"No credentials available for {service}")

        except Exception as e:
            logger.error(f"Error getting session for {service}: {e}", exc_info=True)
            raise

    async def store_credentials(self, service: str, credentials: Dict[str, Any]) -> None:
        """Securely store credentials for a service"""
        try:
            encrypted_creds = self._encrypt_credentials(credentials)
            keyring.set_password(
                "service_integration",
                f"{service}_credentials",
                encrypted_creds.decode()
            )
            self.auth_cache[service] = encrypted_creds

        except Exception as e:
            logger.error(f"Error storing credentials for {service}: {e}")
            raise

    async def remove_credentials(self, service: str) -> None:
        """Remove stored credentials for a service"""
        try:
            keyring.delete_password("service_integration", f"{service}_credentials")
            self.auth_cache.pop(service, None)
            await self.invalidate_session(service)
        except Exception as e:
            logger.error(f"Error removing credentials for {service}: {e}")
            raise

    async def invalidate_session(self, service: str) -> None:
        """Invalidate and remove a service session"""
        try:
            if service in self.sessions:
                session = self.sessions.pop(service)
                if session.get("cleanup_callback"):
                    await session["cleanup_callback"]()

            if service in self.refresh_tasks:
                self.refresh_tasks[service].cancel()

        except Exception as e:
            logger.error(f"Error invalidating session for {service}: {e}")

    async def register_session_validator(self, service: str, validator: Callable[[Dict[str, Any]], Coroutine[Any, Any, bool]]) -> None:
        """Register a custom session validator for a service"""
        self.session_validators[service] = validator

    async def _create_session(self, service: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session for a service"""
        try:
            config = self.service_configs.get(service, {})
            auth_type = config.get("auth_type", "oauth2")

            session = None
            if auth_type == "oauth2":
                session = await self._create_oauth2_session(service, credentials)
            elif auth_type == "api_key":
                session = await self._create_api_key_session(service, credentials)
            elif auth_type == "jwt":
                session = await self._create_jwt_session(service, credentials)
            elif auth_type == "basic":
                session = await self._create_basic_auth_session(service, credentials)
            else:
                raise ValueError(f"Unsupported auth type: {auth_type}")

            if session:
                session["created_at"] = datetime.now()
                session["service"] = service
                self.sessions[service] = session

                if "refresh_token" in session:
                    self._schedule_refresh(service)

            return session

        except Exception as e:
            logger.error(f"Error creating session for {service}: {e}")
            raise

    async def _create_oauth2_session(self, service: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Create an OAuth2 session"""
        config = self.service_configs[service]
        token_url = config["token_url"]

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data={
                "grant_type": "password",
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
                "username": credentials["username"],
                "password": credentials["password"]
            }) as response:
                if response.status != 200:
                    raise Exception(f"OAuth2 authentication failed: {await response.text()}")

                token_data = await response.json()
                return {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_at": datetime.now() + timedelta(seconds=token_data["expires_in"]),
                    "token_type": token_data["token_type"]
                }

    async def _create_api_key_session(self, service: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Create an API key session"""
        return {
            "api_key": credentials["api_key"],
            "expires_at": datetime.now() + timedelta(days=30)  # Default expiration
        }

    async def _create_jwt_session(self, service: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JWT session"""
        config = self.service_configs[service]
        claims = {
            "sub": credentials["subject"],
            "iss": config.get("issuer"),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(claims, credentials["private_key"], algorithm="RS256")
        return {
            "token": token,
            "expires_at": claims["exp"]
        }

    async def _create_basic_auth_session(self, service: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic auth session"""
        return {
            "username": credentials["username"],
            "password": credentials["password"],
            "expires_at": datetime.now() + timedelta(hours=24)
        }

    async def _refresh_session(self, service: str) -> Optional[Dict[str, Any]]:
        """Refresh an existing session"""
        try:
            current_session = self.sessions[service]
            if "refresh_token" not in current_session:
                return None

            config = self.service_configs[service]
            token_url = config["token_url"]

            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data={
                    "grant_type": "refresh_token",
                    "refresh_token": current_session["refresh_token"],
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"]
                }) as response:
                    if response.status != 200:
                        return None

                    token_data = await response.json()
                    new_session = {
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token", current_session["refresh_token"]),
                        "expires_at": datetime.now() + timedelta(seconds=token_data["expires_in"]),
                        "token_type": token_data["token_type"]
                    }

                    self.sessions[service] = new_session
                    return new_session

        except Exception as e:
            logger.error(f"Error refreshing session for {service}: {e}")
            return None

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> bytes:
        """Encrypt credentials for storage"""
        try:
            credentials_json = json.dumps(credentials)
            return self.cipher_suite.encrypt(credentials_json.encode())
        except Exception as e:
            logger.error(f"Error encrypting credentials: {e}")
            raise

    def _decrypt_credentials(self, service: str) -> Dict[str, Any]:
        """Decrypt stored credentials"""
        try:
            encrypted_creds = self.auth_cache[service]
            decrypted_json = self.cipher_suite.decrypt(encrypted_creds)
            return json.loads(decrypted_json)
        except Exception as e:
            logger.error(f"Error decrypting credentials: {e}")
            raise

    async def _is_session_valid(self, service: str) -> bool:
        """Check if a session is still valid"""
        try:
            session = self.sessions[service]
            
            # Check custom validator if exists
            if service in self.session_validators:
                is_valid = await self.session_validators[service](session)
                if not is_valid:
                    return False

            # Check expiration
            if "expires_at" in session:
                return datetime.now() < session["expires_at"]

            return False

        except Exception as e:
            logger.error(f"Error checking session validity for {service}: {e}")
            return False

    def _schedule_refresh(self, service: str) -> None:
        """Schedule a session refresh"""
        session = self.sessions[service]
        if "expires_at" not in session:
            return

        refresh_at = session["expires_at"] - timedelta(minutes=5)
        delay = (refresh_at - datetime.now()).total_seconds()
        
        if delay > 0:
            self.refresh_tasks[service] = asyncio.create_task(self._delayed_refresh(service, delay))

    async def _delayed_refresh(self, service: str, delay: float) -> None:
        """Execute a delayed session refresh"""
        await asyncio.sleep(delay)
        await self._refresh_session(service)

    async def _monitor_sessions(self) -> None:
        """Monitor active sessions and handle refreshes/cleanup"""
        while True:
            try:
                for service in list(self.sessions.keys()):
                    if not await self._is_session_valid(service):
                        if not await self._refresh_session(service):
                            await self.invalidate_session(service)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in session monitor: {e}")
                await asyncio.sleep(60)
