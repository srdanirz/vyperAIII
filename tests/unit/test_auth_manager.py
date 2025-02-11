import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import jwt
import keyring
from cryptography.fernet import Fernet

from agents.auth_manager import AuthenticationManager

@pytest.fixture
async def auth_manager():
    """Create a test instance of AuthenticationManager."""
    manager = AuthenticationManager(encryption_key=Fernet.generate_key())
    yield manager
    await manager.cleanup()

@pytest.fixture
def mock_keyring():
    with patch('keyring.set_password') as mock_set, \
         patch('keyring.get_password') as mock_get, \
         patch('keyring.delete_password') as mock_del:
        yield {
            'set': mock_set,
            'get': mock_get,
            'delete': mock_del
        }

@pytest.fixture
def test_service_config():
    return {
        "test_service": {
            "auth_type": "oauth2",
            "token_url": "https://test.com/token",
            "client_id": "test_client",
            "client_secret": "test_secret"
        }
    }

@pytest.mark.asyncio
async def test_session_creation(auth_manager, test_service_config):
    """Test creating a new session."""
    await auth_manager.initialize(test_service_config)
    
    credentials = {
        "username": "test_user",
        "password": "test_pass"
    }
    
    session = await auth_manager.get_session("test_service", credentials)
    
    assert session is not None
    assert "access_token" in session
    assert "expires_at" in session

@pytest.mark.asyncio
async def test_session_caching(auth_manager, test_service_config):
    """Test session caching and reuse."""
    await auth_manager.initialize(test_service_config)
    
    # Create initial session
    credentials = {
        "username": "test_user",
        "password": "test_pass"
    }
    
    session1 = await auth_manager.get_session("test_service", credentials)
    session2 = await auth_manager.get_session("test_service")
    
    assert session1 == session2
    assert auth_manager.sessions["test_service"] == session1

@pytest.mark.asyncio
async def test_session_expiration(auth_manager, test_service_config):
    """Test session expiration handling."""
    await auth_manager.initialize(test_service_config)
    
    # Create session that expires soon
    credentials = {
        "username": "test_user",
        "password": "test_pass"
    }
    
    session = await auth_manager.get_session("test_service", credentials)
    session["expires_at"] = datetime.now() - timedelta(minutes=5)
    auth_manager.sessions["test_service"] = session
    
    # Try to get session again
    new_session = await auth_manager.get_session("test_service")
    
    assert new_session != session
    assert new_session["expires_at"] > datetime.now()

@pytest.mark.asyncio
async def test_credential_storage(auth_manager, mock_keyring):
    """Test secure credential storage."""
    credentials = {
        "username": "test_user",
        "password": "test_pass"
    }
    
    await auth_manager.store_credentials("test_service", credentials)
    
    assert mock_keyring["set"].called
    assert "test_service" in auth_manager.auth_cache

@pytest.mark.asyncio
async def test_credential_removal(auth_manager, mock_keyring):
    """Test credential removal."""
    await auth_manager.remove_credentials("test_service")
    
    assert mock_keyring["delete"].called
    assert "test_service" not in auth_manager.auth_cache

@pytest.mark.asyncio
async def test_session_validation(auth_manager, test_service_config):
    """Test custom session validation."""
    await auth_manager.initialize(test_service_config)
    
    # Register custom validator
    async def validator(session):
        return session.get("custom_valid", False)
    
    await auth_manager.register_session_validator("test_service", validator)
    
    # Test invalid session
    session = {
        "access_token": "test",
        "expires_at": datetime.now() + timedelta(hours=1),
        "custom_valid": False
    }
    auth_manager.sessions["test_service"] = session
    
    assert not await auth_manager._is_session_valid("test_service")
    
    # Test valid session
    session["custom_valid"] = True
    assert await auth_manager._is_session_valid("test_service")

@pytest.mark.asyncio
async def test_session_refresh(auth_manager, test_service_config):
    """Test session refresh mechanism."""
    await auth_manager.initialize(test_service_config)
    
    # Create session with refresh token
    session = {
        "access_token": "old_token",
        "refresh_token": "refresh_token",
        "expires_at": datetime.now() - timedelta(minutes=5)
    }
    auth_manager.sessions["test_service"] = session
    
    # Mock refresh response
    async def mock_refresh():
        return {
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_at": datetime.now() + timedelta(hours=1)
        }
    
    with patch.object(auth_manager, '_refresh_session', side_effect=mock_refresh):
        new_session = await auth_manager.get_session("test_service")
        
        assert new_session["access_token"] == "new_token"
        assert new_session["refresh_token"] == "new_refresh"

@pytest.mark.asyncio
async def test_encryption(auth_manager):
    """Test credential encryption/decryption."""
    test_data = {
        "sensitive": "information",
        "numbers": [1, 2, 3]
    }
    
    # Encrypt
    encrypted = auth_manager._encrypt_credentials(test_data)
    assert isinstance(encrypted, bytes)
    
    # Decrypt
    decrypted = auth_manager._decrypt_credentials(encrypted)
    assert decrypted == test_data

@pytest.mark.asyncio
async def test_cleanup(auth_manager, test_service_config):
    """Test cleanup process."""
    await auth_manager.initialize(test_service_config)
    
    # Create some sessions
    credentials = {
        "username": "test_user",
        "password": "test_pass"
    }
    await auth_manager.get_session("test_service", credentials)
    
    # Perform cleanup
    await auth_manager.cleanup()
    
    assert len(auth_manager.sessions) == 0
    assert len(auth_manager.refresh_tasks) == 0

@pytest.mark.asyncio
async def test_monitoring_sessions(auth_manager, test_service_config):
    """Test session monitoring functionality."""
    await auth_manager.initialize(test_service_config)
    
    # Create expired session
    auth_manager.sessions["test_service"] = {
        "access_token": "test",
        "expires_at": datetime.now() - timedelta(hours=1)
    }
    
    # Run one monitoring cycle
    await auth_manager._monitor_sessions()
    
    # Session should be removed
    assert "test_service" not in auth_manager.sessions

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.auth_manager"])