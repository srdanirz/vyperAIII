# agents/security_agent.py

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import jwt
from cryptography.fernet import Fernet
import aiohttp
import asyncio
from pathlib import Path
import re

from .base_agent import BaseAgent
from llm_factory import get_llm

logger = logging.getLogger(__name__)

class SecurityAgent(BaseAgent):
    """
    Agente especializado en seguridad y validación.
    
    Capacidades:
    - Validación de entradas y salidas
    - Detección de contenido malicioso
    - Auditoría de acciones
    - Encriptación/desencriptación
    - Control de acceso
    - Detección de anomalías
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__("security", api_key, metadata, shared_data)
        self.engine_mode = engine_mode
        
        # Inicializar componentes de seguridad
        self._initialize_security_components()
        
        # Estado y métricas
        self.security_state = {
            "threat_level": "low",
            "active_threats": [],
            "blocked_ips": set(),
            "suspicious_patterns": set()
        }
        
        # Caché de validaciones
        self.validation_cache = {}
        
        # Reglas de seguridad
        self._load_security_rules()

    def _initialize_security_components(self) -> None:
        """Inicializa componentes de seguridad."""
        try:
            # Generar clave para encriptación
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
            
            # Configurar JWT
            self.jwt_secret = hashlib.sha256(
                self.encryption_key
            ).hexdigest()
            
            # Cargar reglas de detección
            self.threat_patterns = {
                "sql_injection": [
                    r"(\b)(select|insert|update|delete|drop|union|exec)(\b)",
                    r"(/\*|\*/|;;|--)"
                ],
                "xss": [
                    r"(<script|javascript:|vbscript:|livescript:)",
                    r"(alert\(|eval\(|execScript\()"
                ],
                "path_traversal": [
                    r"(\.\.\/|\.\.\\)",
                    r"(/etc/|/var/|/usr/)"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error initializing security components: {e}")
            raise

    def _load_security_rules(self) -> None:
        """Carga reglas de seguridad desde configuración."""
        try:
            self.security_rules = {
                "input_validation": {
                    "max_length": 10000,
                    "allowed_tags": ["p", "br", "b", "i", "u"],
                    "blocked_keywords": ["exec", "eval", "script"],
                    "max_nested_depth": 5
                },
                "rate_limiting": {
                    "max_requests_per_minute": 60,
                    "max_tokens_per_request": 4000,
                    "cooldown_period": 300
                },
                "content_security": {
                    "allowed_domains": ["trusted-domain.com"],
                    "max_file_size": 10 * 1024 * 1024,  # 10MB
                    "allowed_file_types": [".txt", ".pdf", ".jpg", ".png"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading security rules: {e}")
            raise

    async def validate_request(
        self,
        request: Dict[str, Any],
        validation_type: str = "full"
    ) -> Dict[str, Any]:
        """
        Valida una solicitud entrante.
        
        Args:
            request: Solicitud a validar
            validation_type: Tipo de validación a realizar
        """
        try:
            validation_results = {
                "is_safe": True,
                "threats_detected": [],
                "validation_details": {}
            }
            
            # Validar estructura básica
            if not self._validate_structure(request):
                validation_results["is_safe"] = False
                validation_results["threats_detected"].append("invalid_structure")
                
            # Validar contenido
            content_validation = await self._validate_content(request)
            if not content_validation["is_safe"]:
                validation_results["is_safe"] = False
                validation_results["threats_detected"].extend(
                    content_validation["threats"]
                )
                
            # Validar límites y recursos
            if validation_type == "full":
                resource_validation = self._validate_resources(request)
                if not resource_validation["is_safe"]:
                    validation_results["is_safe"] = False
                    validation_results["threats_detected"].extend(
                        resource_validation["threats"]
                    )
            
            # Detectar amenazas
            threats = await self._detect_threats(request)
            if threats:
                validation_results["is_safe"] = False
                validation_results["threats_detected"].extend(threats)
            
            # Registrar resultado
            self._log_validation_result(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return {
                "is_safe": False,
                "threats_detected": ["validation_error"],
                "error": str(e)
            }

    async def encrypt_data(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Encripta datos sensibles.
        
        Args:
            data: Datos a encriptar
            context: Contexto adicional
        """
        try:
            # Convertir a bytes si es necesario
            if isinstance(data, dict):
                data = json.dumps(data).encode()
            elif isinstance(data, str):
                data = data.encode()
            
            # Encriptar
            encrypted_data = self.cipher_suite.encrypt(data)
            
            # Generar hash para verificación
            data_hash = hashlib.sha256(data).hexdigest()
            
            return {
                "encrypted_data": encrypted_data,
                "hash": data_hash,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise

    async def decrypt_data(
        self,
        encrypted_data: bytes,
        expected_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Desencripta datos.
        
        Args:
            encrypted_data: Datos encriptados
            expected_hash: Hash esperado para verificación
        """
        try:
            # Desencriptar
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Verificar hash si se proporciona
            if expected_hash:
                current_hash = hashlib.sha256(decrypted_data).hexdigest()
                if current_hash != expected_hash:
                    raise ValueError("Data integrity check failed")
            
            # Intentar convertir a JSON si es posible
            try:
                decrypted_data = json.loads(decrypted_data)
            except json.JSONDecodeError:
                pass
            
            return {
                "decrypted_data": decrypted_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

    async def generate_auth_token(
        self,
        user_data: Dict[str, Any],
        expiration: int = 3600
    ) -> Dict[str, Any]:
        """
        Genera un token JWT de autenticación.
        
        Args:
            user_data: Datos del usuario
            expiration: Tiempo de expiración en segundos
        """
        try:
            # Preparar payload
            payload = {
                **user_data,
                "exp": datetime.utcnow().timestamp() + expiration,
                "iat": datetime.utcnow().timestamp()
            }
            
            # Generar token
            token = jwt.encode(
                payload,
                self.jwt_secret,
                algorithm="HS256"
            )
            
            return {
                "token": token,
                "expiration": payload["exp"],
                "type": "Bearer"
            }
            
        except Exception as e:
            logger.error(f"Error generating auth token: {e}")
            raise

    async def validate_auth_token(self, token: str) -> Dict[str, Any]:
        """
        Valida un token JWT.
        
        Args:
            token: Token JWT a validar
        """
        try:
            # Decodificar y validar token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            # Verificar expiración
            if payload["exp"] < datetime.utcnow().timestamp():
                raise jwt.ExpiredSignatureError("Token has expired")
            
            return {
                "is_valid": True,
                "payload": payload
            }
            
        except jwt.ExpiredSignatureError:
            return {
                "is_valid": False,
                "error": "Token expired"
            }
        except jwt.InvalidTokenError as e:
            return {
                "is_valid": False,
                "error": str(e)
            }

    async def detect_anomalies(
        self,
        data: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detecta anomalías en los datos.
        
        Args:
            data: Datos a analizar
            baseline: Datos de referencia
        """
        try:
            anomalies = []
            
            # Usar modelo de lenguaje para análisis contextual
            anomaly_prompt = {
                "role": "system",
                "content": "Analyze the following data for security anomalies and suspicious patterns:"
            }
            
            analysis = await self.llm.agenerate([
                [
                    anomaly_prompt,
                    {
                        "role": "user",
                        "content": f"Data: {json.dumps(data)}\nBaseline: {json.dumps(baseline)}"
                    }
                ]
            ])
            
            # Procesar respuesta
            analysis_text = analysis.generations[0][0].message.content
            
            # Detectar patrones sospechosos
            for pattern_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, str(data), re.IGNORECASE):
                        anomalies.append({
                            "type": pattern_type,
                            "pattern": pattern,
                            "confidence": 0.9
                        })
            
            # Actualizar estado
            if anomalies:
                self.security_state["threat_level"] = "elevated"
                self.security_state["active_threats"].extend(anomalies)
            
            return {
                "anomalies_detected": bool(anomalies),
                "anomalies": anomalies,
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise

    def _validate_structure(self, request: Dict[str, Any]) -> bool:
        """Valida la estructura básica de una solicitud."""
        try:
            required_fields = {"type", "content", "metadata"}
            return all(field in request for field in required_fields)
        except Exception as e:
            logger.error(f"Error validating structure: {e}")
            return False

    async def _validate_content(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Valida el contenido de una solicitud."""
        try:
            threats = []
            rules = self.security_rules["input_validation"]
            
            # Validar longitud
            if len(str(request)) > rules["max_length"]:
                threats.append("content_too_long")
            
            # Validar palabras bloqueadas
            content_str = str(request["content"]).lower()
            for keyword in rules["blocked_keywords"]:
                if keyword in content_str:
                    threats.append(f"blocked_keyword_{keyword}")
            
            # Validar profundidad de anidamiento
            if isinstance(request["content"], dict):
                depth = self._get_nested_depth(request["content"])
                if depth > rules["max_nested_depth"]:
                    threats.append("excessive_nesting")
            
            return {
                "is_safe": len(threats) == 0,
                "threats": threats
            }
            
        except Exception as e:
            logger.error(f"Error validating content: {e}")
            return {"is_safe": False, "threats": ["validation_error"]}

    def _validate_resources(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Valida el uso de recursos."""
        try:
            threats = []
            rules = self.security_rules["rate_limiting"]
            
            # Validar tokens
            if "tokens" in request["metadata"]:
                if request["metadata"]["tokens"] > rules["max_tokens_per_request"]:
                    threats.append("token_limit_exceeded")
            
            return {
                "is_safe": len(threats) == 0,
                "threats": threats
            }
            
        except Exception as e:
            logger.error(f"Error validating resources: {e}")
            return {"is_safe": False, "threats": ["resource_validation_error"]}

    async def _detect_threats(
        self,
        request: Dict[str, Any]
    ) -> List[str]:
        """Detecta amenazas en una solicitud."""
        try:
            threats = []
            content_str = str(request)
            
            # Validar patrones de amenaza
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_str, re.IGNORECASE):
                        threats.append(threat_type)
                        # Actualizar patrones sospechosos
                        self.security_state["suspicious_patterns"].add(pattern)
            
            # Validar con LLM si se detectan amenazas
            if threats:
                confirmation = await self._confirm_threats_with_llm(
                    request,
                    threats
                )
                if confirmation["is_threat"]:
                    return confirmation["threats"]
            
            return []
            
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
            return ["threat_detection_error"]

    async def _confirm_threats_with_llm(
        self,
        request: Dict[str, Any],
        detected_threats: List[str]
    ) -> Dict[str, Any]:
        """Confirma amenazas usando el modelo de lenguaje."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a security expert. Analyze the following request and confirm if it contains the detected threats:"
                },
                {
                    "role": "user",
                    "content": f"Request: {json.dumps(request)}\nDetected threats: {detected_threats}"
                }
            ]
            
            response = await self.llm.agenerate([messages])
            analysis = response.generations[0][0].message.content
            
            # Procesar respuesta
            is_threat = "confirmed" in analysis.lower() or "threat" in analysis.lower()
            
            return {
                "is_threat": is_threat,
                "threats": detected_threats if is_threat else [],
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error confirming threats: {e}")
            return {"is_threat": True, "threats": detected_threats}

    def _get_nested_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calcula la profundidad de anidamiento de un objeto."""
        if not isinstance(obj, (dict, list)) or current_depth > 100:
            return current_depth
        
        if isinstance(obj, dict):
            return max(
                (self._get_nested_depth(v, current_depth + 1) for v in obj.values()),
                default=current_depth
            )
        
        if isinstance(obj, list):
            return max(
                (self._get_nested_depth(item, current_depth + 1) for item in obj),
                default=current_depth
            )
        
        return current_depth

    def _log_validation_result(
        self,
        result: Dict[str, Any]
    ) -> None:
        """Registra resultado de validación."""
        try:
            # Añadir a caché
            cache_key = hashlib.md5(
                json.dumps(result).encode()
            ).hexdigest()
            
            self.validation_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Limpiar caché antigua
            self._cleanup_old_cache()
            
        except Exception as e:
            logger.error(f"Error logging validation result: {e}")

    def _cleanup_old_cache(self, max_age: int = 3600) -> None:
        """Limpia entradas antiguas del caché."""
        try:
            now = datetime.now()
            old_entries = [
                k for k, v in self.validation_cache.items()
                if (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > max_age
            ]
            
            for key in old_entries:
                del self.validation_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")

    async def cleanup(self) -> None:
        """Limpia recursos y registros."""
        try:
            # Limpiar cachés
            self.validation_cache.clear()
            
            # Resetear estado
            self.security_state = {
                "threat_level": "low",
                "active_threats": [],
                "blocked_ips": set(),
                "suspicious_patterns": set()
            }
            
            logger.info("Security Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del agente."""
        return {
            "agent_type": "SecurityAgent",
            "engine_mode": self.engine_mode,
            "security_state": {
                **self.security_state,
                "blocked_ips": list(self.security_state["blocked_ips"]),
                "suspicious_patterns": list(self.security_state["suspicious_patterns"])
            },
            "cache_size": len(self.validation_cache)
        }