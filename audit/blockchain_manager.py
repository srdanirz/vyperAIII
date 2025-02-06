# audit/blockchain_manager.py

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import json
from pathlib import Path
import aiohttp
import asyncio
from eth_account import Account
from web3 import Web3
from web3.middleware import geth_poa_middleware

logger = logging.getLogger(__name__)

class BlockchainManager:
    """
    Gestor de integración con blockchain para auditoría.
    
    Características:
    - Registro inmutable de decisiones
    - Verificación de integridad
    - Smart contracts para auditoría
    - Prueba de ejecución
    - Trazabilidad completa
    """
    
    def __init__(self):
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar conexión Web3
        self.w3 = self._initialize_web3()
        
        # Cargar contratos
        self.contracts = self._load_contracts()
        
        # Estado y métricas
        self.state = {
            "connected": False,
            "last_block": None,
            "pending_transactions": {}
        }
        
        # Cache de transacciones
        self.transaction_cache = {}
        
        # Iniciar monitoreo
        self._start_monitoring()

    def _load_config(self) -> Dict[str, Any]:
        """Carga configuración blockchain."""
        try:
            config_path = Path(__file__).parent / "blockchain_config.yaml"
            if not config_path.exists():
                return self._get_default_config()
            
            with open(config_path) as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto."""
        return {
            "network": {
                "type": "ethereum",
                "network": "mainnet",
                "provider_url": "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"
            },
            "contracts": {
                "audit": {
                    "address": "0x...",
                    "abi_path": "contracts/audit.json"
                }
            },
            "gas": {
                "limit": 200000,
                "price_strategy": "medium"
            },
            "retry": {
                "max_attempts": 3,
                "delay": 1
            }
        }

    def _initialize_web3(self) -> Web3:
        """Inicializa conexión Web3."""
        try:
            # Crear proveedor
            if self.config["network"]["type"] == "ethereum":
                provider_url = self.config["network"]["provider_url"]
                w3 = Web3(Web3.HTTPProvider(provider_url))
            else:
                raise ValueError(f"Unsupported network type: {self.config['network']['type']}")
            
            # Añadir middleware si es necesario
            if self.config["network"]["network"] != "mainnet":
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Verificar conexión
            if not w3.is_connected():
                raise ConnectionError("Failed to connect to blockchain network")
            
            return w3
            
        except Exception as e:
            logger.error(f"Error initializing Web3: {e}")
            raise

    def _load_contracts(self) -> Dict[str, Any]:
        """Carga contratos inteligentes."""
        try:
            contracts = {}
            
            for name, contract_config in self.config["contracts"].items():
                # Cargar ABI
                abi_path = Path(contract_config["abi_path"])
                if not abi_path.exists():
                    logger.warning(f"Contract ABI not found: {name}")
                    continue
                    
                with open(abi_path) as f:
                    abi = json.load(f)
                
                # Crear contrato
                contract = self.w3.eth.contract(
                    address=contract_config["address"],
                    abi=abi
                )
                
                contracts[name] = contract
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error loading contracts: {e}")
            return {}

    def _start_monitoring(self) -> None:
        """Inicia monitoreo de blockchain."""
        asyncio.create_task(self._monitor_blockchain())
        asyncio.create_task(self._monitor_pending_transactions())

    async def record_action(
        self,
        action_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Registra una acción en blockchain.
        
        Args:
            action_type: Tipo de acción
            data: Datos a registrar
            metadata: Metadata adicional
        """
        try:
            # Preparar datos
            record = {
                "type": action_type,
                "data": data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Calcular hash
            record_hash = self._calculate_hash(record)
            
            # Preparar transacción
            contract = self.contracts["audit"]
            
            # Construir transacción
            transaction = await self._build_transaction(
                contract.functions.recordAction(
                    record_hash,
                    json.dumps(record)
                )
            )
            
            # Enviar transacción
            tx_hash = await self._send_transaction(transaction)
            
            # Registrar en caché
            self.transaction_cache[tx_hash] = {
                "record": record,
                "hash": record_hash,
                "status": "pending"
            }
            
            return {
                "transaction_hash": tx_hash,
                "record_hash": record_hash,
                "timestamp": record["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error recording action: {e}")
            raise

    async def verify_record(
        self,
        record_hash: str
    ) -> Dict[str, Any]:
        """
        Verifica un registro en blockchain.
        
        Args:
            record_hash: Hash del registro
        """
        try:
            # Obtener registro
            contract = self.contracts["audit"]
            record_data = await contract.functions.getRecord(record_hash).call()
            
            if not record_data:
                return {
                    "verified": False,
                    "error": "Record not found"
                }
            
            # Verificar hash
            stored_hash = record_data[0]
            if stored_hash != record_hash:
                return {
                    "verified": False,
                    "error": "Hash mismatch"
                }
            
            # Decodificar datos
            record = json.loads(record_data[1])
            
            return {
                "verified": True,
                "record": record,
                "blockchain_timestamp": record_data[2]
            }
            
        except Exception as e:
            logger.error(f"Error verifying record: {e}")
            raise

    async def get_audit_trail(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene trail de auditoría.
        
        Args:
            start_time: Tiempo inicial
            end_time: Tiempo final
            action_type: Tipo de acción opcional
        """
        try:
            contract = self.contracts["audit"]
            
            # Obtener eventos
            events = await contract.events.ActionRecorded.get_logs(
                fromBlock=self.state["last_block"] - 1000  # Últimos 1000 bloques
            )
            
            # Filtrar y procesar eventos
            audit_trail = []
            
            for event in events:
                record = json.loads(event["args"]["record"])
                timestamp = datetime.fromisoformat(record["timestamp"])
                
                # Aplicar filtros
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                if action_type and record["type"] != action_type:
                    continue
                
                audit_trail.append({
                    "record": record,
                    "block_number": event["blockNumber"],
                    "transaction_hash": event["transactionHash"].hex(),
                    "verified": True
                })
            
            return sorted(
                audit_trail,
                key=lambda x: datetime.fromisoformat(x["record"]["timestamp"])
            )
            
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            raise

    async def _build_transaction(
        self,
        contract_function: Any
    ) -> Dict[str, Any]:
        """Construye una transacción."""
        try:
            # Obtener cuenta
            account = Account.from_key(self.config["private_key"])
            
            # Obtener nonce
            nonce = await self.w3.eth.get_transaction_count(
                account.address,
                'pending'
            )
            
            # Estimar gas
            gas_estimate = await contract_function.estimate_gas({
                'from': account.address
            })
            
            # Obtener precio de gas
            gas_price = await self._get_gas_price()
            
            return {
                'nonce': nonce,
                'gasPrice': gas_price,
                'gas': min(
                    gas_estimate * 12 // 10,  # 20% extra
                    self.config["gas"]["limit"]
                ),
                'to': contract_function.address,
                'data': contract_function._encode_transaction_data(),
                'chainId': await self.w3.eth.chain_id
            }
            
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise

    async def _send_transaction(
        self,
        transaction: Dict[str, Any]
    ) -> str:
        """Envía una transacción."""
        try:
            # Firmar transacción
            account = Account.from_key(self.config["private_key"])
            signed_tx = account.sign_transaction(transaction)
            
            # Enviar transacción
            tx_hash = await self.w3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Registrar transacción pendiente
            self.state["pending_transactions"][tx_hash.hex()] = {
                "status": "pending",
                "timestamp": datetime.now().isoformat()
            }
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            raise

    async def _monitor_blockchain(self) -> None:
        """Monitorea eventos en blockchain."""
        while True:
            try:
                # Obtener último bloque
                current_block = await self.w3.eth.block_number
                
                if self.state["last_block"] != current_block:
                    # Procesar nuevo bloque
                    await self._process_new_block(current_block)
                    self.state["last_block"] = current_block
                
                await asyncio.sleep(15)  # Esperar 15 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring blockchain: {e}")
                await asyncio.sleep(15)

    async def _monitor_pending_transactions(self) -> None:
        """Monitorea transacciones pendientes."""
        while True:
            try:
                for tx_hash, tx_info in list(self.state["pending_transactions"].items()):
                    # Verificar estado
                    receipt = await self.w3.eth.get_transaction_receipt(tx_hash)
                    
                    if receipt:
                        # Actualizar estado
                        if receipt["status"] == 1:
                            await self._handle_successful_transaction(tx_hash, receipt)
                        else:
                            await self._handle_failed_transaction(tx_hash, receipt)
                
                await asyncio.sleep(30)  # Esperar 30 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring transactions: {e}")
                await asyncio.sleep(30)

    async def _process_new_block(self, block_number: int) -> None:
        """Procesa un nuevo bloque."""
        try:
            # Obtener eventos
            contract = self.contracts["audit"]
            events = await contract.events.ActionRecorded.get_logs(
                fromBlock=block_number,
                toBlock=block_number
            )
            
            # Procesar eventos
            for event in events:
                await self._handle_action_event(event)
                
        except Exception as e:
            logger.error(f"Error processing block {block_number}: {e}")

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calcula hash de datos."""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            raise

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Esperar transacciones pendientes
            pending_count = len(self.state["pending_transactions"])
            if pending_count > 0:
                logger.warning(f"Waiting for {pending_count} pending transactions")
                await asyncio.sleep(60)  # Esperar 1 minuto máximo
            
            # Limpiar estado
            self.state["connected"] = False
            self.state["pending_transactions"].clear()
            self.transaction_cache.clear()
            
            logger.info("Blockchain Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        return {
            "connected": self.state["connected"],
            "last_block": self.state["last_block"],
            "pending_transactions": len(self.state["pending_transactions"]),
            "cached_transactions": len(self.transaction_cache)
        }