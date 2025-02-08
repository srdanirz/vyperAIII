# agents/vision_agent.py

import logging
import asyncio
import base64
import io
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import aiohttp
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from deepface import DeepFace
import pytesseract
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .base_agent import BaseAgent
from core.llm import get_llm

logger = logging.getLogger(__name__)

class VisionAgent(BaseAgent):
    """
    Agente especializado en procesamiento de imágenes y video con soporte
    para múltiples modelos y capacidades avanzadas.
    
    Capacidades:
    - Análisis de imágenes (objetos, escenas, texto, rostros)
    - Procesamiento de video
    - OCR multilenguaje
    - Generación y edición de imágenes
    - Análisis facial y emocional
    - Segmentación y clasificación
    """
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__("vision_processing", api_key, metadata, shared_data)
        self.engine_mode = engine_mode
        
        # Inicializar modelos según el motor seleccionado
        self.llm = get_llm(engine_mode, api_key, model="gpt-4-vision-preview" if engine_mode == "openai" else "deepseek-vision")
        
        # Modelos locales para tareas específicas
        self.object_detector = None
        self.face_analyzer = None
        self.ocr_processor = None
        
        # Cache y estado
        self.results_cache = {}
        self.processing_state = {}
        
        # Inicializar modelos bajo demanda
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Inicializa modelos locales bajo demanda."""
        try:
            # Modelo de detección de objetos de Hugging Face
            self.object_detector = AutoModelForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                revision="no_timm"
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                revision="no_timm"
            )
            
            # Configurar OCR
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    async def process_image(
        self,
        image_path: Union[str, Path],
        tasks: List[str]
    ) -> Dict[str, Any]:
        """
        Procesa una imagen realizando múltiples tareas.
        
        Args:
            image_path: Ruta a la imagen
            tasks: Lista de tareas a realizar (e.g., ["object_detection", "ocr", "face_analysis"])
        
        Returns:
            Dict con resultados de cada tarea
        """
        try:
            # Validar formato
            path = Path(image_path)
            if path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image format: {path.suffix}")
            
            # Cargar imagen
            image = Image.open(path)
            image_array = np.array(image)
            
            # Procesar tareas en paralelo
            tasks_coroutines = []
            for task in tasks:
                if task == "object_detection":
                    tasks_coroutines.append(self._detect_objects(image))
                elif task == "ocr":
                    tasks_coroutines.append(self._perform_ocr(image_array))
                elif task == "face_analysis":
                    tasks_coroutines.append(self._analyze_faces(image_array))
                elif task == "scene_understanding":
                    tasks_coroutines.append(self._understand_scene(image))
                else:
                    logger.warning(f"Unsupported task: {task}")
            
            results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
            
            # Procesar resultados
            return {
                "timestamp": datetime.now().isoformat(),
                "results": {
                    task: result if not isinstance(result, Exception) else str(result)
                    for task, result in zip(tasks, results)
                },
                "metadata": {
                    "image_size": image.size,
                    "format": image.format,
                    "mode": image.mode
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    async def analyze_video(
        self,
        video_path: Union[str, Path],
        analysis_type: str = "full",
        frame_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Analiza un video frame por frame.
        
        Args:
            video_path: Ruta al video
            analysis_type: Tipo de análisis ('full', 'keyframes', 'motion')
            frame_interval: Intervalo entre frames a analizar
            
        Returns:
            Dict con resultados del análisis
        """
        try:
            path = Path(video_path)
            if path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
                raise ValueError(f"Unsupported video format: {path.suffix}")
            
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            frames_analysis = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Convertir BGR a RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_image = Image.fromarray(frame_rgb)
                    
                    # Analizar frame
                    analysis = await self._analyze_frame(
                        frame_image,
                        analysis_type
                    )
                    frames_analysis.append({
                        "frame_number": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                        "analysis": analysis
                    })
                
                frame_count += 1
            
            cap.release()
            
            return {
                "video_metadata": {
                    "duration": frame_count / cap.get(cv2.CAP_PROP_FPS),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "total_frames": frame_count,
                    "resolution": (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    )
                },
                "analysis_type": analysis_type,
                "frames_analyzed": len(frames_analysis),
                "frame_interval": frame_interval,
                "results": frames_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise

    async def generate_image(
        self,
        prompt: str,
        size: Tuple[int, int] = (1024, 1024),
        style: str = "natural",
        num_images: int = 1
    ) -> Dict[str, Any]:
        """
        Genera imágenes basadas en un prompt.
        
        Args:
            prompt: Descripción de la imagen a generar
            size: Tamaño de la imagen (ancho, alto)
            style: Estilo de la imagen
            num_images: Número de imágenes a generar
            
        Returns:
            Dict con URLs o datos de las imágenes generadas
        """
        try:
            if self.engine_mode == "openai":
                return await self._generate_dalle(prompt, size, num_images)
            else:
                return await self._generate_deepseek(prompt, size, num_images)
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

    async def _detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Detecta objetos en una imagen usando el modelo local."""
        try:
            # Preparar imagen para el modelo
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.object_detector(**inputs)
            
            # Procesar resultados
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.9
            )[0]
            
            detections = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                detections.append({
                    "label": self.object_detector.config.id2label[label.item()],
                    "confidence": score.item(),
                    "box": box.tolist()
                })
            
            return {
                "num_detections": len(detections),
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            raise

    async def _perform_ocr(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Realiza OCR en una imagen."""
        try:
            # Convertir a escala de grises para mejor OCR
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Aplicar umbral adaptativo
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Realizar OCR
            text = pytesseract.image_to_string(thresh)
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            # Estructurar resultados
            boxes = []
            for i, word in enumerate(data['text']):
                if word.strip():
                    boxes.append({
                        'text': word,
                        'conf': data['conf'][i],
                        'box': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        }
                    })
            
            return {
                "full_text": text.strip(),
                "words": boxes,
                "language": pytesseract.image_to_osd(thresh)
            }
            
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            raise

    async def _analyze_faces(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analiza rostros en una imagen usando DeepFace."""
        try:
            results = DeepFace.analyze(
                image_array,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            
            if not isinstance(results, list):
                results = [results]
            
            faces = []
            for face in results:
                faces.append({
                    "age": face.get('age'),
                    "gender": face.get('gender'),
                    "dominant_race": face.get('dominant_race'),
                    "emotion": face.get('dominant_emotion'),
                    "region": face.get('region')
                })
            
            return {
                "num_faces": len(faces),
                "faces": faces
            }
            
        except Exception as e:
            logger.error(f"Error analyzing faces: {e}")
            raise

    async def _understand_scene(self, image: Image.Image) -> Dict[str, Any]:
        """Analiza y describe la escena usando el LLM."""
        try:
            # Convertir imagen a base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            messages = [
                {
                    "role": "system",
                    "content": "Analyze the image and provide a detailed description including:"
                    "\n- Main elements and their arrangement"
                    "\n- Colors and lighting"
                    "\n- Mood and atmosphere"
                    "\n- Notable details or patterns"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                    ]
                }
            ]
            
            response = await self.llm.agenerate([messages])
            description = response.generations[0][0].message.content
            
            # Estructurar la descripción
            return {
                "description": description,
                "analysis_type": "scene_understanding",
                "model_used": self.engine_mode
            }
            
        except Exception as e:
            logger.error(f"Error understanding scene: {e}")
            raise

    async def _analyze_frame(
        self,
        frame: Image.Image,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Analiza un frame de video según el tipo de análisis."""
        try:
            results = {}
            
            if analysis_type in ['full', 'objects']:
                results['objects'] = await self._detect_objects(frame)
                
            if analysis_type in ['full', 'scene']:
                results['scene'] = await self._understand_scene(frame)
                
            if analysis_type in ['full', 'faces']:
                frame_array = np.array(frame)
                results['faces'] = await self._analyze_faces(frame_array)
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            raise

    async def _generate_dalle(
        self,
        prompt: str,
        size: Tuple[int, int],
        num_images: int
    ) -> Dict[str, Any]:
        """
        Genera imágenes usando DALL-E 3.
        
        Args:
            prompt: Descripción de la imagen
            size: Dimensiones de la imagen
            num_images: Número de imágenes a generar
            
        Returns:
            Dict con URLs y metadatos de las imágenes generadas
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "prompt": prompt,
                        "n": num_images,
                        "size": f"{size[0]}x{size[1]}",
                        "model": "dall-e-3",
                        "quality": "standard",
                        "response_format": "url"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise ValueError(f"DALL-E API error: {error_data}")
                        
                    data = await response.json()
                    
                    return {
                        "engine": "dalle-3",
                        "images": [
                            {
                                "url": img["url"],
                                "prompt": prompt,
                                "size": size,
                                "created": datetime.now().isoformat()
                            }
                            for img in data["data"]
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt.split()),
                            "total_tokens": len(prompt.split()) * num_images
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Error generating DALL-E image: {e}")
            raise

    async def _generate_deepseek(
        self,
        prompt: str,
        size: Tuple[int, int],
        num_images: int
    ) -> Dict[str, Any]:
        """
        Genera imágenes usando DeepSeek Image Generator.
        
        Args:
            prompt: Descripción de la imagen
            size: Dimensiones de la imagen
            num_images: Número de imágenes a generar
            
        Returns:
            Dict con URLs y metadatos de las imágenes generadas
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "prompt": prompt,
                        "n": num_images,
                        "size": f"{size[0]}x{size[1]}",
                        "response_format": "url"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise ValueError(f"DeepSeek API error: {error_data}")
                        
                    data = await response.json()
                    
                    return {
                        "engine": "deepseek",
                        "images": [
                            {
                                "url": img["url"],
                                "prompt": prompt,
                                "size": size,
                                "created": datetime.now().isoformat()
                            }
                            for img in data["data"]
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt.split()),
                            "total_tokens": len(prompt.split()) * num_images
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Error generating DeepSeek image: {e}")
            raise

    async def edit_image(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        prompt: str = "",
        size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Edita una imagen existente usando máscaras y prompts.
        
        Args:
            image_path: Ruta a la imagen original
            mask_path: Ruta opcional a la máscara
            prompt: Descripción de los cambios deseados
            size: Nuevo tamaño opcional
            
        Returns:
            Dict con la imagen editada y metadatos
        """
        try:
            # Cargar y validar imagen original
            image = Image.open(image_path)
            if size:
                image = image.resize(size)
            
            # Cargar máscara si existe
            mask = None
            if mask_path:
                mask = Image.open(mask_path)
                if size:
                    mask = mask.resize(size)
            
            # Convertir imágenes a base64
            image_b64 = self._image_to_base64(image)
            mask_b64 = self._image_to_base64(mask) if mask else None
            
            # Usar la API correspondiente
            if self.engine_mode == "openai":
                return await self._edit_dalle(image_b64, mask_b64, prompt)
            else:
                return await self._edit_deepseek(image_b64, mask_b64, prompt)
                
        except Exception as e:
            logger.error(f"Error editing image: {e}")
            raise

    def _image_to_base64(self, image: Optional[Image.Image]) -> Optional[str]:
        """Convierte una imagen PIL a base64."""
        if not image:
            return None
            
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def _edit_dalle(
        self,
        image_b64: str,
        mask_b64: Optional[str],
        prompt: str
    ) -> Dict[str, Any]:
        """Edita una imagen usando DALL-E."""
        try:
            payload = {
                "image": image_b64,
                "prompt": prompt,
                "model": "dall-e-3",
                "n": 1
            }
            if mask_b64:
                payload["mask"] = mask_b64
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise ValueError(f"DALL-E edit error: {error_data}")
                        
                    data = await response.json()
                    
                    return {
                        "engine": "dalle-3",
                        "edited_image": data["data"][0]["url"],
                        "prompt": prompt,
                        "created": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Error editing with DALL-E: {e}")
            raise

    async def _edit_deepseek(
        self,
        image_b64: str,
        mask_b64: Optional[str],
        prompt: str
    ) -> Dict[str, Any]:
        """Edita una imagen usando DeepSeek."""
        try:
            payload = {
                "image": image_b64,
                "prompt": prompt,
                "n": 1
            }
            if mask_b64:
                payload["mask"] = mask_b64
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/images/edits",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise ValueError(f"DeepSeek edit error: {error_data}")
                        
                    data = await response.json()
                    
                    return {
                        "engine": "deepseek",
                        "edited_image": data["data"][0]["url"],
                        "prompt": prompt,
                        "created": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Error editing with DeepSeek: {e}")
            raise

    async def cleanup(self) -> None:
        """Limpia recursos y libera memoria."""
        try:
            # Limpiar caché
            self.results_cache.clear()
            
            # Liberar modelos si existen
            if self.object_detector:
                del self.object_detector
            if self.face_analyzer:
                del self.face_analyzer
                
            # Limpiar estado
            self.processing_state.clear()
            
            logger.info("Vision Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del agente."""
        return {
            "agent_type": "VisionAgent",
            "engine_mode": self.engine_mode,
            "models_loaded": {
                "object_detector": self.object_detector is not None,
                "face_analyzer": self.face_analyzer is not None,
                "ocr_processor": self.ocr_processor is not None
            },
            "cache_size": len(self.results_cache),
            "processing_state": self.processing_state
        }