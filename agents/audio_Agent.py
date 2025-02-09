# agents/audio_agent.py

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import wave
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline
import torch
from pydub import AudioSegment
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .base_agent import BaseAgent
from core.llm import get_llm

logger = logging.getLogger(__name__)

class AudioAgent(BaseAgent):
    """
    Agente especializado en procesamiento de audio con capacidades avanzadas.
    
    Capacidades:
    - Transcripción de audio a texto
    - Análisis de sentimiento en audio
    - Detección de idioma
    - Síntesis de voz
    - Análisis de características musicales
    - Limpieza y mejora de audio
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__("audio_processing", api_key, metadata, shared_data)
        self.engine_mode = engine_mode
        self.llm = get_llm(engine_mode, api_key)
        
        # Inicializar modelos locales
        self._initialize_models()
        
        # Cache y estado
        self.transcription_cache = {}
        self.analysis_cache = {}
        self.processing_state = {}

    def _initialize_models(self) -> None:
        """Inicializa modelos locales para tareas específicas."""
        try:
            # Modelo de reconocimiento de voz
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model="facebook/wav2vec2-large-960h"
            )
            
            # Detector de idioma
            self.lang_detector = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-large-xlsr-53"
            )
            
            # Análisis de sentimientos en audio
            self.emotion_detector = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er"
            )
            
        except Exception as e:
            logger.error(f"Error initializing audio models: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        timestamp_granularity: str = "word"
    ) -> Dict[str, Any]:
        """
        Transcribe audio a texto con timestamps.
        
        Args:
            audio_path: Ruta al archivo de audio
            language: Código de idioma opcional
            timestamp_granularity: 'word' o 'sentence'
        """
        try:
            # Validar formato
            path = Path(audio_path)
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported audio format: {path.suffix}")
            
            # Verificar caché
            cache_key = f"{path}_{language}_{timestamp_granularity}"
            if cache_key in self.transcription_cache:
                return self.transcription_cache[cache_key]
            
            # Cargar y preprocesar audio
            audio, sr = librosa.load(path)
            
            # Usar el motor apropiado
            if self.engine_mode == "openai":
                result = await self._transcribe_whisper(audio, sr, language)
            else:
                result = await self._transcribe_deepseek(audio, sr, language)
            
            # Añadir timestamps si es necesario
            if timestamp_granularity:
                result = await self._add_timestamps(
                    result,
                    audio,
                    sr,
                    timestamp_granularity
                )
            
            # Guardar en caché
            self.transcription_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    async def analyze_audio(
        self,
        audio_path: Union[str, Path],
        analysis_types: List[str] = ["emotion", "language", "music"]
    ) -> Dict[str, Any]:
        """
        Realiza análisis completo de un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            analysis_types: Tipos de análisis a realizar
        """
        try:
            path = Path(audio_path)
            audio, sr = librosa.load(path)
            
            results = {}
            tasks = []
            
            for analysis_type in analysis_types:
                if analysis_type == "emotion":
                    tasks.append(self._analyze_emotion(audio))
                elif analysis_type == "language":
                    tasks.append(self._detect_language(audio))
                elif analysis_type == "music":
                    tasks.append(self._analyze_music(audio, sr))
                    
            # Ejecutar análisis en paralelo
            analysis_results = await asyncio.gather(*tasks)
            
            # Combinar resultados
            for analysis_type, result in zip(analysis_types, analysis_results):
                results[analysis_type] = result
            
            return {
                "timestamp": datetime.now().isoformat(),
                "audio_metadata": {
                    "duration": len(audio) / sr,
                    "sample_rate": sr,
                    "format": path.suffix
                },
                "analysis_results": results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            raise

    async def generate_speech(
        self,
        text: str,
        voice: str = "neutral",
        language: str = "en",
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Genera audio a partir de texto (text-to-speech).
        
        Args:
            text: Texto a convertir en audio
            voice: Tipo de voz a usar
            language: Código de idioma
            output_path: Ruta para guardar el audio
        """
        try:
            if self.engine_mode == "openai":
                result = await self._generate_speech_openai(text, voice, language)
            else:
                result = await self._generate_speech_deepseek(text, voice, language)
            
            # Guardar audio si se especifica ruta
            if output_path:
                output_path = Path(output_path)
                with open(output_path, 'wb') as f:
                    f.write(result['audio_data'])
                
                result['file_path'] = str(output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise

    async def clean_audio(
        self,
        audio_path: Union[str, Path],
        methods: List[str] = ["noise", "normalization", "echo"]
    ) -> Dict[str, Any]:
        """
        Limpia y mejora la calidad del audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            methods: Métodos de limpieza a aplicar
        """
        try:
            audio, sr = librosa.load(audio_path)
            cleaned_audio = audio.copy()
            
            processing_steps = []
            
            if "noise" in methods:
                cleaned_audio = await self._remove_noise(cleaned_audio)
                processing_steps.append("noise_reduction")
                
            if "normalization" in methods:
                cleaned_audio = await self._normalize_audio(cleaned_audio)
                processing_steps.append("normalization")
                
            if "echo" in methods:
                cleaned_audio = await self._remove_echo(cleaned_audio)
                processing_steps.append("echo_reduction")
            
            # Generar archivo de salida
            output_path = Path(audio_path).with_stem(f"{Path(audio_path).stem}_cleaned")
            sf.write(output_path, cleaned_audio, sr)
            
            return {
                "original_path": str(audio_path),
                "cleaned_path": str(output_path),
                "processing_steps": processing_steps,
                "improvement_metrics": await self._calculate_improvement(audio, cleaned_audio)
            }
            
        except Exception as e:
            logger.error(f"Error cleaning audio: {e}")
            raise

    async def _transcribe_whisper(
        self,
        audio: np.ndarray,
        sr: int,
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Transcribe usando Whisper de OpenAI."""
        try:
            # Guardar audio temporal para Whisper
            temp_path = Path("temp_whisper.wav")
            sf.write(temp_path, audio, sr)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={
                        "file": open(temp_path, "rb"),
                        "model": "whisper-1",
                        "language": language if language else "auto",
                        "response_format": "verbose_json"
                    }
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Whisper API error: {await response.text()}")
                    
                    result = await response.json()
                    
            # Limpiar archivo temporal
            temp_path.unlink()
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "model": "whisper-1"
            }
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            raise

    async def _transcribe_deepseek(
        self,
        audio: np.ndarray,
        sr: int,
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Transcribe usando DeepSeek."""
        try:
            # Similar a Whisper pero con la API de DeepSeek
            temp_path = Path("temp_deepseek.wav")
            sf.write(temp_path, audio, sr)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={
                        "file": open(temp_path, "rb"),
                        "language": language if language else "auto",
                        "response_format": "verbose_json"
                    }
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"DeepSeek API error: {await response.text()}")
                    
                    result = await response.json()
                    
            temp_path.unlink()
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "model": "deepseek-audio"
            }
            
        except Exception as e:
            logger.error(f"Error in DeepSeek transcription: {e}")
            raise

    async def _analyze_emotion(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analiza emociones en el audio."""
        try:
            results = self.emotion_detector(audio)
            
            return {
                "primary_emotion": results[0]["label"],
                "confidence": results[0]["score"],
                "all_emotions": [
                    {"emotion": r["label"], "score": r["score"]}
                    for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            raise

    async def _detect_language(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detecta el idioma del audio."""
        try:
            results = self.lang_detector(audio)
            
            return {
                "detected_language": results[0]["label"],
                "confidence": results[0]["score"],
                "alternatives": [
                    {"language": r["label"], "score": r["score"]}
                    for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            raise

    async def _analyze_music(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """Analiza características musicales."""
        try:
            # Extraer características
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)
            
            return {
                "tempo": float(tempo),
                "key_signature": self._estimate_key(chroma),
                "spectral_features": {
                    "mean_mfcc": float(np.mean(mfcc)),
                    "std_mfcc": float(np.std(mfcc))
                },
                "audio_quality": {
                    "snr": float(self._calculate_snr(audio)),
                    "dynamic_range": float(self._calculate_dynamic_range(audio))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing music: {e}")
            raise

    def _estimate_key(self, chroma: np.ndarray) -> str:
        """Estima la tonalidad musical."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_sum = np.sum(chroma, axis=1)
        key_idx = np.argmax(chroma_sum)
        return notes[key_idx]

    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calcula la relación señal-ruido."""
        noise = audio - np.mean(audio)
        return 10 * np.log10(np.sum(audio**2) / np.sum(noise**2))

    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calcula el rango dinámico."""
        return float(np.max(audio) - np.min(audio))

    async def cleanup(self) -> None:
        """Limpia recursos y memoria."""
        try:
            # Limpiar cachés
            self.transcription_cache.clear()
            self.analysis_cache.clear()
            
            # Liberar modelos
            if hasattr(self, 'asr_model'):
                del self.asr_model
            if hasattr(self, 'lang_detector'):
                del self.lang_detector
            if hasattr(self, 'emotion_detector'):
                del self.emotion_detector
            
            # Limpiar estado
            self.processing_state.clear()
            
            # Limpiar archivos temporales
            for temp_file in Path('.').glob('temp_*.wav'):
                temp_file.unlink(missing_ok=True)
                
            logger.info("Audio Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del agente."""
        return {
            "agent_type": "AudioAgent",
            "engine_mode": self.engine_mode,
            "models_loaded": {
                "asr_model": hasattr(self, 'asr_model'),
                "lang_detector": hasattr(self, 'lang_detector'),
                "emotion_detector": hasattr(self, 'emotion_detector')
            },
            "cache_sizes": {
                "transcription": len(self.transcription_cache),
                "analysis": len(self.analysis_cache)
            },
            "processing_state": self.processing_state
        }

    async def _add_timestamps(
        self,
        transcription: Dict[str, Any],
        audio: np.ndarray,
        sr: int,
        granularity: str
    ) -> Dict[str, Any]:
        """Añade timestamps precisos al texto transcrito."""
        try:
            # Detectar silencios para segmentación
            non_silent = librosa.effects.split(
                audio,
                top_db=20,
                frame_length=2048,
                hop_length=512
            )
            
            # Convertir a timestamps
            segments = []
            text = transcription["text"]
            words = text.split()
            
            if granularity == "word":
                # Asignar timestamps por palabra
                for i, (start, end) in enumerate(non_silent):
                    if i < len(words):
                        segments.append({
                            "word": words[i],
                            "start": float(start) / sr,
                            "end": float(end) / sr
                        })
            else:
                # Asignar timestamps por frase
                sentences = text.split(". ")
                sentence_boundaries = [0] + [
                    i for i, char in enumerate(text)
                    if char == "."
                ] + [len(text)]
                
                current_segment = 0
                for i in range(len(sentences)):
                    start_idx = sentence_boundaries[i]
                    end_idx = sentence_boundaries[i + 1]
                    
                    # Encontrar segmentos de audio correspondientes
                    while current_segment < len(non_silent):
                        if current_segment + 1 < len(non_silent):
                            segments.append({
                                "sentence": sentences[i],
                                "start": float(non_silent[current_segment][0]) / sr,
                                "end": float(non_silent[current_segment + 1][0]) / sr
                            })
                            current_segment += 2
                            break
                        current_segment += 1
            
            transcription["segments"] = segments
            return transcription
            
        except Exception as e:
            logger.error(f"Error adding timestamps: {e}")
            raise

    async def _remove_noise(self, audio: np.ndarray) -> np.ndarray:
        """Elimina ruido del audio usando filtrado espectral."""
        try:
            # Estimar ruido de fondo
            noise_sample = audio[:int(len(audio) * 0.1)]  # Usar primeros 10%
            noise_spectrum = np.mean(np.abs(librosa.stft(noise_sample)) ** 2, axis=1)
            
            # Aplicar filtrado espectral
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Sustracción espectral
            cleaned_magnitude = np.maximum(
                magnitude ** 2 - noise_spectrum[:, np.newaxis],
                0
            ) ** 0.5
            
            # Reconstruir señal
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft)
            
            return cleaned_audio
            
        except Exception as e:
            logger.error(f"Error removing noise: {e}")
            raise

    async def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normaliza el volumen del audio."""
        try:
            return librosa.util.normalize(audio)
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            raise

    async def _remove_echo(self, audio: np.ndarray) -> np.ndarray:
        """Reduce el eco usando deconvolución."""
        try:
            # Estimar la respuesta del impulso
            ir_length = int(len(audio) * 0.1)  # 10% de la longitud
            ir = np.zeros(ir_length)
            ir[0] = 1
            
            # Aplicar deconvolución
            cleaned_audio = signal.deconvolve(audio, ir)[0]
            return cleaned_audio
            
        except Exception as e:
            logger.error(f"Error removing echo: {e}")
            raise

    async def _calculate_improvement(
        self,
        original: np.ndarray,
        cleaned: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas de mejora."""
        try:
            return {
                "snr_improvement": self._calculate_snr(cleaned) - self._calculate_snr(original),
                "dynamic_range_improvement": self._calculate_dynamic_range(cleaned) - self._calculate_dynamic_range(original),
                "clarity_score": float(np.correlate(original, cleaned)[0])
            }
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")
            raise