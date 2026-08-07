import os
import time
import torch
from glob import glob
from tqdm import tqdm
from loguru import logger

from .config import WhisperConfig, Paths
from .audio_loader import AudioLoader
from .whisper_engine import IntrospectionEngine
from .pooling import PoolingEngine
from .data_packer import DataPacker
from .visualizer import Visualizer

class Phase1Pipeline:
    def __init__(self):
        self.config = WhisperConfig()
        self.paths = Paths()
        self.audio_loader = AudioLoader(self.config.sample_rate)
        self.engine = IntrospectionEngine(self.config)
        self.pooling_engine = PoolingEngine(embed_dim=1280)
        
        os.makedirs(self.paths.results_dir, exist_ok=True)

    def run_batch(self):
        audio_files = glob(os.path.join(self.paths.raw_audio_dir, "*.wav"))
        if not audio_files:
            logger.error(f"No .wav files found in {self.paths.raw_audio_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files. Commencing batch processing...")

        for file_path in tqdm(audio_files, desc="Processing Audio Files"):
            file_name = os.path.basename(file_path).split('.')[0]
            output_dir = os.path.join(self.paths.results_dir, file_name)
            
            # Resume capability: Skip if completed metadata exists
            meta_path = os.path.join(output_dir, "metadata.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, "r") as f:
                    try:
                        if json.load(f).get("status") == "completed":
                            logger.info(f"Skipping {file_name}: Already completed.")
                            continue
                    except Exception:
                        pass

            logger.info(f"Processing: {file_name}")
            
            # Reset peak GPU memory tracking
            torch.cuda.reset_peak_memory_stats(self.config.device)
            start_time = time.time()
            
            try:
                # 1. Load Audio
                audio_array = self.audio_loader.load(file_path)
                duration = len(audio_array) / self.config.sample_rate
                
                # 2. Whisper Forward Pass & Tensor Introspection
                result_dict = self.engine.process(audio_array)
                
                # 3. Pooling Experiments
                pooling_results = self.pooling_engine.compute_all_pools(
                    result_dict["final_encoder_state"]
                )
                
                # 4. Measure Performance Metrics
                latency = time.time() - start_time
                peak_vram_mb = torch.cuda.max_memory_allocated(self.config.device) / (1024 ** 2)
                
                metrics = {
                    "file_name": file_name,
                    "audio_duration_seconds": duration,
                    "inference_latency_seconds": latency,
                    "peak_vram_mb": peak_vram_mb,
                    "real_time_factor": latency / duration if duration > 0 else 0
                }
                
                # 5. Save Structured Outputs
                dirs = DataPacker.save_results(
                    output_dir, 
                    result_dict, 
                    metrics, 
                    pooling_results, 
                    self.config
                )
                
                # 6. Generate Exploratory Visualizations
                Visualizer.generate_plots(
                    dirs["visualizations"], 
                    result_dict["clean_tokens_data"], 
                    result_dict["final_encoder_state"],
                    pooling_results
                )
                
                logger.success(f"Successfully processed and archived: {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_name}. Error: {str(e)}")
                raise e

if __name__ == "__main__":
    pipeline = Phase1Pipeline()
    pipeline.run_batch()
            if os.path.exists(meta_path):
                import json
                with open(meta_path, "r") as f:
                    try:
                        if json.load(f).get("status") == "completed":
                            logger.info(f"Skipping {file_name}: Already processed.")
                            continue
                    except: pass

            logger.info(f"Processing: {file_name}")
            
            # Metrics Tracking
            torch.cuda.reset_peak_memory_stats(self.config.device)
            start_time = time.time()
            
            try:
                # 1. Load Audio
                audio_array = self.audio_loader.load(file_path)
                duration = len(audio_array) / self.config.sample_rate
                
                # 2. Forward Pass
                result_dict = self.engine.process(audio_array)
                
                # 3. Capture GPU Metrics
                latency = time.time() - start_time
                peak_vram_mb = torch.cuda.max_memory_allocated(self.config.device) / (1024 ** 2)
                
                metrics = {
                    "file_name": file_name,
                    "audio_duration_seconds": duration,
                    "inference_latency_seconds": latency,
                    "peak_vram_mb": peak_vram_mb,
                    "real_time_factor": latency / duration if duration > 0 else 0
                }
                
                # 4. Save and Visualize
                DataPacker.save_results(output_dir, result_dict, metrics)
                Visualizer.generate_plots(
                    output_dir, 
                    result_dict["tokens"], 
                    result_dict["confidences"], 
                    result_dict["final_encoder_state"]
                )
                
                logger.success(f"Successfully generated outputs for {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_name}. Error: {str(e)}")

if __name__ == "__main__":
    pipeline = Phase1Pipeline()
    pipeline.run_batch()
