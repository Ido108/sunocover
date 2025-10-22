# your_rvc_script_new.py
import argparse
import os
import sys
from pathlib import Path
from multiprocessing import cpu_count

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from fairseq import checkpoint_utils
from scipy.io import wavfile
import warnings

# השבת הזהרות של PyTorch
warnings.filterwarnings("ignore", message=".*weights_only.*")

# --- מוסיפים את הנתיבים הנכונים של RVC ---
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "RVC-v2-UI", "src"))

# Monkey patch faiss to handle Hebrew paths
import shutil
def patch_faiss_for_hebrew_paths():
    try:
        import faiss
        original_read_index = faiss.read_index
        
        def read_index_with_hebrew_support(filename):
            try:
                return original_read_index(filename)
            except:
                # אם נכשל, נסה להעתיק לתיקיה זמנית עם שם באנגלית
                if not isinstance(filename, str):
                    filename = str(filename)
                
                # צור תיקיה זמנית
                temp_dir = os.path.join(now_dir, "temp_index")
                os.makedirs(temp_dir, exist_ok=True)
                
                # שם קובץ זמני
                temp_filename = os.path.join(temp_dir, "temp_index.index")
                
                # העתק את הקובץ
                shutil.copy2(filename, temp_filename)
                
                # קרא מהקובץ הזמני
                result = original_read_index(temp_filename)
                
                # נקה את הקובץ הזמני
                try:
                    os.remove(temp_filename)
                except:
                    pass
                    
                return result
        
        faiss.read_index = read_index_with_hebrew_support
        return True
    except Exception as e:
        print(f"Warning: Could not patch faiss for Hebrew support: {e}")
        return False

# החל את הpatch
patch_faiss_for_hebrew_paths()

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
# Custom load_audio that handles Hebrew paths better
import ffmpeg
import numpy as np

def load_audio_safe(file, sr):
    """Load audio with better path handling for Hebrew characters"""
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        # אם הנתיב מכיל תווים בעברית, העתק לתיקיה זמנית
        if any('\u0590' <= c <= '\u05FF' for c in file):
            temp_dir = os.path.join(now_dir, "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)

            temp_file = os.path.join(temp_dir, "temp_input.wav")
            shutil.copy2(file, temp_file)
            file_to_process = temp_file
        else:
            file_to_process = file

        try:
            out, err = (
                ffmpeg.input(file_to_process, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error stderr: {e.stderr.decode() if e.stderr else 'No stderr'}", file=sys.stderr)
            raise RuntimeError(f"FFmpeg failed: {e}")

        # נקה קובץ זמני אם יצרנו
        if file_to_process != file:
            try:
                os.remove(file_to_process)
            except:
                pass

    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

# השתמש בfunc המותאמת במקום המקורית
load_audio = load_audio_safe
from vc_infer_pipeline import VC

BASE_DIR = Path(now_dir) / "RVC-v2-UI"


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    config_path = BASE_DIR / "src" / "configs" / config_file
                    if config_path.exists():
                        with open(config_path, "r") as f:
                            strr = f.read().replace("true", "false")
                        with open(config_path, "w") as f:
                            f.write(strr)
                pipeline_path = BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py"
                if pipeline_path.exists():
                    with open(pipeline_path, "r") as f:
                        strr = f.read().replace("3.7", "3.0")
                    with open(pipeline_path, "w") as f:
                        f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                pipeline_path = BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py"
                if pipeline_path.exists():
                    with open(pipeline_path, "r") as f:
                        strr = f.read().replace("3.7", "3.0")
                    with open(pipeline_path, "w") as f:
                        f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='', )
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


def get_vc(device, is_half, config, model_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model):
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    
    # תקן נתיב אינדקס כך שיעבוד עם תווים בעברית
    if index_path and os.path.exists(index_path):
        # וודא שהנתיב תקין
        index_path = os.path.abspath(index_path)
    else:
        # אם אין אינדקס או שהוא לא קיים, השתמש בNone
        index_path = None
        index_rate = 0.0
    
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length)
    wavfile.write(output_path, tgt_sr, audio_opt)


def process_rvc(input_path, model_path, output_path, pitch, index_path, index_rate, protect):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = torch.cuda.is_available()
        config = Config(device, is_half)

        # Load Hubert model
        hubert_path = BASE_DIR / "rvc_models" / "hubert_base.pt"
        if not hubert_path.exists():
            print(f"Error: {hubert_path} not found. Make sure RVC models are properly installed.", file=sys.stderr)
            sys.exit(1)
        
        print("Loading Hubert model...")
        hubert_model = load_hubert(config.device, config.is_half, str(hubert_path))
        
        print("Loading RVC model...")
        cpt, version, net_g, tgt_sr, vc = get_vc(config.device, config.is_half, config, model_path)
        print(f"Model loaded successfully. Target SR: {tgt_sr}, Version: {version}")

        print("Starting voice conversion...")
        rvc_infer(
            index_path,
            index_rate,
            input_path,
            output_path,
            pitch,
            "rmvpe",  # f0_method
            cpt,
            version,
            net_g,
            3,  # filter_radius
            tgt_sr,
            0.25,  # rms_mix_rate
            protect,
            120,  # crepe_hop_length
            vc,
            hubert_model
        )
        
        print(f"Conversion successful. Output written to: {output_path}")
        # מדפיסים את הנתיב כדי שהשרת הראשי ידע מה לשלוח בחזרה
        print(output_path, flush=True)

    except Exception as e:
        print(f"Error in RVC processing: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pitch", type=int, required=True)
    parser.add_argument("--index_rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.33)
    args = parser.parse_args()

    # חיפוש אוטומטי של קובץ אינדקס תואם למודל
    index_file = None
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # נסה מספר אפשרויות לשם קובץ האינדקס
    possible_names = [
        f"{model_name}.index",
    ]
    
    for name in possible_names:
        potential_index_path = os.path.join(model_dir, name)
        if os.path.exists(potential_index_path):
            index_file = potential_index_path
            print("Found index file automatically:", os.path.basename(index_file))
            break
    
    # אם לא נמצא, חפש כל קובץ .index בתיקיה
    if index_file is None:
        try:
            for file in os.listdir(model_dir):
                if file.endswith('.index'):
                    index_file = os.path.join(model_dir, file)
                    print("Found index file by extension:", os.path.basename(index_file))
                    break
        except Exception as e:
            print("Error accessing model directory:", str(e))
    
    if index_file is None:
        print("No index file found for the model. Running without index (may affect quality).")
    else:
        print("Will use index file for better quality conversion.")

    try:
        process_rvc(
            args.input_path,
            args.model_path,
            args.output_path,
            args.pitch,
            index_file,
            args.index_rate,
            args.protect
        )
    except Exception as e:
        print(f"Failed to process RVC: {e}", file=sys.stderr)
        sys.exit(1)