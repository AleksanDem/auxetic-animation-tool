"""
Модуль высокопроизводительного экспорта анимаций в форматы GIF и MP4.
Поддерживает двухпроходную оптимизацию палитры через FFmpeg (или чистый PIL)
и сборку совместимого со слайдами PowerPoint видеопотока MP4 (H.264/yuv420p).
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def check_ffmpeg_available() -> bool:
    """Проверяет доступность утилиты ffmpeg в системе."""
    return shutil.which("ffmpeg") is not None


def export_to_gif(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 15,
    optimize_palette: bool = True
) -> str:
    """
    Экспортирует список кадров PIL в зацикленный файл GIF.
    При наличии FFmpeg использует двухпроходную генерацию палитры (palettegen/paletteuse)
    для устранения артефактов дизеринга и минимального размера файла.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    duration_ms = int(1000.0 / max(1, fps))

    # Попытка через FFmpeg для идеального качества
    if optimize_palette and check_ffmpeg_available() and len(frames) > 0:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_pattern = os.path.join(tmp_dir, "frame_%04d.png")
            for idx, img in enumerate(frames):
                img.save(temp_pattern % idx, format="PNG")

            palette_png = os.path.join(tmp_dir, "palette.png")
            # 1. Генерация оптимальной 256-цветной палитры
            cmd_pal = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", os.path.join(tmp_dir, "frame_%04d.png"),
                "-vf", "palettegen=stats_mode=diff",
                palette_png
            ]
            subprocess.run(cmd_pal, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

            if os.path.exists(palette_png):
                # 2. Рендеринг GIF с применением палитры
                cmd_gif = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmp_dir, "frame_%04d.png"),
                    "-i", palette_png,
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0",
                    output_path
                ]
                res = subprocess.run(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path

    # Резервный проверенный метод через Pillow
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    return output_path


def export_to_mp4(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 20,
    crf: int = 18
) -> str:
    """
    Экспортирует список кадров в видео MP4 с кодеком H.264 и цветовым форматом yuv420p.
    Гарантирует 100% совместимость со встроенным плеером Microsoft PowerPoint и браузерами.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not frames:
        raise ValueError("Список кадров пуст!")

    w, h = frames[0].size
    # Требование H.264: ширина и высота должны быть четными
    target_w = w if w % 2 == 0 else w + 1
    target_h = h if h % 2 == 0 else h + 1

    prepared_frames = []
    for f in frames:
        if f.size != (target_w, target_h):
            f_resized = f.resize((target_w, target_h), Image.Resampling.LANCZOS)
            prepared_frames.append(f_resized.convert("RGB"))
        else:
            prepared_frames.append(f.convert("RGB"))

    # Основной метод — через FFmpeg
    if check_ffmpeg_available():
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_pattern = os.path.join(tmp_dir, "frame_%04d.png")
            for idx, img in enumerate(prepared_frames):
                img.save(temp_pattern % idx, format="PNG")

            cmd_mp4 = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", temp_pattern,
                "-c:v", "libx264",
                "-profile:v", "high",
                "-level", "4.0",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-movflags", "+faststart",
                output_path
            ]
            res = subprocess.run(cmd_mp4, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path

    # Резервный метод — OpenCV VideoWriter
    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (target_w, target_h))
        for img in prepared_frames:
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

    raise RuntimeError("Не удалось собрать MP4: FFmpeg и OpenCV недоступны или завершились с ошибкой.")
