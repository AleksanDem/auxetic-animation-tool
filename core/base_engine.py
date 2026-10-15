"""
Базовый движок для генерации кадров и управления компоновками (core/base_engine.py).
Обеспечивает унифицированную координатную геометрию, тайминги кинематики,
согласованные пропорции рамок карточек и управление зазором между ними.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .styles import PALETTES


class BaseAnimationEngine(ABC):
    """Абстрактный базовый класс для физических движков деформирования."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.palette_name = config.get("palette", "Academic (Blue-Orange)")
        self.palette = PALETTES.get(self.palette_name, PALETTES["Academic (Blue-Orange)"])
        self.dpi = config.get("dpi", 85)
        self.layout = config.get("layout", "Горизонтальная (Side-by-Side 1x2)")
        
        # 1. Соотношение сторон отдельной рамки и зазор между рамками
        self.frame_aspect = float(config.get("frame_aspect", 1.0))
        self.frame_gap = float(config.get("frame_gap", 0.15))
        
        # 5. Стиль внешней рамки (кружок/радиокнопка, по умолчанию - без рамок)
        self.border_style = config.get("border_style", "Без внешних рамок")
        
        # Шрифты, стиль и положение надписей
        self.font_size_title = float(config.get("font_size_title", 12.5))
        self.font_size_box = float(config.get("font_size_box", 10.0))
        self.font_size_status = float(config.get("font_size_status", 9.5))
        self.box_position = config.get("box_position", "Внизу")
        self.box_style = config.get("box_style", "Чистый текст (без плашки)")
        self.box_y_offset = float(config.get("box_y_offset", 0.0))
        self.title_y_offset = float(config.get("title_y_offset", 0.0))
        
        # Аналитическая формула на анимации
        self.formula_position = config.get("formula_position", "Не отображать")
        self.formula_text = config.get("formula_text", "")
        self.formula_font_size = float(config.get("formula_font_size", 20.0))
        self.formula_x_offset = float(config.get("formula_x_offset", 0.0))
        self.formula_y_offset = float(config.get("formula_y_offset", 0.0))
        self.formula_bg_style = config.get("formula_bg_style", "Прозрачный фон (по умолчанию)")

        self.show_arrows = config.get("show_arrows", True)
        self.show_labels = config.get("show_labels", True)
        self.show_boxes = config.get("show_boxes", True)
        self.show_grid = config.get("show_grid", True)
        self.show_ghost = config.get("show_ghost", True)
        self.k_disp = float(config.get("k_disp", 1.0))

    def get_figure_dimensions(self) -> Tuple[float, float]:
        """
        Рассчитывает общую ширину и высоту полотна Matplotlib в дюймах исходя из
        количества окон, соотношения сторон отдельной рамки (frame_aspect), зазора (frame_gap)
        и выбранного положения аналитической формулы.
        """
        if "Вертикальная" in self.layout:
            ncols, nrows = 1, 2
        elif "2 на 2" in self.layout or "2x2" in self.layout:
            ncols, nrows = 2, 2
        else:
            # Горизонтальная 1x2
            ncols, nrows = 2, 1

        aspect = self.frame_aspect if self.frame_aspect > 0 else 1.0
        h_cell = 4.8
        w_cell = h_cell * aspect

        gap = self.frame_gap
        total_w = ncols * w_cell * (1.0 + (ncols - 1) * gap * 0.45)
        total_h = nrows * h_cell * (1.0 + (nrows - 1) * gap * 0.45)

        # Резервирование пространства под формулу
        has_formula = bool(self.formula_text and "Не отображать" not in self.formula_position)
        if has_formula:
            if "Справа от рамок" in self.formula_position:
                total_w += 3.2
            elif "Под рамками" in self.formula_position:
                total_h += 1.1
            elif "Между рамками" in self.formula_position and ncols > 1:
                total_w += 1.4

        # Ограничение максимального размера
        max_dim = max(total_w, total_h)
        if max_dim > 18.0:
            scale = 18.0 / max_dim
            total_w *= scale
            total_h *= scale

        return total_w, total_h

    def adjust_figure_margins(self, fig: plt.Figure, is_3d: bool = False):
        """
        Централизованно настраивает поля фигуры с учетом зазора frame_gap
        и резервирования места под аналитическую формулу.
        """
        pos = self.formula_position
        has_formula = bool(self.formula_text and "Не отображать" not in pos)

        left = 0.03
        right = 0.97
        top = 0.96 if not is_3d else 0.86
        bottom = 0.04 if not is_3d else 0.06
        wspace = self.frame_gap
        hspace = self.frame_gap

        if has_formula:
            if "Справа от рамок" in pos:
                right = 0.72
            elif "Под рамками" in pos:
                bottom = 0.18 if not is_3d else 0.16
            elif "Между рамками" in pos:
                wspace = max(self.frame_gap, 0.28)

        fig.subplots_adjust(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            wspace=wspace,
            hspace=hspace
        )

    def render_analytical_formula(self, fig: plt.Figure):
        """
        Отрисовывает блок аналитических формул на полотне анимации
        в заданной позиции: 'Между рамками', 'Под рамками', 'Справа от рамок'.
        """
        if not self.formula_text or "Не отображать" in self.formula_position:
            return

        pos = self.formula_position
        if "Между рамками" in pos:
            x = 0.50 + self.formula_x_offset
            y = 0.50 + self.formula_y_offset
            ha = "center"
            va = "center"
        elif "Под рамками" in pos:
            x = 0.50 + self.formula_x_offset
            y = 0.075 + self.formula_y_offset
            ha = "center"
            va = "center"
        elif "Справа от рамок" in pos:
            x = 0.865 + self.formula_x_offset
            y = 0.50 + self.formula_y_offset
            ha = "center"
            va = "center"
        else:
            return

        bbox_props = None
        # Цвет текста формулы: темный для светлых палитр, светлый для темных
        text_color = self.palette.get("text_dark", self.palette.get("text", "#0F172A"))

        if "Белая плашка" in self.formula_bg_style:
            bbox_props = dict(
                boxstyle="round,pad=0.45",
                fc="#FFFFFF",
                ec="#CBD5E1",
                lw=0.8,
                alpha=0.98
            )
            text_color = "#0F172A"
        elif "Темная плашка" in self.formula_bg_style:
            bbox_props = dict(
                boxstyle="round,pad=0.45",
                fc="#181F2A",
                ec="#388BFD",
                lw=1.0,
                alpha=0.95
            )
            text_color = "#F0F6FC"
        else:
            # Прозрачный фон (по умолчанию)
            bbox_props = None

        fig.text(
            x, y,
            self.formula_text,
            fontsize=self.formula_font_size,
            color=text_color,
            ha=ha,
            va=va,
            multialignment="center",
            bbox=bbox_props,
            zorder=100
        )


    def compute_time_profile(self, n_frames: int, motion_type: str = "symmetric_cycle", hold_frames: int = 4) -> np.ndarray:
        """Генерирует временной профиль деформации в диапазоне [-1, 1] или [0, 1]."""
        if n_frames < 6:
            n_frames = 6

        if motion_type == "symmetric_cycle":
            n_quarter = max(3, (n_frames - 2 * hold_frames) // 4)
            profile = []
            for i in range(n_quarter):
                t = i / (n_quarter - 1)
                profile.append(0.5 * (1.0 - np.cos(np.pi * t)))
            for _ in range(hold_frames):
                profile.append(1.0)
            for i in range(n_quarter):
                t = i / (n_quarter - 1)
                profile.append(0.5 * (1.0 + np.cos(np.pi * t)))
            for i in range(n_quarter):
                t = i / (n_quarter - 1)
                profile.append(-0.5 * (1.0 - np.cos(np.pi * t)))
            for _ in range(hold_frames):
                profile.append(-1.0)
            for i in range(n_quarter):
                t = i / (n_quarter - 1)
                profile.append(-0.5 * (1.0 + np.cos(np.pi * t)))

            prof_arr = np.array(profile)
            if len(prof_arr) != n_frames:
                orig_x = np.linspace(0, 1, len(prof_arr))
                target_x = np.linspace(0, 1, n_frames)
                return np.interp(target_x, orig_x, prof_arr)
            return prof_arr

        elif motion_type == "unidirectional_hold":
            n_active = max(4, (n_frames - hold_frames * 2) // 2)
            profile = []
            for i in range(n_active):
                t = i / (n_active - 1)
                profile.append(0.5 * (1.0 - np.cos(np.pi * t)))
            for _ in range(hold_frames):
                profile.append(1.0)
            for i in range(n_active):
                t = i / (n_active - 1)
                profile.append(0.5 * (1.0 + np.cos(np.pi * t)))
            for _ in range(hold_frames):
                profile.append(0.0)

            prof_arr = np.array(profile)
            if len(prof_arr) != n_frames:
                orig_x = np.linspace(0, 1, len(prof_arr))
                target_x = np.linspace(0, 1, n_frames)
                return np.interp(target_x, orig_x, prof_arr)
            return prof_arr

        else:
            t_vals = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
            return np.sin(t_vals)

    def setup_layout_axes(self, fig: plt.Figure, is_3d: bool = False) -> List[Tuple[plt.Axes, float, str, str, bool]]:
        """
        Создает подграфики в строгом соответствии с компоновкой:
        - Горизонтальная (Side-by-Side 1x2)
        - Вертикальная (Top/Bottom 2x1)
        - Сетка 2 на 2 (2x2)
        """
        nu1 = float(self.config.get("nu1", 0.33))
        nu2 = float(self.config.get("nu2", -0.50))
        nu3 = float(self.config.get("nu3", 0.0))
        nu4 = float(self.config.get("nu4", -1.0))

        title1 = self.config.get("title1", f"Неауксетик (ν = {nu1:+.2f})")
        title2 = self.config.get("title2", f"Ауксетик (ν = {nu2:+.2f})")
        title3 = self.config.get("title3", f"Материал 3 (ν = {nu3:+.2f})")
        title4 = self.config.get("title4", f"Материал 4 (ν = {nu4:+.2f})")

        sub1 = self.config.get("sub1", "")
        sub2 = self.config.get("sub2", "")
        sub3 = self.config.get("sub3", "")
        sub4 = self.config.get("sub4", "")

        panels_data = []

        if "Вертикальная" in self.layout:
            kw = {"projection": "3d"} if is_3d else {}
            ax1 = fig.add_subplot(2, 1, 1, **kw)
            ax2 = fig.add_subplot(2, 1, 2, **kw)
            panels_data.append((ax1, nu1, title1, sub1, nu1 < 0))
            panels_data.append((ax2, nu2, title2, sub2, nu2 < 0))

        elif "2 на 2" in self.layout or "2x2" in self.layout:
            kw = {"projection": "3d"} if is_3d else {}
            ax1 = fig.add_subplot(2, 2, 1, **kw)
            ax2 = fig.add_subplot(2, 2, 2, **kw)
            ax3 = fig.add_subplot(2, 2, 3, **kw)
            ax4 = fig.add_subplot(2, 2, 4, **kw)
            panels_data.append((ax1, nu1, title1, sub1, nu1 < 0))
            panels_data.append((ax2, nu2, title2, sub2, nu2 < 0))
            panels_data.append((ax3, nu3, title3, sub3, nu3 < 0))
            panels_data.append((ax4, nu4, title4, sub4, nu4 < 0))

        else:
            # Горизонтальная 1x2
            kw = {"projection": "3d"} if is_3d else {}
            ax1 = fig.add_subplot(1, 2, 1, **kw)
            ax2 = fig.add_subplot(1, 2, 2, **kw)
            panels_data.append((ax1, nu1, title1, sub1, nu1 < 0))
            panels_data.append((ax2, nu2, title2, sub2, nu2 < 0))

        return panels_data

    def render_card_decorations(
        self,
        ax: plt.Axes,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        title: str,
        subtitle: str,
        status_text: str,
        is_auxetic: bool
    ):
        """Отрисовывает аккуратную карточку с заголовком, статусом и внешней рамкой выбранного стиля."""
        c_style = self.palette["auxetic"] if is_auxetic else self.palette["classic"]

        # 5. Отрисовка внешней рамки карточки (выбор кружком/радиокнопкой)
        if "Без внешних" not in self.border_style:
            if "Цветная" in self.border_style:
                edge_col = c_style["accent"]
                lw = 1.2
                ls = "-"
            elif "Пунктирная" in self.border_style:
                edge_col = self.palette["grid_orig"]
                lw = 0.8
                ls = "--"
            elif "Ультратонкая" in self.border_style:
                edge_col = self.palette["grid_orig"]
                lw = 0.6
                ls = "-"
            else:
                # Тонкая нейтральная
                edge_col = self.palette["grid_orig"]
                lw = 0.9
                ls = "-"

            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=edge_col,
                linewidth=lw,
                linestyle=ls,
                zorder=2
            )
            ax.add_patch(rect)

        # 1. Верхний статус деформации
        if status_text and self.show_labels:
            ax.text(
                (x_min + x_max) / 2.0,
                y_max - 0.04 * (y_max - y_min) + self.title_y_offset * (y_max - y_min),
                status_text,
                fontsize=self.font_size_status,
                fontweight="bold",
                color=c_style["accent"],
                ha="center",
                va="top",
                zorder=20
            )

        # 2. Главный заголовок панели (с переносом скобок на новую строку)
        if title and self.show_labels:
            y_title_offset = 0.035 if not status_text else 0.12
            ax.text(
                (x_min + x_max) / 2.0,
                y_max - y_title_offset * (y_max - y_min) + self.title_y_offset * (y_max - y_min),
                title,
                fontsize=self.font_size_title,
                fontweight="bold",
                color=c_style["primary"],
                ha="center",
                va="top",
                multialignment="center",
                zorder=20
            )

        # 3. Академическая плашка описания с настраиваемым положением (по умолчанию чистый текст)
        if subtitle and self.show_boxes and self.box_position != "Скрыть":
            if "Вверху" in self.box_position:
                box_y = y_max - 0.28 * (y_max - y_min) + self.box_y_offset * (y_max - y_min)
            else:
                # Внизу (по умолчанию)
                box_y = y_min + 0.08 * (y_max - y_min) + self.box_y_offset * (y_max - y_min)

            bbox_box = None
            if "Классическая" in self.box_style or ("плашка" in self.box_style.lower() and "без" not in self.box_style.lower()):
                bbox_box = dict(
                    boxstyle="round,pad=0.35",
                    fc=c_style.get("box_bg", "#F8FAFC"),
                    ec=c_style.get("box_border", "#E2E8F0"),
                    lw=0.8
                )

            ax.text(
                (x_min + x_max) / 2.0,
                box_y,
                subtitle,
                fontsize=self.font_size_box,
                fontweight="normal",
                color=c_style.get("box_text", c_style["primary"]),
                ha="center",
                va="center",
                multialignment="center",
                zorder=20,
                bbox=bbox_box
            )


    def fig_to_pil(self, fig: plt.Figure) -> Image.Image:
        """Быстро конвертирует текущую фигуру Matplotlib в объект PIL Image без I/O на диск."""
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        im = Image.fromarray(rgba)
        return im.convert("RGB")

    @abstractmethod
    def generate_single_frame(self, t_param: float) -> Image.Image:
        """Рендерит один стоп-кадр для заданного значения кинематического параметра t_param."""
        pass

    def generate_all_frames(self, n_frames: int, progress_callback=None) -> List[Image.Image]:
        """Генерирует полную последовательность кадров анимации."""
        motion_type = self.config.get("motion_type", "symmetric_cycle")
        hold_frames = self.config.get("hold_frames", 4)
        t_profile = self.compute_time_profile(n_frames, motion_type, hold_frames)

        frames = []
        for idx, t_val in enumerate(t_profile):
            frame = self.generate_single_frame(t_val)
            frames.append(frame)
            if progress_callback:
                progress_callback(idx + 1, len(t_profile))
        return frames
