"""
Физический движок сценария 2: 3D-изгиб пластин (core/engine_bending.py).
Демонстрирует фундаментальный геометрический эффект ауксетиков:
- Синкластический изгиб (КУПОЛ/ЧАША) при nu < 0 (кривизны одного знака);
- Антикластический изгиб (СЕДЛО) при nu > 0 (кривизны противоположных знаков).
Включает управление 3D-ракурсом камеры, цветовыми градиентами высоты и сеткой.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from .base_engine import BaseAnimationEngine


class BendingEngine(BaseAnimationEngine):
    """Генератор 3D-анимаций изгиба пластин."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.L = config.get("plate_size", 1.6)
        self.grid_n = config.get("grid_n", 25)
        self.max_curvature = config.get("max_curvature", 0.55)
        self.camera_elev = config.get("camera_elev", 28.0)
        self.camera_azim = config.get("camera_azim", -55.0)

    def generate_single_frame(self, t_param: float) -> Image.Image:
        """
        Рендерит один 3D-кадр изгиба пластины для параметра кривизны t_param in [0, 1].
        """
        kappa = self.max_curvature * abs(t_param) * self.k_disp
        fig_w, fig_h = self.get_figure_dimensions()

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        fig.patch.set_facecolor(self.palette["bg"])

        panels = self.setup_layout_axes(fig, is_3d=True)

        x = np.linspace(-self.L, self.L, self.grid_n)
        y = np.linspace(-self.L, self.L, self.grid_n)
        X, Y = np.meshgrid(x, y)

        for ax, nu, title, subtitle, is_auxetic in panels:
            ax.set_facecolor(self.palette["card_bg"])

            # Аналитическое уравнение прогиба пластины Сен-Венана / Кирхгофа:
            # w(x, y) = 0.5 * kappa * (x^2 - nu * y^2)
            Z = 0.5 * kappa * (X**2 - nu * (Y**2))

            c_style = self.palette["auxetic"] if is_auxetic else self.palette["classic"]
            cmap = c_style["cmap"]

            # Отрисовка 3D-поверхности
            ax.plot_surface(
                X, Y, Z,
                cmap=cmap,
                edgecolor=self.palette["grid_def"],
                linewidth=0.40,
                alpha=0.85,
                antialiased=True,
                zorder=4
            )

            # Настройка ракурса
            ax.view_init(elev=self.camera_elev, azim=self.camera_azim)

            # Фиксированные границы для стабильности видеоряда
            ax.set_xlim(-1.2 * self.L, 1.2 * self.L)
            ax.set_ylim(-1.2 * self.L, 1.2 * self.L)
            ax.set_zlim(-0.85 * self.L, 1.25 * self.L)

            # Академическая чистота: скрываем тики осей
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.grid(False)

            # Тонкие грани куба сцены
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(self.palette["grid_orig"])
                pane.set_linewidth(0.8)

            # Заголовок панели
            if self.show_labels and title:
                ax.set_title(
                    title,
                    fontsize=self.font_size_title,
                    fontweight="bold",
                    color=c_style["primary"],
                    pad=6 + self.title_y_offset * 20
                )

        # Поясняющие плашки (размещаются под/над панелями)
        if self.show_boxes and self.box_position != "Скрыть":
            n_p = len(panels)
            # Расчет базовой высоты плашки
            base_y = 0.88 if "Вверху" in self.box_position else 0.06
            y_box = base_y + self.box_y_offset * 0.15

            for idx, (ax, nu, title, subtitle, is_auxetic) in enumerate(panels):
                c_style = self.palette["auxetic"] if is_auxetic else self.palette["classic"]
                if subtitle:
                    x_center = (idx + 0.5) / n_p
                    fig.text(
                        x_center, y_box, subtitle,
                        fontsize=self.font_size_box,
                        fontweight="bold",
                        color=c_style["box_text"],
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            fc=c_style["box_bg"],
                            ec=c_style["box_border"],
                            lw=1.0
                        )
                    )

        # Настройка полей полотна и зазора между рамками
        self.adjust_figure_margins(fig, is_3d=True)

        # Отрисовка аналитической формулы при ее включении
        self.render_analytical_formula(fig)

        im = self.fig_to_pil(fig)
        plt.close(fig)
        return im
