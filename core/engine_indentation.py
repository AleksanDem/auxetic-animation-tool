"""
Физический движок сценария 3: Контактное индентирование жестким индентором (core/engine_indentation.py).
Моделирует контактное внедрение индентора в полупространство:
- При nu < 0: поперечное стягивание массы к оси удара, локальное самоуплотнение (динамическое запирание);
- При nu > 0: поперечное вытеснение материала в стороны, локальное утонение (склонность к срезу пробки).
Поддерживает коническую (по умолчанию) и сферическую формы индентора,
режим прозрачного пунктирного контура, векторные поля quiver и точное соотношение сторон рамки.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.special import erf
from PIL import Image

from .base_engine import BaseAnimationEngine


class IndentationEngine(BaseAnimationEngine):
    """Генератор анимаций контактного индентирования."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.width = float(config.get("width", 3.5))
        self.depth = float(config.get("depth", 3.0))
        self.Nx = int(config.get("grid_nx", 29))
        self.Ny = int(config.get("grid_ny", 16))
        self.max_indent = float(config.get("max_indent", 0.95))
        self.a_param = float(config.get("a_param", 1.05))
        self.b_param = float(config.get("b_param", 1.60))
        self.indentor_shape = config.get("indentor_shape", "Конический (Conical/Wedge)")
        self.indentor_transparent = config.get("indentor_transparent", False)
        self.show_quiver = config.get("show_quiver", True)

    def _compute_fields(self, X: np.ndarray, Y: np.ndarray, delta: float, nu: float):
        """Аналитический расчет полей смещений и локальной плотности."""
        if abs(delta) < 1e-5:
            return X, Y, np.zeros_like(X), np.zeros_like(Y), np.ones_like(X)

        eff_delta = delta * self.k_disp
        depth_decay = np.exp(Y / self.b_param)
        lateral_decay = np.exp(- (X**2) / (2.0 * self.a_param**2))
        uy = - eff_delta * lateral_decay * depth_decay

        # Поле горизонтальных смещений (интеграл Буссинеска)
        amp_x = nu * (eff_delta / self.b_param) * np.sqrt(np.pi / 2.0) * self.a_param
        ux = amp_x * erf(X / (np.sqrt(2.0) * self.a_param)) * depth_decay

        # Объемная деформация и плотность
        eps_y = uy / self.b_param
        eps_vol = (1.0 - nu) * eps_y
        rho_rel = 1.0 - eps_vol

        return X + ux, Y + uy, ux, uy, rho_rel

    def _draw_indentor(self, ax: plt.Axes, delta: float):
        """Отрисовывает индентор (конический по умолчанию или сферический) с опцией прозрачности."""
        eff_delta = delta * self.k_disp
        ind_y_bot = - eff_delta
        ind_w = 0.65
        ind_h = 1.20

        is_transparent = getattr(self, "indentor_transparent", False)
        if is_transparent:
            color_ind = "none"
            color_edge = self.palette["text_dark"]
            line_style = "--"
            line_width = 2.0
            text_color = self.palette["text_dark"]
        else:
            color_ind = self.palette["indentor"]
            color_edge = self.palette["indentor_edge"]
            line_style = "-"
            line_width = 1.8
            text_color = "#F8FAFC"

        if "Сферический" in self.indentor_shape:
            # Сферический наконечник (полуокружность радиуса ind_w)
            R = ind_w
            thetas = np.linspace(np.pi, 2 * np.pi, 36)
            arc_x = R * np.cos(thetas)
            arc_y = (ind_y_bot + R) + R * np.sin(thetas)
            poly = [(-ind_w, ind_y_bot + ind_h)]
            for px, py in zip(arc_x, arc_y):
                poly.append((px, py))
            poly.append((ind_w, ind_y_bot + ind_h))
        else:
            # Конический индентор (по умолчанию)
            poly = [
                (-ind_w, ind_y_bot + ind_h),
                (-ind_w * 0.95, ind_y_bot + ind_h * 0.45),
                (0.0, ind_y_bot),
                (ind_w * 0.95, ind_y_bot + ind_h * 0.45),
                (ind_w, ind_y_bot + ind_h)
            ]

        patch = Polygon(
            poly,
            closed=True,
            facecolor=color_ind,
            edgecolor=color_edge,
            linewidth=line_width,
            linestyle=line_style,
            zorder=12
        )
        ax.add_patch(patch)

        # Тело индентора чистое без надписей


    def generate_single_frame(self, t_param: float) -> Image.Image:
        """
        Рендерит один кадр контактного индентирования для параметра заглубления t_param in [0, 1].
        """
        delta = self.max_indent * abs(t_param)
        fig_w, fig_h = self.get_figure_dimensions()

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        fig.patch.set_facecolor(self.palette["bg"])

        panels = self.setup_layout_axes(fig, is_3d=False)

        x_grid = np.linspace(-self.width, self.width, self.Nx)
        y_grid = np.linspace(-self.depth, 0.0, self.Ny)
        X0, Y0 = np.meshgrid(x_grid, y_grid)

        # Расчет границ карточки с точным соотношением сторон
        aspect = self.frame_aspect if self.frame_aspect > 0 else 1.0
        x_req = self.width * 1.12
        y_span_req = self.depth * 1.85

        if "Без внешних" in self.border_style:
            x_card = x_req
            y_span = y_span_req
        elif (2 * x_req / y_span_req) < aspect:
            y_span = y_span_req
            x_card = (y_span * aspect) / 2.0
        else:
            x_card = x_req
            y_span = (2 * x_card) / aspect

        # Смещаем ноль так, чтобы верхняя кромка материала была ближе к середине
        y_center = -self.depth * 0.35
        y_min = y_center - y_span / 2.0
        y_max = y_center + y_span / 2.0

        # Убираем надпись статуса сверху
        state_text = ""

        for ax, nu, title, subtitle, is_auxetic in panels:
            ax.set_facecolor(self.palette["card_bg"])
            ax.set_xlim(-x_card * 1.02, x_card * 1.02)
            ax.set_ylim(y_min * 1.02, y_max * 1.02)
            ax.set_aspect("equal")
            ax.axis("off")

            # Расчет деформированного состояния
            X_def, Y_def, Ux, Uy, Rho = self._compute_fields(X0, Y0, delta, nu)
            c_style = self.palette["auxetic"] if is_auxetic else self.palette["classic"]

            # 1. Цветовая карта локальной плотности
            cmap = c_style["cmap"]
            levels = np.linspace(1.0, 1.85, 18) if is_auxetic else np.linspace(1.0, 1.55, 12)
            ax.contourf(X_def, Y_def, Rho, levels=levels, cmap=cmap, alpha=0.55, extend="both", zorder=3)

            # 2. Недеформированный контур полупространства (Ghost, тонкий пунктир)
            if self.show_ghost:
                ax.plot(
                    [-self.width, self.width, self.width, -self.width, -self.width],
                    [-self.depth, -self.depth, 0.0, 0.0, -self.depth],
                    color=self.palette["grid_orig"], linestyle="--", linewidth=0.85, alpha=0.45, zorder=2
                )

            # 3. Деформированная КЭ-сетка (деликатная тонкая)
            if self.show_grid:
                for j in range(self.Ny):
                    ax.plot(X_def[j, :], Y_def[j, :], color=self.palette["grid_def"], linewidth=0.55, alpha=0.35, zorder=4)
                for i in range(self.Nx):
                    ax.plot(X_def[:, i], Y_def[:, i], color=self.palette["grid_def"], linewidth=0.55, alpha=0.35, zorder=4)

            # Верхняя контактная поверхность
            ax.plot(X_def[-1, :], Y_def[-1, :], color=self.palette["text_dark"], linewidth=1.7, zorder=6)

            # 4. Векторное поле перемещений узлов (Quiver)
            if self.show_quiver and delta > 0.08:
                step_x = 2
                step_y = 2
                q_X = X_def[self.Ny-7::step_y, 4:-4:step_x]
                q_Y = Y_def[self.Ny-7::step_y, 4:-4:step_x]
                q_Ux = Ux[self.Ny-7::step_y, 4:-4:step_x]
                q_Uy = Uy[self.Ny-7::step_y, 4:-4:step_x]

                arr_col = c_style["arrow"]
                ax.quiver(
                    q_X, q_Y, q_Ux, q_Uy,
                    color=arr_col,
                    scale=2.8,
                    width=0.0065,
                    headwidth=3.6,
                    headlength=4.0,
                    alpha=0.85,
                    zorder=8
                )

            # 5. Жесткий индентор
            self._draw_indentor(ax, delta)

            # 6. Оформление карточки
            self.render_card_decorations(
                ax=ax,
                x_min=-x_card,
                x_max=x_card,
                y_min=y_min,
                y_max=y_max,
                title=title,
                subtitle=subtitle,
                status_text=state_text,
                is_auxetic=is_auxetic
            )

        # Настройка полей полотна и зазора между рамками
        self.adjust_figure_margins(fig, is_3d=False)

        # Отрисовка аналитической формулы при ее включении
        self.render_analytical_formula(fig)

        im = self.fig_to_pil(fig)
        plt.close(fig)
        return im
