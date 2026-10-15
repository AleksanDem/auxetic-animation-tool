"""
Физический движок сценария 1: Одноосное растяжение и сжатие (core/engine_uniaxial.py).
Строгая континуумная механика сплошной среды:
- Деформируемая КЭ-сетка;
- Изополя локального изменения объема / плотности;
- Динамические векторные стрелки нагрузки и поперечного отклика;
- Адаптивные границы карточки с точным соотношением сторон рамки (frame_aspect)
  и гарантированной защитой от вылезания контура при любых коэффициентах Пуассона.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .base_engine import BaseAnimationEngine


class UniaxialEngine(BaseAnimationEngine):
    """Генератор анимаций одноосного деформирования континуума."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.W0 = float(config.get("width", 1.0))
        self.H0 = float(config.get("height", 1.5))
        self.Nx = int(config.get("grid_nx", 15))
        self.Ny = int(config.get("grid_ny", 21))
        self.max_strain = float(config.get("max_strain", 0.20))

    def _compute_frame_bounds(self, panels: List[Tuple[Any, float, str, str, bool]]) -> Tuple[float, float]:
        """
        Рассчитывает полуширину X_card и полувысоту Y_card рамки карточки так,
        чтобы деформированный образец, стрелки сил и двухстрочный заголовок гармонично
        размещались без наложений и без лишней боковой пустоты.
        """
        # Максимальная высота образца
        h_max = self.H0 * (1.0 + self.max_strain * self.k_disp)
        # Стрелка нагрузки (0.28 * H0) + двухстрочный заголовок (~0.65) + защитный отступ
        y_req = h_max + 0.28 * self.H0 + 0.85

        # Максимальная ширина образца среди всех панелей
        w_max = self.W0
        for _, nu, _, _, _ in panels:
            w_def = self.W0 * (1.0 + abs(nu) * self.max_strain * self.k_disp)
            if w_def > w_max:
                w_max = w_def

        # Запас под боковые стрелки отклика (0.30 * W0) + отступ
        x_req = w_max + 0.30 * self.W0 + 0.30

        if "Без внешних" in self.border_style:
            # В минималистичном режиме без рамок границы плотно облегают геометрию:
            # образцы сближаются, а расстояние регулируется слайдером зазора!
            return x_req, y_req

        aspect = self.frame_aspect if self.frame_aspect > 0 else 1.0

        # Согласовываем с заданным соотношением сторон рамки R = aspect
        if (x_req / y_req) < aspect:
            y_card = y_req
            x_card = y_req * aspect
        else:
            x_card = x_req
            y_card = x_req / aspect

        return x_card, y_card

    def generate_single_frame(self, t_param: float) -> Image.Image:
        """
        Рендерит один кадр одноосного деформирования для кинематического параметра t_param in [-1, 1].
        """
        eps_base = self.max_strain * t_param
        fig_w, fig_h = self.get_figure_dimensions()

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        fig.patch.set_facecolor(self.palette["bg"])

        panels = self.setup_layout_axes(fig, is_3d=False)
        x_card, y_card = self._compute_frame_bounds(panels)

        x0_grid = np.linspace(-self.W0, self.W0, self.Nx)
        y0_grid = np.linspace(-self.H0, self.H0, self.Ny)
        X0, Y0 = np.meshgrid(x0_grid, y0_grid)

        # 1. Убираем надпись "растяжение" сверху
        state_text = ""
        if eps_base > 0.01:
            arr_load_col = self.palette["load_tension"]
        elif eps_base < -0.01:
            arr_load_col = self.palette["load_compression"]
        else:
            arr_load_col = "#64748B"

        for ax, nu, title, subtitle, is_auxetic in panels:
            ax.set_facecolor(self.palette["card_bg"])
            ax.set_xlim(-x_card * 1.02, x_card * 1.02)
            ax.set_ylim(-y_card * 1.02, y_card * 1.02)
            ax.set_aspect("equal")
            ax.axis("off")

            eps_y = eps_base * self.k_disp
            eps_x = - nu * eps_y

            # Деформированные координаты континуума
            X_def = X0 * (1.0 + eps_x)
            Y_def = Y0 * (1.0 + eps_y)

            # Относительное изменение объема / плотность
            rho_rel = 1.0 - (eps_x + eps_y)
            c_style = self.palette["auxetic"] if is_auxetic else self.palette["classic"]

            # 1. Цветовая карта локальной плотности
            cmap = c_style["cmap"]
            levels = np.linspace(0.70, 1.50, 16)
            ax.contourf(
                X_def, Y_def, np.full_like(X_def, rho_rel),
                levels=levels, cmap=cmap, alpha=0.45, extend="both", zorder=3
            )

            # 2. Деформированная КЭ-сетка (деликатная тонкая)
            if self.show_grid:
                for j in range(self.Ny):
                    ax.plot(X_def[j, :], Y_def[j, :], color=self.palette["grid_def"], linewidth=0.55, alpha=0.35, zorder=4)
                for i in range(self.Nx):
                    ax.plot(X_def[:, i], Y_def[:, i], color=self.palette["grid_def"], linewidth=0.55, alpha=0.35, zorder=4)

            # 3. Недеформированный контур (Ghost, тонкий пунктир)
            if self.show_ghost:
                ax.plot(
                    [-self.W0, self.W0, self.W0, -self.W0, -self.W0],
                    [-self.H0, -self.H0, self.H0, self.H0, -self.H0],
                    color=self.palette["grid_orig"], linestyle="--", linewidth=0.85, alpha=0.45, zorder=2
                )

            # 4. Внешний деформированный контур
            ax.plot(X_def[0, :], Y_def[0, :], color=self.palette["text_dark"], linewidth=1.6, zorder=6)
            ax.plot(X_def[-1, :], Y_def[-1, :], color=self.palette["text_dark"], linewidth=1.6, zorder=6)
            ax.plot(X_def[:, 0], Y_def[:, 0], color=self.palette["text_dark"], linewidth=1.6, zorder=6)
            ax.plot(X_def[:, -1], Y_def[:, -1], color=self.palette["text_dark"], linewidth=1.6, zorder=6)

            # 5. Векторы сил и поперечного отклика
            curr_H = self.H0 * (1.0 + eps_y)
            curr_W = self.W0 * (1.0 + eps_x)

            if self.show_arrows and abs(eps_base) > 0.02:
                # Вертикальные стрелки внешней нагрузки
                dy_arr = 0.28 * self.H0 if eps_base > 0 else -0.28 * self.H0
                ax.annotate(
                    "", xy=(0, curr_H + dy_arr), xytext=(0, curr_H),
                    arrowprops=dict(arrowstyle="->", color=arr_load_col, lw=2.4, mutation_scale=13),
                    zorder=10
                )
                ax.annotate(
                    "", xy=(0, -curr_H - dy_arr), xytext=(0, -curr_H),
                    arrowprops=dict(arrowstyle="->", color=arr_load_col, lw=2.4, mutation_scale=13),
                    zorder=10
                )

                # Боковые стрелки отклика (демонстрация знака nu)
                dx_arr = 0.30 * self.W0 if eps_x > 0 else -0.30 * self.W0
                arr_resp_col = c_style["arrow"]
                ax.annotate(
                    "", xy=(curr_W + dx_arr, 0), xytext=(curr_W, 0),
                    arrowprops=dict(arrowstyle="->", color=arr_resp_col, lw=2.4, mutation_scale=13),
                    zorder=10
                )
                ax.annotate(
                    "", xy=(-curr_W - dx_arr, 0), xytext=(-curr_W, 0),
                    arrowprops=dict(arrowstyle="->", color=arr_resp_col, lw=2.4, mutation_scale=13),
                    zorder=10
                )

            # 6. Оформление карточки с точным соотношением сторон
            self.render_card_decorations(
                ax=ax,
                x_min=-x_card,
                x_max=x_card,
                y_min=-y_card,
                y_max=y_card,
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
