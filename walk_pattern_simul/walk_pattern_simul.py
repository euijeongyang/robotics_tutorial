import sys
import math

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot
from spatialmath import SE3

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui_walk_pattern import Ui_MainWindow

# TODO
# 증폭 가능 범위 계산해서 알려주기
# 코드 정리...


CM_TO_MM = 10.0

HALF_PELVIS = 30.0
THIGH = 120.0
CALF = 120.0

# [hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll]
Q_INIT = np.radians([0.0, 0.0, -30.0, 60.0, -20.0, 0.0])

GAIT_CYCLE_SEC = 1.0
FRAME_DT_SEC = 0.05

class FootTrajectory:
    def __init__(self):
        self.points = []

    def put_point(self, time, pos, vel=0.0, acc=0.0):
        self.points.append((time, pos, vel, acc))

    def result(self, t):
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]

            if p0[0] <= t <= p1[0]:
                return self._quintic(p0, p1, t)

        return self.points[-1][1]

    @staticmethod
    def _quintic(p0, p1, t):
        t0, x0, v0, a0 = p0
        t1, x1, v1, a1 = p1
        T = t1 - t0

        c0 = x0
        c1 = v0
        c2 = a0 / 2.0
        c3 = (20.0 * (x1 - x0) - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a1 - a0) * T**2) / (2.0 * T**3)
        c4 = (30.0 * (x0 - x1) + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a1 - 2.0 * a0) * T**2) / (2.0 * T**4)
        c5 = (12.0 * (x1 - x0) - 6.0 * T * (v1 + v0) - (a1 - a0) * T**2) / (2.0 * T**5)

        tau = t - t0

        return (c0 + c1 * tau + c2 * tau**2 + c3 * tau**3 + c4 * tau**4 + c5 * tau**5)

class WalkPattern:
    def __init__(self):
        self.step = FootTrajectory()
        self.swing = FootTrajectory()
        self.rise = FootTrajectory()

        # Step pattern
        self.step.put_point(0.0, -0.5)
        self.step.put_point(0.3, -0.5)
        self.step.put_point(0.5, 0.5)
        self.step.put_point(0.8, 0.5)
        self.step.put_point(1.0, -0.5)

        # Swing pattern
        self.swing.put_point(0.0, -1.0)
        self.swing.put_point(0.5, 1.0)
        self.swing.put_point(1.0, -1.0)

        # Rise pattern
        self.rise.put_point(0.0, 0.0)
        self.rise.put_point(0.3, 0.0)
        self.rise.put_point(0.5, 1.0)
        self.rise.put_point(0.7, 0.0)
        self.rise.put_point(1.0, 0.0)

    def values(self, phase):
        phase %= 1.0
        return (
            self.step.result(phase),
            self.swing.result(phase),
            self.rise.result(phase),
        )


class LegIK:
    @staticmethod
    def clamp(value):
        if value < -1.000001 or value > 1.000001:
            raise ValueError("Target is outside the leg workspace.")
        return max(-1.0, min(1.0, value))

    @classmethod
    def solve(cls, px, py, pz, side):
        yaw = 0.0

        if side == "right":
            dy = py + HALF_PELVIS
        else:
            dy = py - HALF_PELVIS

        knee_cos = (px**2 + dy**2 + pz**2 - THIGH**2 - CALF**2) / (2.0 * THIGH * CALF)
        knee = math.acos(cls.clamp(knee_cos))

        hip_roll_pre = math.atan2(
            pz,
            -px * math.sin(yaw) + dy * math.cos(yaw),
        )

        a = -CALF * math.sin(knee)
        b = THIGH + CALF * math.cos(knee)
        c = px * math.cos(yaw) + dy * math.sin(yaw)
        d = (
            -px * math.sin(yaw) * math.cos(hip_roll_pre)
            + dy * math.cos(yaw) * math.cos(hip_roll_pre)
            + pz * math.sin(hip_roll_pre)
        )

        hip_pitch_sin = (a * d - b * c) / (a**2 + b**2)
        hip_pitch = math.asin(cls.clamp(hip_pitch_sin))

        hip_roll = hip_roll_pre + math.pi / 2.0
        ankle_pitch = -knee - hip_pitch

        ankle_roll = -hip_roll

        return np.array([yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll, ])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.pattern = WalkPattern()
        self.phase = 0.0

        self.right_leg = self.make_leg("Right Leg", -HALF_PELVIS)
        self.left_leg = self.make_leg("Left Leg", HALF_PELVIS)

        self.right_leg.q = Q_INIT.copy()
        self.left_leg.q = Q_INIT.copy()

        self.init_right_foot = np.asarray(self.right_leg.fkine(Q_INIT).t, dtype=float)
        self.init_left_foot = np.asarray(self.left_leg.fkine(Q_INIT).t, dtype=float)

        self.make_robot_viewer()
        self.make_pattern_graphs()

        self.init_step = self.ui.doubleSpinBox_step.value()
        self.init_swing = self.ui.doubleSpinBox_swing.value()
        self.init_rise = self.ui.doubleSpinBox_rise.value()

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animation_step)

        # Buttons
        self.ui.btn_walk_on.clicked.connect(self.walk_on)
        self.ui.btn_walk_stop.clicked.connect(self.walk_stop)
        self.ui.btn_init.clicked.connect(self.init_robot)

        self.ui.btn_view_front.clicked.connect(self.front_view)
        self.ui.btn_view_side.clicked.connect(self.side_view)
        self.ui.btn_view_top.clicked.connect(self.top_view)
        self.ui.btn_view_home.clicked.connect(self.home_view)

        self.ui.doubleSpinBox_step.valueChanged.connect(self.draw_patterns)
        self.ui.doubleSpinBox_swing.valueChanged.connect(self.draw_patterns)
        self.ui.doubleSpinBox_rise.valueChanged.connect(self.draw_patterns)

        self.ui.radioButton_both.setChecked(True)

        self.draw_patterns()
        self.home_view()

    @staticmethod
    def make_leg(name, hip_y):
        ets = (
            rtb.ET.Rz()               # hip yaw
            * rtb.ET.Rx()             # hip roll
            * rtb.ET.Ry()             # hip pitch
            * rtb.ET.tz(-THIGH)
            * rtb.ET.Ry()             # knee pitch
            * rtb.ET.tz(-CALF)
            * rtb.ET.Ry()             # ankle pitch
            * rtb.ET.Rx()             # ankle roll
        )

        robot = rtb.Robot(ets, name=name)
        robot.base = SE3.Ty(hip_y)
        return robot

    def make_robot_viewer(self):
        self.env = PyPlot()
        self.env.launch(
            name="12-DOF Biped Walk",
            limits=[-220, 220, -180, 180, -320, 120],
        )

        self.env.add(self.right_leg)
        self.env.add(self.left_leg)

        # Pelvis reference line
        self.env.ax.plot(
            [0, 0],
            [-HALF_PELVIS, HALF_PELVIS],
            [0, 0],
            linewidth=5,
        )

        self.env.ax.set_xlabel("X [mm]")
        self.env.ax.set_ylabel("Y [mm]")
        self.env.ax.set_zlabel("Z [mm]")

        self.env.step(0.001)

    def make_pattern_graphs(self):
        self.fig_right = Figure()
        self.canvas_right = FigureCanvas(self.fig_right)
        self.ax_right = self.fig_right.add_subplot(111)

        right_layout = QVBoxLayout(self.ui.widget_right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas_right)

        self.fig_left = Figure()
        self.canvas_left = FigureCanvas(self.fig_left)
        self.ax_left = self.fig_left.add_subplot(111)

        left_layout = QVBoxLayout(self.ui.widget_left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas_left)

    def draw_patterns(self):
        step_amp = self.ui.doubleSpinBox_step.value()
        swing_amp = self.ui.doubleSpinBox_swing.value()
        rise_amp = self.ui.doubleSpinBox_rise.value()

        phases = np.linspace(0.0, 1.0, 300)

        right_step = []
        right_swing = []
        right_rise = []
        left_step = []
        left_swing = []
        left_rise = []

        for phase in phases:
            step, swing, rise = self.pattern.values(phase)
            right_step.append(step_amp * step)
            right_swing.append(-swing_amp * swing)
            right_rise.append(rise_amp * rise)

            left_phase = (phase + 0.5) % 1.0
            step, swing, rise = self.pattern.values(left_phase)
            left_step.append(step_amp * step)
            left_swing.append(swing_amp * swing)
            left_rise.append(rise_amp * rise)

        self.ax_right.clear()
        self.ax_right.plot(phases, right_step, label="Step")
        self.ax_right.plot(phases, right_swing, label="Swing")
        self.ax_right.plot(phases, right_rise, label="Rise")
        self.ax_right.set_xlim(0.0, 1.0)
        self.ax_right.set_xlabel("Phase")
        self.ax_right.set_ylabel("Offset [mm]")
        self.ax_right.grid(True)
        self.ax_right.legend(loc="upper right")
        self.right_cursor = self.ax_right.axvline(self.phase, linestyle="--")

        self.ax_left.clear()
        self.ax_left.plot(phases, left_step, label="Step")
        self.ax_left.plot(phases, left_swing, label="Swing")
        self.ax_left.plot(phases, left_rise, label="Rise")
        self.ax_left.set_xlim(0.0, 1.0)
        self.ax_left.set_xlabel("Phase")
        self.ax_left.set_ylabel("Offset [mm]")
        self.ax_left.grid(True)
        self.ax_left.legend(loc="upper right")
        self.left_cursor = self.ax_left.axvline(self.phase, linestyle="--")

        self.fig_right.tight_layout()
        self.fig_left.tight_layout()
        self.canvas_right.draw_idle()
        self.canvas_left.draw_idle()

    def walk_on(self):
        if not self.timer.isActive():
            self.timer.start(int(FRAME_DT_SEC * 1000))

    def walk_stop(self):
        self.timer.stop()

    def init_robot(self):
        self.walk_stop()
        self.phase = 0.0

        self.right_leg.q = Q_INIT.copy()
        self.left_leg.q = Q_INIT.copy()

        self.ui.radioButton_both.setChecked(True)

        self.ui.doubleSpinBox_step.setValue(self.init_step)
        self.ui.doubleSpinBox_swing.setValue(self.init_swing)
        self.ui.doubleSpinBox_rise.setValue(self.init_rise)

        self.draw_patterns()
        self.home_view()
        self.env.step(0.001)

    def animation_step(self):
        try:
            step_amp = self.ui.doubleSpinBox_step.value() * CM_TO_MM
            swing_amp = self.ui.doubleSpinBox_swing.value() * CM_TO_MM
            rise_amp = self.ui.doubleSpinBox_rise.value() * CM_TO_MM

            # Right leg always moves.
            step_r, swing_r, rise_r = self.pattern.values(self.phase)
            target_r = self.init_right_foot + np.array([
                step_amp * step_r,
                -swing_amp * swing_r,
                rise_amp * rise_r,
            ])
            self.right_leg.q = LegIK.solve(*target_r, side="right")

            if self.ui.radioButton_both.isChecked():
                left_phase = (self.phase + 0.5) % 1.0
                step_l, swing_l, rise_l = self.pattern.values(left_phase)
                target_l = self.init_left_foot + np.array([
                    step_amp * step_l,
                    swing_amp * swing_l,
                    rise_amp * rise_l,
                ])
                self.left_leg.q = LegIK.solve(*target_l, side="left")
            else:
                self.left_leg.q = Q_INIT.copy()

            self.env.step(0.001)

            self.right_cursor.set_xdata([self.phase, self.phase])
            self.left_cursor.set_xdata([self.phase, self.phase])
            self.canvas_right.draw_idle()
            self.canvas_left.draw_idle()

            self.phase += FRAME_DT_SEC / GAIT_CYCLE_SEC
            self.phase %= 1.0

        except ValueError as error:
            self.walk_stop()
            QMessageBox.warning(self, "IK Error", str(error))


    def home_view(self):
        self.env.ax.view_init(elev=18, azim=-58)
        self.env.ax.figure.canvas.draw_idle()

    def front_view(self):
        self.env.ax.view_init(elev=0, azim=0)
        self.env.ax.figure.canvas.draw_idle()

    def side_view(self):
        self.env.ax.view_init(elev=0, azim=-90)
        self.env.ax.figure.canvas.draw_idle()

    def top_view(self):
        self.env.ax.view_init(elev=90, azim=-90)
        self.env.ax.figure.canvas.draw_idle()

    def closeEvent(self, event):
        self.walk_stop()

        try:
            self.env.close()
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
