import sys
import math

import roboticstoolbox as rtb
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from ui_axis_2 import Ui_Form

L1 = 1.0
L2 = 1.0

INIT_THETA1 = -45
INIT_THETA2 = -90

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.Btn_FK.clicked.connect(self.forward_kinematic)
        self.ui.Btn_IK.clicked.connect(self.inverse_kinematic)
        self.ui.Btn_Pattern.clicked.connect(self.pattern_on)
        self.ui.Btn_Init.clicked.connect(self.initialize)
        self.ui.Btn_Quit.clicked.connect(self.quit)

        link1 = rtb.RevoluteDH(d=0, a=L1, alpha=0)
        link2 = rtb.RevoluteDH(d=0, a=L2, alpha=0)

        self.ROBOT = rtb.DHRobot(
            [link1, link2],
            name="leg"
        )

        # 로봇 그래프는 딱 한 번 생성
        self.env = self.ROBOT.plot(
            [math.radians(INIT_THETA1), math.radians(INIT_THETA2)],
            backend="pyplot",
            block=False,
            jointaxes=False,
            eeframe=False,
            shadow=False,
            limits=[-3, 3, -3, 3, -1, 1]
        )
        # 시점 설정
        ax = self.env.ax
        ax.view_init(elev=90, azim=-90)
        self.body_line, = self.env.ax.plot(
            [0, 0],
            [0, 1],
            linewidth=5,
        )

    def initialize(self):
        self.ui.lineEdit_theta1.setText(str(INIT_THETA1))
        self.ui.lineEdit_theta2.setText(str(INIT_THETA2))

        self.forward_kinematic()

    def forward_kinematic(self):
        theta1 = math.radians(float(self.ui.lineEdit_theta1.text()))
        theta2 = math.radians(float(self.ui.lineEdit_theta2.text()))

        self.ROBOT.q = [theta1, theta2]
        self.env.step(0.05)

        x = L1*math.cos(theta1) + L2*math.cos(theta1 + theta2)
        y = L1*math.sin(theta1) + L2*math.sin(theta1 + theta2)

        self.ui.lineEdit_x_coordi.setText(f"{x:.2f}")
        self.ui.lineEdit_y_coordi.setText(f"{y:.2f}")

    def calc_angle(self, coordi_x, coordi_y):
        # 두번째 각도
        cos_b = (coordi_x**2 + coordi_y**2 - L1**2 - L2**2)/(2*L1*L2)
        cos_b = max(-1.0, min(1.0, cos_b))

        sin_b = -math.sqrt(1-cos_b**2)
        # tan_b = sin_b/cos_b
        # theta2 = math.atan(tan_b)
        theta2 = math.atan2(sin_b, cos_b)
        # 첫번째 각도
        k1 = L1 + L2*cos_b
        k2 = L2*sin_b
        cos_a = (k1*coordi_x + k2*coordi_y) / (k1**2 + k2**2)
        sin_a = -math.sqrt(1-cos_a**2) # sin_a = (y*k1 - x*k2) / (k1**2 + k2**2)
        theta1 = math.atan2(sin_a, cos_a)

        return theta1, theta2

    def inverse_kinematic(self):
        x = float(self.ui.lineEdit_x_coordi.text())
        y = float(self.ui.lineEdit_y_coordi.text())

        theta1, theta2 = self.calc_angle(x, y)

        self.ROBOT.q = [theta1, theta2]
        self.env.step(0.05)

        self.ui.lineEdit_theta1.setText(f"{math.degrees(theta1):.2f}")
        self.ui.lineEdit_theta2.setText(f"{math.degrees(theta2):.2f}")

    def check_input(self, x, y, move_x, move_y):
        end_x = x + move_x
        end_y = y + move_y

        max_reach = L1 + L2
        min_reach = abs(L1 - L2)

        start_r = math.hypot(x, y)
        end_r = math.hypot(end_x, end_y)

        # 시작점 검사
        if not (min_reach <= start_r <= max_reach):
            QMessageBox.warning(
                self,
                "Pattern Error",
                "현재 X, Y 좌표가 작업 영역 밖에 있습니다."
            )
            return

        # 패턴 끝점 검사
        if not (min_reach <= end_r <= max_reach):
            QMessageBox.warning(
                self,
                "Pattern Error",
                (
                    "입력한 이동량으로는 패턴을 수행할 수 없습니다.\n\n"
                    f"Target: ({end_x:.3f}, {end_y:.3f})\n"
                    f"Distance: {end_r:.3f}\n"
                    f"Max reach: {max_reach:.3f}"
                )
            )
            return

    def pattern_on(self):
        x = float(self.ui.lineEdit_x_coordi.text())
        y = float(self.ui.lineEdit_y_coordi.text())

        move_x = float(self.ui.lineEdit_x_move.text())
        move_y = float(self.ui.lineEdit_y_move.text())

        self.check_input(x, y, move_x, move_y)

        for i in range(33):
            time = i * math.pi / 16

            nowIK_x = x + move_x*abs(math.sin(time))
            nowIK_y = y + move_y*abs(math.sin(time))

            theta1, theta2 = self.calc_angle(nowIK_x, nowIK_y)

            self.ROBOT.q = [theta1, theta2]
            self.env.step(0.05)

            self.ui.lineEdit_theta1.setText(f"{math.degrees(theta1):.2f}")
            self.ui.lineEdit_theta2.setText(f"{math.degrees(theta2):.2f}")
            self.ui.lineEdit_x_coordi.setText(f"{nowIK_x:.2f}")
            self.ui.lineEdit_y_coordi.setText(f"{nowIK_y:.2f}")

            QApplication.processEvents()

        theta1, theta2 = self.calc_angle(x, y)

        self.ROBOT.q = [theta1, theta2]
        self.env.step(0.05)

        self.ui.lineEdit_theta1.setText(f"{math.degrees(theta1):.2f}")
        self.ui.lineEdit_theta2.setText(f"{math.degrees(theta2):.2f}")
        self.ui.lineEdit_x_coordi.setText(f"{x:.2f}")
        self.ui.lineEdit_y_coordi.setText(f"{y:.2f}")

    def quit(self):
        self.close()

    def closeEvent(self, event):
        self.env.close()
        plt.close("all")
        event.accept()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())