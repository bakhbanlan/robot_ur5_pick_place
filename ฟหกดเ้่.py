import numpy as np
from scipy.optimize import fsolve

# พารามิเตอร์ที่ให้มา
R_s = 0.1       # ความต้านทานสเตเตอร์ (ohm)
R_r_prime = 0.1 # ความต้านทานโรเตอร์ (ohm)
X_eq = 0.5      # รีแอคแตนซ์เทียบเท่า (ohm)
s_original = 0.02  # Slip เดิม

# ฟังก์ชันคำนวณ T_motor / V^2
def T_motor_per_V2(s):
    return (R_r_prime / s) / ((R_s + R_r_prime / s)**2 + X_eq**2)

# สมการที่ต้องแก้เพื่อหา s_new
def equation(s_new):
    T_ratio = T_motor_per_V2(s_new) / T_motor_per_V2(s_original)
    voltage_ratio_squared = (0.8)**2  # (V_new / V_original)^2
    load_ratio_squared = ((1 - s_new) / (1 - s_original))**2
    return T_ratio * voltage_ratio_squared - load_ratio_squared

# หาค่า s_new โดยใช้ fsolve
s_new_initial_guess = s_original  # ค่าเริ่มต้นสำหรับการหาคำตอบ
s_new = fsolve(equation, s_new_initial_guess)[0]

# คำนวณความเร็วใหม่เป็นเปอร์เซ็นต์ของความเร็วเดิม
n_new_percent = (1 - s_new) / (1 - s_original) * 100

# แสดงผลลัพธ์
print(f"ความเร็วใหม่เป็น {n_new_percent:.2f}% ของความเร็วเดิม")