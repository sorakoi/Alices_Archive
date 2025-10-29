# 用来生成遥感的波长

import numpy as np

# AVIRIS传感器参数
start_wavelength_nm = 400.0  # 起始波长 0.4 μm = 400 nm
end_wavelength_nm = 2500.0   # 结束波长 2.5 μm = 2500 nm
num_bands = 224              # 总波段数

# 生成均匀分布的波长数组（纳米）
wavelengths_nm = np.linspace(start_wavelength_nm, end_wavelength_nm, num_bands)

# 保留6位小数
wavelengths_nm = np.round(wavelengths_nm, 6)

print("AVIRIS传感器224个波段的中心波长（纳米）：")
print("wavelengths = [")
for i, wl in enumerate(wavelengths_nm):
    if i < len(wavelengths_nm) - 1:
        print(f"    {wl:.6f},")  # 保留6位小数
    else:
        print(f"    {wl:.6f}")   # 最后一个元素不加逗号
print("]")

# 同时生成剔除水吸收波段后的可用波长数组
bad_bands_indices = list(range(107, 112)) + list(range(153, 167)) + [223]  # 0-based索引
good_bands_mask = np.ones(num_bands, dtype=bool)
good_bands_mask[bad_bands_indices] = False

usable_wavelengths = wavelengths_nm[good_bands_mask]

print(f"\n剔除20个水吸收波段后，剩余{len(usable_wavelengths)}个可用波段的波长：")
print("usable_wavelengths = [")
for i, wl in enumerate(usable_wavelengths):
    if i < len(usable_wavelengths) - 1:
        print(f"    {wl:.6f},")
    else:
        print(f"    {wl:.6f}")
print("]")

# 验证波长范围和间隔
print(f"\n验证信息：")
print(f"总波段数: {len(wavelengths_nm)}")
print(f"波长范围: {wavelengths_nm[0]:.2f} - {wavelengths_nm[-1]:.2f} nm")
print(f"平均波长间隔: {(wavelengths_nm[1] - wavelengths_nm[0]):.6f} nm")
print(f"可用波段数: {len(usable_wavelengths)}")
