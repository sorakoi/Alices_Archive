import torch
import time
import sys


def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_basic_gpu_info():
    """测试基本 GPU 信息"""
    print_separator("1. 基本 GPU 信息")

    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA 可用: {cuda_available}")

    if not cuda_available:
        print("❌ CUDA 不可用，测试终止")
        return False

    # CUDA 版本信息
    print(f"✅ PyTorch CUDA 版本: {torch.version.cuda}")
    print(f"✅ cuDNN 版本: {torch.backends.cudnn.version()}")

    # GPU 数量和信息
    gpu_count = torch.cuda.device_count()
    print(f"✅ 可用 GPU 数量: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {gpu_name}")
        print(f"     - 显存总量: {gpu_props.total_memory / 1024 ** 3:.2f} GB")
        print(f"     - 计算能力: {gpu_props.major}.{gpu_props.minor}")
        print(f"     - 多处理器数量: {gpu_props.multi_processor_count}")

    return True


def test_gpu_tensor_operations():
    """测试 GPU 张量操作"""
    print_separator("2. GPU 张量操作测试")

    # 创建张量并移动到 GPU
    device = torch.device('cuda')

    # 基本张量操作
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)

    print(f"✅ 张量创建成功")
    print(f"   x 形状: {x.shape}, 设备: {x.device}")
    print(f"   y 形状: {y.shape}, 设备: {y.device}")

    # 各种运算测试
    z_add = x + y
    z_matmul = torch.matmul(x, y)
    z_sum = torch.sum(x)

    print(f"✅ 基本运算测试通过")
    print(f"   加法结果形状: {z_add.shape}")
    print(f"   矩阵乘法形状: {z_matmul.shape}")
    print(f"   求和结果: {z_sum.item():.4f}")

    return True


def test_gpu_performance():
    """测试 GPU 性能"""
    print_separator("3. GPU 性能测试")

    device = torch.device('cuda')

    # 创建更大的张量进行性能测试
    size = 5000
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)

    # 预热 GPU
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()  # 等待 GPU 完成

    # 计时矩阵乘法
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # 确保计算完成
    end_time = time.time()

    gpu_time = end_time - start_time

    print(f"✅ GPU 矩阵乘法测试完成")
    print(f"   矩阵大小: {size} x {size}")
    print(f"   GPU 计算时间: {gpu_time:.4f} 秒")

    # 对比 CPU 性能（可选）
    if torch.cuda.device_count() == 0:  # 如果没有 GPU，跳过对比
        return True

    print("\n--- 与 CPU 对比 ---")
    a_cpu = a.cpu()
    b_cpu = b.cpu()

    start_time_cpu = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    end_time_cpu = time.time()

    cpu_time = end_time_cpu - start_time_cpu
    print(f"   CPU 计算时间: {cpu_time:.4f} 秒")
    print(f"   GPU 加速比: {cpu_time / gpu_time:.2f}x")

    return True


def test_gpu_memory():
    """测试 GPU 内存管理"""
    print_separator("4. GPU 内存管理测试")

    if not torch.cuda.is_available():
        return False

    # 初始内存状态
    initial_allocated = torch.cuda.memory_allocated()
    initial_cached = torch.cuda.memory_reserved()

    print(f"初始显存使用: {initial_allocated / 1024 ** 2:.2f} MB")
    print(f"初始缓存显存: {initial_cached / 1024 ** 2:.2f} MB")

    # 分配大块内存
    device = torch.device('cuda')
    large_tensor = torch.randn(5000, 5000).to(device)

    allocated_after = torch.cuda.memory_allocated()
    cached_after = torch.cuda.memory_reserved()

    print(f"分配后显存使用: {allocated_after / 1024 ** 2:.2f} MB")
    print(f"分配后缓存显存: {cached_after / 1024 ** 2:.2f} MB")

    # 清理
    del large_tensor
    torch.cuda.empty_cache()

    final_allocated = torch.cuda.memory_allocated()
    final_cached = torch.cuda.memory_reserved()

    print(f"清理后显存使用: {final_allocated / 1024 ** 2:.2f} MB")
    print(f"清理后缓存显存: {final_cached / 1024 ** 2:.2f} MB")

    print("✅ GPU 内存管理测试通过")
    return True


def test_neural_network_gpu():
    """测试神经网络在 GPU 上的运行"""
    print_separator("5. 神经网络 GPU 测试")

    if not torch.cuda.is_available():
        return False

    device = torch.device('cuda')

    # 创建一个简单的 CNN 模型
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = torch.nn.Linear(64 * 8 * 8, 256)
            self.fc2 = torch.nn.Linear(256, 10)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 创建模型并移动到 GPU
    model = SimpleCNN().to(device)
    print(f"✅ 模型创建成功，设备: {next(model.parameters()).device}")

    # 创建模拟数据
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_target = torch.randint(0, 10, (batch_size,)).to(device)

    # 前向传播
    output = model(dummy_input)
    print(f"✅ 前向传播成功")
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")

    # 计算损失
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)
    print(f"✅ 损失计算成功: {loss.item():.4f}")

    # 反向传播
    model.zero_grad()
    loss.backward()
    print("✅ 反向传播成功")

    # 检查梯度
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"✅ 梯度计算: {'成功' if has_gradients else '失败'}")

    return True


def main():
    """主测试函数"""
    print_separator("PyTorch GPU 全面测试")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")

    all_tests_passed = True

    # 运行所有测试
    tests = [
        test_basic_gpu_info,
        test_gpu_tensor_operations,
        test_gpu_performance,
        test_gpu_memory,
        test_neural_network_gpu
    ]

    for test in tests:
        try:
            result = test()
            if not result:
                all_tests_passed = False
                print(f"❌ {test.__name__} 失败")
        except Exception as e:
            print(f"❌ {test.__name__} 出错: {e}")
            all_tests_passed = False

    print_separator("测试总结")
    if all_tests_passed:
        print("🎉 所有 GPU 测试通过！PyTorch GPU 版本安装成功！")
        print("您现在可以正常使用 GPU 进行深度学习训练了。")
    else:
        print("⚠️  部分测试失败，请检查 PyTorch GPU 安装。")
        print("可能的问题：")
        print("  - CUDA 驱动版本不匹配")
        print("  - PyTorch 版本与 CUDA 版本不兼容")
        print("  - GPU 内存不足")
        print("  - 系统权限问题")

    return all_tests_passed


if __name__ == "__main__":
    main()
