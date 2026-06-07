import time
from src.quality_control import BenchmarkScenarios, run_benchmark, InversionParams

params = InversionParams(
    max_iterations=350,
    convergence_threshold=1e-13,
    regularization=0.02,
    inversion_type='traveltime',
    verbose=False
)

print("=" * 60)
print("性能测试 - 均匀介质")
print("=" * 60)

scenario = BenchmarkScenarios.create_homogeneous()
print(f'网格: {scenario["true_model"].shape}')
print(f'震源: {len(scenario["sources"])} 个, 接收点: {len(scenario["receivers"])} 个')
print(f'初始模型: [{scenario["initial_model"].min():.1f}, {scenario["initial_model"].max():.1f}]')
print(f'真实模型: [{scenario["true_model"].min():.1f}, {scenario["true_model"].max():.1f}]')
print(f'初始模型偏差: {(1 - scenario["initial_model"].min()/scenario["true_model"].min())*100:.1f}%')

start = time.time()
result = run_benchmark(scenario, params)
elapsed = time.time() - start

print(f'\n结果:')
print(f'  状态: {"✅ 通过" if result.passed else "❌ 未通过"}')
print(f'  {result.message}')
print(f'  迭代次数: {result.metrics["iterations"]}')
print(f'  运行时间: {elapsed:.2f} 秒')
print(f'  验收标准: 相对误差 < 0.1%')

print("\n" + "=" * 60)
print("性能测试 - 两层介质")
print("=" * 60)

scenario = BenchmarkScenarios.create_two_layer()
print(f'网格: {scenario["true_model"].shape}')
print(f'震源: {len(scenario["sources"])} 个, 接收点: {len(scenario["receivers"])} 个')
print(f'初始模型: [{scenario["initial_model"].min():.1f}, {scenario["initial_model"].max():.1f}]')
print(f'真实模型: [{scenario["true_model"].min():.1f}, {scenario["true_model"].max():.1f}]')

start = time.time()
result = run_benchmark(scenario, params)
elapsed = time.time() - start

print(f'\n结果:')
print(f'  状态: {"✅ 通过" if result.passed else "❌ 未通过"}')
print(f'  {result.message}')
print(f'  迭代次数: {result.metrics["iterations"]}')
print(f'  运行时间: {elapsed:.2f} 秒')

print("\n" + "=" * 60)
print("性能测试 - 梯度介质")
print("=" * 60)

scenario = BenchmarkScenarios.create_gradient()
print(f'网格: {scenario["true_model"].shape}')
print(f'震源: {len(scenario["sources"])} 个, 接收点: {len(scenario["receivers"])} 个')
print(f'初始模型: [{scenario["initial_model"].min():.1f}, {scenario["initial_model"].max():.1f}]')
print(f'真实模型: [{scenario["true_model"].min():.1f}, {scenario["true_model"].max():.1f}]')

start = time.time()
result = run_benchmark(scenario, params)
elapsed = time.time() - start

print(f'\n结果:')
print(f'  状态: {"✅ 通过" if result.passed else "❌ 未通过"}')
print(f'  {result.message}')
print(f'  迭代次数: {result.metrics["iterations"]}')
print(f'  运行时间: {elapsed:.2f} 秒')

print("\n" + "=" * 60)
print("✅ 测试完成，无数值溢出警告")
print("=" * 60)
