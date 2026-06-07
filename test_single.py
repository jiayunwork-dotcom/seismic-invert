import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from src.quality_control import (
    BenchmarkScenarios, run_benchmark, InversionParams
)
import numpy as np

print("=" * 60)
print("单场景快速测试")
print("=" * 60)

default_params = InversionParams(
    max_iterations=350,
    convergence_threshold=1e-13,
    regularization=0.02,
    inversion_type='traveltime',
    verbose=True
)

scenario = BenchmarkScenarios.create_homogeneous()
print(f"\n场景: {scenario['name']}")
print(f"震源: {len(scenario['sources'])} 个")
print(f"接收点: {len(scenario['receivers'])} 个")
print(f"初始模型: [{np.min(scenario['initial_model']):.1f}, {np.max(scenario['initial_model']):.1f}]")
print(f"真实模型: [{np.min(scenario['true_model']):.1f}, {np.max(scenario['true_model']):.1f}]")

print("\n开始反演...")
result = run_benchmark(scenario, default_params)

print(f"\n结果:")
print(f"  状态: {'通过' if result.passed else '未通过'}")
print(f"  {result.message}")
print(f"  迭代次数: {result.metrics['iterations']}")
print(f"  反演模型范围: [{np.min(result.inverted_model):.2f}, {np.max(result.inverted_model):.2f}]")
print(f"  目标函数: {result.metrics['final_objective']:.2e}")
