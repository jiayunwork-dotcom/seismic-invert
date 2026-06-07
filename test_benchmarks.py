import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from src.quality_control import (
    BenchmarkScenarios, run_benchmark, InversionParams,
    get_all_benchmark_scenarios
)
import numpy as np
import sys

print("=" * 60)
print("地震反演基准测试验证")
print("=" * 60)

default_params = InversionParams(
    max_iterations=350,
    convergence_threshold=1e-13,
    regularization=0.02,
    inversion_type='traveltime',
    verbose=False
)

all_passed = True

scenarios = get_all_benchmark_scenarios()

for i, scenario in enumerate(scenarios):
    print(f"\n{'─' * 60}")
    print(f"场景 {i+1}: {scenario['name']}")
    print(f"说明: {scenario['description']}")
    print(f"初始模型范围: [{np.min(scenario['initial_model']):.1f}, {np.max(scenario['initial_model']):.1f}] m/s")
    print(f"真实模型范围: [{np.min(scenario['true_model']):.1f}, {np.max(scenario['true_model']):.1f}] m/s")
    print(f"观测时间范围: [{np.min(scenario['observed_times']):.4f}, {np.max(scenario['observed_times']):.4f}] s")
    print(f"网格大小: {scenario['true_model'].shape}")
    print(f"震源数量: {len(scenario['sources'])}, 接收点数量: {len(scenario['receivers'])}")
    print(f"{'─' * 60}")

    try:
        result = run_benchmark(scenario, default_params)

        print(f"\n📊 结果:")
        print(f"  状态: {'✅ 通过' if result.passed else '❌ 未通过'}")
        print(f"  {result.message}")
        print(f"  平均绝对误差: {result.absolute_error:.4f} m/s")
        print(f"  平均相对误差: {result.relative_error * 100:.4f} %")
        print(f"  迭代次数: {result.metrics['iterations']}")
        print(f"  最终目标函数: {result.metrics['final_objective']:.6e}")
        print(f"  收敛: {'是' if result.metrics['converged'] else '否'}")
        print(f"  反演模型范围: [{np.min(result.inverted_model):.2f}, {np.max(result.inverted_model):.2f}] m/s")

        if not result.passed:
            all_passed = False

        if result.scenario_type == 'homogeneous':
            avg_v = np.mean(result.inverted_model)
            expected_v = scenario['expected_velocity']
            rel_err = abs(avg_v - expected_v) / expected_v * 100
            tolerance = scenario['tolerance'] * 100
            print(f"\n🎯 验收标准: 相对误差 < {tolerance:.2f}%")
            print(f"   当前相对误差: {rel_err:.4f}% {'✓' if rel_err < tolerance else '✗'}")

        elif result.scenario_type == 'two_layer':
            interface_idx = int(scenario['interface_depth'] / scenario['dz'])
            v1_inv = np.mean(result.inverted_model[:interface_idx, :])
            v2_inv = np.mean(result.inverted_model[interface_idx:, :])
            v1_true = scenario['expected_velocities']['upper']
            v2_true = scenario['expected_velocities']['lower']
            err1 = abs(v1_inv - v1_true) / v1_true * 100
            err2 = abs(v2_inv - v2_true) / v2_true * 100
            tolerance = scenario['tolerance'] * 100
            print(f"\n🎯 验收标准: 各层相对误差 < {tolerance:.2f}%")
            print(f"   上层: 反演={v1_inv:.2f} m/s, 真实={v1_true:.2f} m/s, 误差={err1:.3f}% {'✓' if err1 < tolerance else '✗'}")
            print(f"   下层: 反演={v2_inv:.2f} m/s, 真实={v2_true:.2f} m/s, 误差={err2:.3f}% {'✓' if err2 < tolerance else '✗'}")

        elif result.scenario_type == 'gradient':
            z = np.arange(scenario['dz'], result.inverted_model.shape[0] * scenario['dz'], scenario['dz'])
            v_profile = np.mean(result.inverted_model, axis=1)[1:]
            if len(v_profile) > 1:
                grad_inv = float(np.polyfit(z, v_profile, 1)[0])
            else:
                grad_inv = 0.0
            grad_true = scenario['expected_gradient']
            rel_err = abs(grad_inv - grad_true) / abs(grad_true + 1e-10) * 100
            tolerance = scenario['tolerance'] * 100
            print(f"\n🎯 验收标准: 梯度相对误差 < {tolerance:.2f}%")
            print(f"   反演梯度: {grad_inv:.4f}/s")
            print(f"   真实梯度: {grad_true:.4f}/s")
            print(f"   相对误差: {rel_err:.3f}% {'✓' if rel_err < tolerance else '✗'}")

    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        all_passed = False

print(f"\n{'=' * 60}")
if all_passed:
    print("🎉 所有基准测试通过！")
else:
    print("⚠️ 部分基准测试未通过")
print(f"{'=' * 60}")

sys.exit(0 if all_passed else 1)
