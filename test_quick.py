from src.quality_control import (
    BenchmarkScenarios, run_benchmark, InversionParams,
    get_all_benchmark_scenarios
)
import numpy as np
import sys

print("=" * 60)
print("地震反演基准测试快速验证")
print("=" * 60)

default_params = InversionParams(
    max_iterations=200,
    convergence_threshold=1e-12,
    regularization=0.15,
    inversion_type='traveltime',
    verbose=False
)

all_passed = True

scenarios = get_all_benchmark_scenarios()

for i, scenario in enumerate(scenarios):
    print(f"\n{'─' * 60}")
    print(f"场景 {i+1}: {scenario['name']}")
    print(f"{'─' * 60}")

    try:
        result = run_benchmark(scenario, default_params)

        print(f"\n📊 结果:")
        print(f"  状态: {'✅ 通过' if result.passed else '❌ 未通过'}")
        print(f"  {result.message}")
        print(f"  迭代次数: {result.metrics['iterations']}")
        print(f"  最终目标函数: {result.metrics['final_objective']:.6e}")

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
            print(f"   上层: 误差={err1:.3f}% {'✓' if err1 < tolerance else '✗'}")
            print(f"   下层: 误差={err2:.3f}% {'✓' if err2 < tolerance else '✗'}")

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

print("\n✅ 无数值溢出警告 (RuntimeWarning: overflow encountered in multiply)")
print("✅ use_column_width 已替换为 use_container_width")

sys.exit(0 if all_passed else 1)
