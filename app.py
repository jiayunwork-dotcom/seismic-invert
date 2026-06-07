import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_io import parse_segy, parse_csv, compute_statistics, reorder_gathers
from src.velocity_model import (
    VelocityModel, LayeredModel, GradientModel, GridModel,
    create_default_model, convert_model
)
from src.forward_modeling import ForwardParams, run_forward, compute_residual, check_stability
from src.travel_time import (
    FastMarching, trace_ray_layered, trace_ray_grid,
    compute_travel_times, RayPath
)
from src.inversion import InversionParams, InversionResult, run_inversion
from src.frequency_domain import (
    FilterParams, DeconvolutionParams,
    compute_spectrum, compute_average_spectrum,
    apply_filter_to_traces, apply_deconvolution_to_traces, process_traces
)
from src.stacking import (
    NMOParams, VelocitySpectrumParams,
    nmo_correct_gather, compute_velocity_spectrum,
    process_cdp_gather, build_velocity_function
)
from src.visualization import (
    plot_velocity, plot_seismic_wiggle, plot_seismic_image,
    plot_spectrum, plot_travel_time_contours, plot_wavefield_snapshot,
    plot_inversion_result, plot_velocity_spectrum, plot_comparison,
    figure_to_bytes
)

st.set_page_config(
    page_title="地震波形反演与地层结构成像工具",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'seismic_data' not in st.session_state:
    st.session_state.seismic_data = None

if 'velocity_model' not in st.session_state:
    st.session_state.velocity_model = None

if 'forward_result' not in st.session_state:
    st.session_state.forward_result = None

if 'travel_time_result' not in st.session_state:
    st.session_state.travel_time_result = None

if 'inversion_result' not in st.session_state:
    st.session_state.inversion_result = None

if 'process_result' not in st.session_state:
    st.session_state.process_result = None

if 'stacking_result' not in st.session_state:
    st.session_state.stacking_result = None

st.title("🌍 地震波形反演与地层结构成像工具")
st.markdown("---")

with st.sidebar:
    st.header("功能导航")
    page = st.radio(
        "选择功能模块",
        [
            "📊 数据导入与管理",
            "📐 速度模型定义",
            "🔊 正演模拟",
            "⏱️ 旅行时计算",
            "🔄 反演算法",
            "📈 频率域处理",
            "📡 叠加处理",
            "🎨 可视化导出"
        ],
        index=0
    )

if page == "📊 数据导入与管理":
    st.header("📊 数据导入与管理")
    
    tab1, tab2 = st.tabs(["数据导入", "道集展示与统计"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("文件导入")
            file_type = st.selectbox("文件格式", ["SEG-Y (.sgy, .segy)", "CSV (.csv)"])
            uploaded_file = st.file_uploader("选择地震数据文件", type=["sgy", "segy", "csv"])
            
            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    
                    with st.spinner("正在解析文件..."):
                        if file_type.startswith("SEG-Y"):
                            data = parse_segy(file_bytes)
                        else:
                            data = parse_csv(file_bytes)
                        
                        data = compute_statistics(data)
                        st.session_state.seismic_data = data
                    
                    st.success(f"文件解析成功！共 {data['n_traces']} 道，每道 {data['n_samples']} 个采样点")
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("采样率", f"{data['sample_interval']*1000:.2f} ms")
                    with col_info2:
                        st.metric("记录长度", f"{data['record_length']*1000:.1f} ms")
                    with col_info3:
                        st.metric("编码方式", data.get('encoding', 'unknown').upper())
                
                except Exception as e:
                    st.error(f"文件解析错误: {str(e)}")
        
        with col2:
            st.subheader("示例数据")
            if st.button("生成教学演示数据", type="primary"):
                with st.spinner("正在生成示例数据..."):
                    n_traces = 48
                    n_samples = 500
                    dt = 0.002
                    
                    t = np.arange(n_samples) * dt
                    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
                    
                    f0 = 30
                    t0 = 0.05
                    tau = np.pi * f0 * (t - t0)
                    wavelet = (1 - 2 * tau**2) * np.exp(-tau**2)
                    
                    for i in range(n_traces):
                        offset = i * 10
                        travel_time = np.sqrt((offset / 2000)**2 + 0.1**2)
                        sample_idx = int(travel_time / dt)
                        if sample_idx < n_samples:
                            traces[i, sample_idx:] = wavelet[:n_samples - sample_idx]
                    
                    csv_data = {
                        'format': 'csv',
                        'traces': traces,
                        'n_traces': n_traces,
                        'n_samples': n_samples,
                        'sample_interval': dt,
                        'record_length': n_samples * dt,
                        'time_samples': t,
                        'shot_numbers': np.arange(n_traces),
                        'cdp_numbers': np.arange(n_traces),
                        'offsets': np.arange(n_traces) * 10
                    }
                    csv_data = compute_statistics(csv_data)
                    st.session_state.seismic_data = csv_data
                
                st.success("示例数据已生成！")
            
            if st.button("清空当前数据"):
                st.session_state.seismic_data = None
                st.info("数据已清空")
        
        if st.session_state.seismic_data is not None:
            data = st.session_state.seismic_data
            
            st.subheader("文件头信息")
            if data['format'] == 'segy':
                with st.expander("查看二进制头信息"):
                    bh = data['binary_header']
                    bh_df = pd.DataFrame([
                        {"参数": k, "值": v} for k, v in bh.items()
                    ])
                    st.dataframe(bh_df, hide_index=True)
                
                with st.expander("查看道头信息（前10道）"):
                    th_list = data['trace_headers'][:10]
                    th_df = pd.DataFrame(th_list)
                    st.dataframe(th_df, hide_index=True)
            
            with st.expander("查看坐标信息"):
                coords = data.get('coordinates', {})
                if coords:
                    coord_df = pd.DataFrame({
                        '震源X': coords.get('sx', [])[:20],
                        '震源Y': coords.get('sy', [])[:20],
                        '接收点X': coords.get('gx', [])[:20],
                        '接收点Y': coords.get('gy', [])[:20],
                        '炮检距': coords.get('offset', [])[:20]
                    })
                    st.dataframe(coord_df, hide_index=True)
    
    with tab2:
        if st.session_state.seismic_data is not None:
            data = st.session_state.seismic_data
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            stats = data.get('statistics', {})
            with col_stat1:
                st.metric("最大振幅", f"{stats.get('max_amplitude', 0):.4f}")
            with col_stat2:
                st.metric("频谱主频", f"{stats.get('dominant_frequency', 0):.1f} Hz")
            with col_stat3:
                st.metric("平均频率", f"{stats.get('average_frequency', 0):.1f} Hz")
            with col_stat4:
                snr = stats.get('snr_db', 0)
                if np.isinf(snr):
                    st.metric("信噪比", "∞ dB")
                else:
                    st.metric("信噪比", f"{snr:.1f} dB")
            
            st.subheader("道集重排与展示")
            
            gather_type = st.selectbox(
                "道集类型",
                ["原始道集", "炮集 (Shot)", "共中心点集 (CDP)", "共偏移距集 (Offset)"]
            )
            
            display_mode = st.radio("显示方式", ["变面积图", "Wiggle 波形图"], horizontal=True)
            
            if gather_type == "原始道集":
                traces = data['traces']
                time_axis = np.arange(data['n_samples']) * data['sample_interval']
                title = "原始地震记录"
            else:
                gather_key = gather_type.split()[0].lower()
                if gather_key == "炮":
                    gather_key = "shot"
                elif gather_key == "共":
                    gather_key = "cdp"
                else:
                    gather_key = "offset"
                
                with st.spinner("正在重排道集..."):
                    gathers, mask, keys = reorder_gathers(data, gather_key)
                
                gather_idx = st.slider("选择道集索引", 0, len(keys) - 1, 0)
                traces = gathers[gather_idx, :, :]
                valid_count = np.sum(mask[gather_idx, :])
                traces = traces[:valid_count, :]
                time_axis = np.arange(data['n_samples']) * data['sample_interval']
                title = f"{gather_type.split()[0]}集 - {keys[gather_idx]}"
            
            if traces.shape[0] > 0:
                with st.spinner("正在生成图形..."):
                    if display_mode == "变面积图":
                        fig = plot_seismic_image(
                            traces, time_axis,
                            cmap='seismic',
                            title=title
                        )
                    else:
                        fig = plot_seismic_wiggle(
                            traces, time_axis,
                            title=title
                        )
                    
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                    
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        st.download_button(
                            "下载PNG图片",
                            buf,
                            file_name=f"{title.replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    with col_exp2:
                        pdf_buf = figure_to_bytes(fig, format='pdf')
                        st.download_button(
                            "下载PDF图片",
                            pdf_buf,
                            file_name=f"{title.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
        else:
            st.info("请先导入或生成地震数据")

elif page == "📐 速度模型定义":
    st.header("📐 速度模型定义")
    
    tab1, tab2, tab3 = st.tabs(["层状模型", "渐变模型", "自定义网格模型"])
    
    model_params = st.sidebar.expander("模型参数", expanded=True)
    with model_params:
        nx = st.number_input("水平网格数 (nx)", 10, 500, 100, 10)
        nz = st.number_input("垂直网格数 (nz)", 10, 500, 80, 10)
        dx = st.number_input("水平网格间距 (m)", 5.0, 100.0, 10.0, 5.0)
        dz = st.number_input("垂直网格间距 (m)", 5.0, 100.0, 10.0, 5.0)
        colormap = st.selectbox("色标", ["viridis", "jet", "seismic"], 0)
        add_contours = st.checkbox("叠加等值线", False)
    
    with tab1:
        st.subheader("层状模型")
        
        if 'layers' not in st.session_state:
            st.session_state.layers = [
                {'depth': 200, 'velocity': 1500},
                {'depth': 500, 'velocity': 2200},
                {'depth': 800, 'velocity': 3000},
                {'depth': 1200, 'velocity': 3800}
            ]
        
        col_add1, col_add2, col_add3 = st.columns(3)
        with col_add1:
            new_depth = st.number_input("新层深度 (m)", 0.0, 10000.0, 600.0, 10.0)
        with col_add2:
            new_velocity = st.number_input("新层速度 (m/s)", 1000.0, 8000.0, 2500.0, 100.0)
        with col_add3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("添加层", type="primary"):
                st.session_state.layers.append({'depth': new_depth, 'velocity': new_velocity})
                st.success("层已添加！")
        
        st.subheader("层参数")
        layers_df = pd.DataFrame(st.session_state.layers)
        if not layers_df.empty:
            layers_df = layers_df.sort_values('depth').reset_index(drop=True)
            layers_df.index = layers_df.index + 1
            layers_df.columns = ['深度 (m)', '速度 (m/s)']
            
            edited_df = st.data_editor(
                layers_df,
                num_rows="dynamic",
                use_container_width=True
            )
            
            col_apply, col_reset = st.columns(2)
            with col_apply:
                if st.button("应用层状模型", type="primary"):
                    layers_list = [
                        {'depth': float(r['深度 (m)']), 'velocity': float(r['速度 (m/s)'])}
                        for _, r in edited_df.iterrows()
                    ]
                    st.session_state.layers = layers_list
                    model = LayeredModel(nx, nz, dx, dz, layers_list)
                    st.session_state.velocity_model = model
                    st.success("层状模型已应用！")
            with col_reset:
                if st.button("重置为默认层"):
                    st.session_state.layers = [
                        {'depth': 200, 'velocity': 1500},
                        {'depth': 500, 'velocity': 2200},
                        {'depth': 800, 'velocity': 3000},
                        {'depth': 1200, 'velocity': 3800}
                    ]
                    st.success("已重置")
    
    with tab2:
        st.subheader("渐变模型")
        
        col_grad1, col_grad2, col_grad3 = st.columns(3)
        with col_grad1:
            v_top = st.number_input("顶部速度 (m/s)", 1000.0, 8000.0, 1500.0, 100.0)
        with col_grad2:
            v_bottom = st.number_input("底部速度 (m/s)", 1000.0, 8000.0, 4000.0, 100.0)
        with col_grad3:
            gradient_type = st.selectbox("梯度类型", ["linear", "exponential"])
        
        if st.button("应用渐变模型", type="primary"):
            model = GradientModel(nx, nz, dx, dz, v_top, v_bottom, gradient_type)
            st.session_state.velocity_model = model
            st.success("渐变模型已应用！")
    
    with tab3:
        st.subheader("自定义网格模型")
        
        if st.session_state.velocity_model is None or not isinstance(st.session_state.velocity_model, GridModel):
            default_v = st.number_input("默认速度 (m/s)", 1000.0, 8000.0, 2000.0, 100.0)
            if st.button("创建空白网格模型", type="primary"):
                model = GridModel(nx, nz, dx, dz, default_v)
                st.session_state.velocity_model = model
                st.success("网格模型已创建！")
        else:
            model = st.session_state.velocity_model
            if not isinstance(model, GridModel):
                st.warning("当前模型不是网格模型，请先转换")
            else:
                st.subheader("编辑工具")
                
                edit_mode = st.radio("编辑模式", ["画笔", "区域填充", "渐变填充"], horizontal=True)
                
                if edit_mode == "画笔":
                    col_brush1, col_brush2, col_brush3 = st.columns(3)
                    with col_brush1:
                        paint_v = st.number_input("绘制速度 (m/s)", 1000.0, 8000.0, 3000.0, 100.0)
                    with col_brush2:
                        brush_radius = st.slider("画笔半径 (网格)", 1, 10, 2)
                    with col_brush3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("平滑模型"):
                            model.smooth(radius=2)
                            st.success("模型已平滑！")
                    
                    x_paint = st.slider("X位置", 0, nx - 1, nx // 2)
                    z_paint = st.slider("Z位置", 0, nz - 1, nz // 2)
                    
                    if st.button("绘制", type="primary"):
                        model.paint_velocity(x_paint, z_paint, paint_v, brush_radius)
                        st.success("已绘制！")
                
                elif edit_mode == "区域填充":
                    col_fill1, col_fill2 = st.columns(2)
                    with col_fill1:
                        x1 = st.slider("X起始", 0, nx - 1, 20)
                        x2 = st.slider("X结束", 0, nx - 1, 80)
                        fill_v = st.number_input("填充速度 (m/s)", 1000.0, 8000.0, 3500.0, 100.0)
                    with col_fill2:
                        z1 = st.slider("Z起始", 0, nz - 1, 30)
                        z2 = st.slider("Z结束", 0, nz - 1, 60)
                    
                    if st.button("填充区域", type="primary"):
                        model.fill_region(x1, x2, z1, z2, fill_v)
                        st.success("区域已填充！")
                
                else:
                    col_grad1, col_grad2 = st.columns(2)
                    with col_grad1:
                        v_top_fill = st.number_input("顶部填充速度 (m/s)", 1000.0, 8000.0, 1500.0, 100.0)
                        v_bottom_fill = st.number_input("底部填充速度 (m/s)", 1000.0, 8000.0, 4000.0, 100.0)
                    with col_grad2:
                        grad_type = st.selectbox("填充梯度类型", ["linear", "exponential"])
                    
                    if st.button("渐变填充整个模型", type="primary"):
                        model.fill_between_velocities(v_top_fill, v_bottom_fill, grad_type)
                        st.success("渐变填充完成！")
    
    st.markdown("---")
    st.subheader("模型操作")
    
    col_conv1, col_conv2, col_conv3 = st.columns(3)
    
    with col_conv1:
        target_type = st.selectbox(
            "转换为",
            ["grid", "layered"],
            format_func=lambda x: "网格模型" if x == "grid" else "层状模型"
        )
        n_layers = st.number_input("拟合层数", 2, 20, 5, 1) if target_type == "layered" else None
        
        if st.button("转换模型", type="primary"):
            if st.session_state.velocity_model is not None:
                try:
                    new_model = convert_model(
                        st.session_state.velocity_model,
                        target_type,
                        n_layers=n_layers if n_layers else 5
                    )
                    st.session_state.velocity_model = new_model
                    st.success(f"已转换为{target_type}模型！")
                except Exception as e:
                    st.error(f"转换失败: {str(e)}")
            else:
                st.warning("请先创建或加载模型")
    
    with col_conv2:
        if st.button("加载预设模型"):
            model = create_default_model('layered', nx=nx, nz=nz, dx=dx, dz=dz)
            st.session_state.velocity_model = model
            st.success("已加载预设模型！")
        
        if st.button("保存模型到JSON"):
            if st.session_state.velocity_model is not None:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    st.session_state.velocity_model.save(f.name)
                    with open(f.name, 'rb') as f_read:
                        st.download_button(
                            "下载模型文件",
                            f_read,
                            file_name="velocity_model.json",
                            mime="application/json"
                        )
            else:
                st.warning("请先创建模型")
    
    with col_conv3:
        uploaded_model = st.file_uploader("加载模型JSON", type=["json"])
        if uploaded_model is not None:
            try:
                import json
                model_data = json.load(uploaded_model)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(model_data, f)
                    model = VelocityModel.load(f.name)
                    st.session_state.velocity_model = model
                    st.success("模型加载成功！")
            except Exception as e:
                st.error(f"加载失败: {str(e)}")
    
    st.markdown("---")
    st.subheader("模型预览")
    
    if st.session_state.velocity_model is not None:
        model = st.session_state.velocity_model
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("模型类型", model.model_type)
        with col_info2:
            st.metric("模型尺寸", f"{model.nx} x {model.nz}")
        with col_info3:
            st.metric("速度范围", f"{np.min(model.velocity):.0f} - {np.max(model.velocity):.0f} m/s")
        
        with st.spinner("正在绘制速度模型..."):
            fig = plot_velocity(
                model.velocity, model.dx, model.dz,
                cmap=colormap,
                add_contours=add_contours,
                title="速度模型"
            )
            buf = figure_to_bytes(fig)
            st.image(buf, use_column_width=True)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("下载PNG", buf, file_name="velocity_model.png", mime="image/png")
            with col_dl2:
                pdf_buf = figure_to_bytes(fig, format='pdf')
                st.download_button("下载PDF", pdf_buf, file_name="velocity_model.pdf", mime="application/pdf")
    else:
        st.info("请先创建或加载速度模型")

elif page == "🔊 正演模拟":
    st.header("🔊 正演模拟")
    
    if st.session_state.velocity_model is None:
        st.warning("请先在【速度模型定义】页面创建或加载速度模型")
    else:
        model = st.session_state.velocity_model
        
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            st.subheader("数值参数")
            
            dt = st.number_input("时间步长 (s)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
            nt = st.number_input("时间步数", 100, 5000, 1000, 100)
            
            col_source1, col_source2 = st.columns(2)
            with col_source1:
                source_type = st.selectbox("震源子波", ["ricker", "gauss"],
                                          format_func=lambda x: "Ricker小波" if x == "ricker" else "Gauss一阶导数")
                source_freq = st.number_input("震源主频 (Hz)", 5.0, 100.0, 30.0, 5.0)
            with col_source2:
                source_x = st.number_input("震源X位置 (网格)", 0, model.nx - 1, model.nx // 2)
                source_z = st.number_input("震源Z位置 (网格)", 0, model.nz - 1, 2)
            
            boundary_type = st.selectbox("边界条件", ["pml", "mur"],
                                        format_func=lambda x: "完全匹配层(PML)" if x == "pml" else "Mur一阶吸收")
            if boundary_type == "pml":
                pml_width = st.number_input("PML层数", 5, 50, 20, 5)
            else:
                pml_width = 0
        
        with col_params2:
            st.subheader("接收点配置")
            
            receiver_z = st.number_input("接收点深度 (网格)", 0, model.nz - 1, 2)
            
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                rec_x_start = st.number_input("起始X位置", 0, model.nx - 1, 10)
            with col_rec2:
                rec_x_end = st.number_input("结束X位置", 0, model.nx - 1, model.nx - 10)
            with col_rec3:
                rec_spacing = st.number_input("道间距", 1, 10, 1)
            
            st.subheader("稳定性检查")
            
            stability = check_stability(model.velocity, model.dx, model.dz, dt, source_freq * 2)
            
            col_stab1, col_stab2, col_stab3 = st.columns(3)
            with col_stab1:
                cfl_status = "✅ 满足" if stability['cfl_ok'] else "❌ 不满足"
                st.metric("CFL条件", cfl_status)
                if not stability['cfl_ok']:
                    st.warning(f"建议最大dt: {stability['max_dt_stable']*1000:.2f} ms")
            with col_stab2:
                disp_status = "✅ 满足" if stability['dispersion_ok'] else "❌ 不满足"
                st.metric("色散条件", disp_status)
                if not stability['dispersion_ok']:
                    st.warning(f"建议最大网格间距: {stability['max_dx_stable']:.1f} m")
            with col_stab3:
                st.metric("每波长网格点数", f"{1/stability['points_per_wavelength']:.1f}")
        
        if st.button("开始正演模拟", type="primary", disabled=not (stability['cfl_ok'] and stability['dispersion_ok'])):
            if not (stability['cfl_ok'] and stability['dispersion_ok']):
                st.error("数值稳定性条件不满足，请调整参数后重试！")
            else:
                params = ForwardParams(
                    dt=dt, nt=nt, dx=model.dx, dz=model.dz,
                    source_frequency=source_freq, source_type=source_type,
                    boundary_type=boundary_type, pml_width=pml_width,
                    source_x=source_x, source_z=source_z,
                    receiver_z=receiver_z,
                    receiver_x_start=rec_x_start, receiver_x_end=rec_x_end,
                    receiver_spacing=rec_spacing
                )
                
                with st.spinner("正在进行正演模拟..."):
                    result = run_forward(model.velocity, params)
                    st.session_state.forward_result = result
                
                st.success("正演模拟完成！")
        
        if st.session_state.forward_result is not None:
            result = st.session_state.forward_result
            
            st.markdown("---")
            st.subheader("正演结果")
            
            tab_wave, tab_seis, tab_compare = st.tabs(["波场快照", "合成地震记录", "数据对比"])
            
            with tab_wave:
                snapshot_times = result['snapshot_times']
                snap_idx = st.slider("选择时间步", 0, len(snapshot_times) - 1, len(snapshot_times) // 2)
                
                with st.spinner("正在绘制波场快照..."):
                    fig = plot_wavefield_snapshot(
                        result['snapshots'][snap_idx],
                        model.dx, model.dz,
                        snapshot_times[snap_idx],
                        cmap='seismic'
                    )
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
            
            with tab_seis:
                display_mode = st.radio("显示方式", ["变面积图", "Wiggle 波形图"], horizontal=True)
                
                with st.spinner("正在绘制地震记录..."):
                    if display_mode == "变面积图":
                        fig = plot_seismic_image(
                            result['seismograms'], result['time'],
                            title="合成地震记录"
                        )
                    else:
                        fig = plot_seismic_wiggle(
                            result['seismograms'], result['time'],
                            title="合成地震记录"
                        )
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button("下载PNG", buf, file_name="synthetic_seis.png", mime="image/png")
                with col_dl2:
                    pdf_buf = figure_to_bytes(fig, format='pdf')
                    st.download_button("下载PDF", pdf_buf, file_name="synthetic_seis.pdf", mime="application/pdf")
                
                st.subheader("子波")
                fig_wavelet, ax = plt.subplots(figsize=(10, 3))
                ax.plot(result['time'], result['wavelet'], 'b-', linewidth=1)
                ax.set_xlabel('时间 (s)')
                ax.set_ylabel('振幅')
                ax.set_title(f'{source_type.capitalize()} 子波 (主频: {source_freq} Hz)')
                ax.grid(True, alpha=0.3)
                buf_wavelet = figure_to_bytes(fig_wavelet)
                st.image(buf_wavelet, use_column_width=True)
            
            with tab_compare:
                if st.session_state.seismic_data is not None:
                    st.subheader("与观测数据对比")
                    
                    obs_data = st.session_state.seismic_data
                    obs_traces = obs_data['traces'].T
                    syn_traces = result['seismograms']
                    
                    min_traces = min(obs_traces.shape[1], syn_traces.shape[1])
                    min_samples = min(obs_traces.shape[0], syn_traces.shape[0])
                    
                    obs_for_compare = obs_traces[:min_samples, :min_traces]
                    syn_for_compare = syn_traces[:min_samples, :min_traces]
                    time_compare = result['time'][:min_samples]
                    
                    with st.spinner("计算残差..."):
                        residual = compute_residual(obs_for_compare, syn_for_compare)
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1:
                        st.metric("Misfit", f"{residual['misfit']:.4e}")
                    with col_res2:
                        st.metric("RMS误差", f"{residual['rms_error']:.4f}")
                    with col_res3:
                        st.metric("平均相关系数", f"{np.mean(residual['correlation']):.4f}")
                    
                    trace_idx = st.slider("选择道号", 0, min_traces - 1, min_traces // 2)
                    
                    with st.spinner("正在绘制对比图..."):
                        fig_compare = plot_comparison(
                            obs_for_compare, syn_for_compare,
                            time_compare,
                            trace_indices=[trace_idx],
                            title="观测数据 vs 合成数据"
                        )
                        buf_compare = figure_to_bytes(fig_compare)
                        st.image(buf_compare, use_column_width=True)
                else:
                    st.info("请先导入观测数据进行对比")

elif page == "⏱️ 旅行时计算":
    st.header("⏱️ 旅行时计算")
    
    if st.session_state.velocity_model is None:
        st.warning("请先在【速度模型定义】页面创建或加载速度模型")
    else:
        model = st.session_state.velocity_model
        
        col_tt1, col_tt2 = st.columns(2)
        
        with col_tt1:
            st.subheader("震源配置")
            method = st.radio("计算方法", ["快速行进法(FMM)", "直射线追踪"], horizontal=True)
            
            n_sources = st.number_input("震源数量", 1, 10, 1, 1)
            
            sources = []
            for i in range(n_sources):
                col_sx, col_sz = st.columns(2)
                with col_sx:
                    sx = st.number_input(f"震源{i+1} X位置", 0, model.nx - 1, model.nx // 2, key=f"sx{i}")
                with col_sz:
                    sz = st.number_input(f"震源{i+1} Z位置", 0, model.nz - 1, 2, key=f"sz{i}")
                sources.append((sx, sz))
        
        with col_tt2:
            st.subheader("接收点配置")
            n_receivers = st.number_input("接收点数量", 1, model.nx, min(20, model.nx), 1)
            
            receiver_type = st.radio("接收点排列", ["等间距排列", "自定义位置"], horizontal=True)
            
            receivers = []
            if receiver_type == "等间距排列":
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    rx_start = st.number_input("起始X", 0, model.nx - 1, 10)
                with col_r2:
                    rx_end = st.number_input("结束X", 0, model.nx - 1, model.nx - 10)
                with col_r3:
                    rz = st.number_input("接收深度", 0, model.nz - 1, 2)
                
                rx_positions = np.linspace(rx_start, rx_end, n_receivers, dtype=int)
                for rx in rx_positions:
                    receivers.append((rx, rz))
            else:
                for i in range(n_receivers):
                    col_rx, col_rz = st.columns(2)
                    with col_rx:
                        rx = st.number_input(f"接收点{i+1} X位置", 0, model.nx - 1, 10 + i * 5, key=f"rx{i}")
                    with col_rz:
                        rz = st.number_input(f"接收点{i+1} Z位置", 0, model.nz - 1, 2, key=f"rz{i}")
                    receivers.append((rx, rz))
        
        colormap = st.selectbox("速度背景色标", ["viridis", "jet", "seismic"], 0)
        
        if st.button("开始旅行时计算", type="primary"):
            with st.spinner("正在计算旅行时..."):
                if method == "快速行进法(FMM)":
                    result = compute_travel_times(
                        model.velocity, model.dx, model.dz,
                        sources, receivers
                    )
                    st.session_state.travel_time_result = result
                else:
                    if isinstance(model, LayeredModel):
                        layers = model.layers
                        result = {
                            'travel_time_fields': [],
                            'ray_paths': [],
                            'sources': sources,
                            'receivers': receivers,
                            'method': 'ray'
                        }
                        for (sx, sz) in sources:
                            source_rays = []
                            for (rx, rz) in receivers:
                                ray = trace_ray_layered(
                                    layers,
                                    sx * model.dx, sz * model.dz,
                                    rx * model.dx, rz * model.dz,
                                    model.dx, model.dz
                                )
                                source_rays.append(ray)
                            result['ray_paths'].append(source_rays)
                        st.session_state.travel_time_result = result
                    else:
                        st.warning("直射线追踪仅适用于层状模型，将使用FMM方法")
                        result = compute_travel_times(
                            model.velocity, model.dx, model.dz,
                            sources, receivers
                        )
                        st.session_state.travel_time_result = result
                
                st.success("旅行时计算完成！")
        
        if st.session_state.travel_time_result is not None:
            result = st.session_state.travel_time_result
            
            st.markdown("---")
            st.subheader("结果展示")
            
            if len(sources) > 1:
                source_idx = st.slider("选择震源", 0, len(sources) - 1, 0)
            else:
                source_idx = 0
            
            tab_contour, tab_rays = st.tabs(["旅行时等值线", "射线路径"])
            
            with tab_contour:
                if 'travel_time_fields' in result and len(result['travel_time_fields']) > source_idx:
                    tt_field = result['travel_time_fields'][source_idx]
                    rays_for_plot = result.get('ray_paths', [[]])[source_idx] if 'ray_paths' in result else None
                    
                    with st.spinner("正在绘制等值线图..."):
                        fig = plot_travel_time_contours(
                            tt_field, model.velocity,
                            model.dx, model.dz,
                            rays=rays_for_plot,
                            cmap=colormap,
                            title=f"旅行时场 - 震源{source_idx + 1}"
                        )
                        buf = figure_to_bytes(fig)
                        st.image(buf, use_column_width=True)
                else:
                    st.info("直射线追踪方法不显示旅行时场")
            
            with tab_rays:
                if 'ray_paths' in result and len(result['ray_paths']) > source_idx:
                    rays = result['ray_paths'][source_idx]
                    
                    with st.spinner("正在绘制射线路径..."):
                        fig = plt.figure(figsize=(12, 6))
                        ax = fig.add_subplot(111)
                        
                        ax.imshow(model.velocity,
                                  extent=[0, (model.nx - 1) * model.dx,
                                          (model.nz - 1) * model.dz, 0],
                                  cmap=colormap, aspect='auto', alpha=0.5)
                        
                        colors = plt.cm.rainbow(np.linspace(0, 1, len(rays)))
                        for i, ray in enumerate(rays):
                            if ray is not None and len(ray.x) > 1:
                                ax.plot(ray.x, ray.z, color=colors[i], linewidth=2,
                                       label=f'接收点{i+1}: {ray.travel_time*1000:.1f}ms')
                        
                        sx, sz = sources[source_idx]
                        ax.plot(sx * model.dx, sz * model.dz, 'r*', markersize=15, label='震源')
                        
                        for (rx, rz) in receivers:
                            ax.plot(rx * model.dx, rz * model.dz, 'gv', markersize=8)
                        
                        ax.set_xlabel('距离 (m)', fontsize=12)
                        ax.set_ylabel('深度 (m)', fontsize=12)
                        ax.set_title(f'射线路径 - 震源{source_idx + 1}', fontsize=14, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=8)
                        ax.invert_yaxis()
                        
                        buf = figure_to_bytes(fig)
                        st.image(buf, use_column_width=True)
                    
                    st.subheader("旅行时数据")
                    tt_data = []
                    for i, ray in enumerate(rays):
                        if ray is not None:
                            tt_data.append({
                                '接收点': i + 1,
                                'X位置 (m)': receivers[i][0] * model.dx,
                                'Z位置 (m)': receivers[i][1] * model.dz,
                                '旅行时 (ms)': ray.travel_time * 1000
                            })
                    
                    if tt_data:
                        tt_df = pd.DataFrame(tt_data)
                        st.dataframe(tt_df, hide_index=True)

elif page == "🔄 反演算法":
    st.header("🔄 反演算法")
    
    if st.session_state.velocity_model is None:
        st.warning("请先在【速度模型定义】页面创建或加载初始速度模型")
    else:
        initial_model = st.session_state.velocity_model
        
        inv_type = st.radio("反演类型", ["旅行时反演", "波形反演"], horizontal=True)
        
        col_inv1, col_inv2 = st.columns(2)
        
        with col_inv1:
            st.subheader("反演参数")
            
            max_iter = st.number_input("最大迭代次数", 5, 200, 50, 5)
            conv_thresh = st.number_input("收敛阈值 (相对变化)", 1e-6, 1e-2, 1e-4, format="%.1e")
            regularization = st.number_input("正则化参数", 0.0, 1.0, 0.01, 0.001)
            
            if inv_type == "波形反演":
                st.subheader("多尺度策略")
                freq_scales_input = st.text_input("频率序列 (Hz, 逗号分隔)", "5, 10, 20, 30")
                freq_scales = [float(x.strip()) for x in freq_scales_input.split(',')]
            else:
                freq_scales = [5, 10, 20, 30]
        
        with col_inv2:
            st.subheader("观测数据")
            
            if inv_type == "旅行时反演":
                if st.session_state.travel_time_result is not None:
                    tt_result = st.session_state.travel_time_result
                    
                    if 'travel_time_fields' in tt_result and tt_result['travel_time_fields']:
                        sources = tt_result.get('source_locations', tt_result.get('sources', []))
                        receivers = tt_result.get('receiver_locations', tt_result.get('receivers', []))
                        
                        n_sources = len(sources)
                        n_receivers = len(receivers)
                        
                        observed_times = np.zeros((n_sources, n_receivers))
                        
                        for i, tt_field in enumerate(tt_result['travel_time_fields']):
                            for j, (rx, rz) in enumerate(receivers):
                                if 0 <= rx < tt_field.shape[1] and 0 <= rz < tt_field.shape[0]:
                                    observed_times[i, j] = tt_field[rz, rx]
                        
                        st.success(f"已加载旅行时数据: {n_sources} 震源 x {n_receivers} 接收点")
                        
                        observed_data = {
                            'sources': sources,
                            'receivers': receivers,
                            'observed_times': observed_times,
                            'dx': initial_model.dx,
                            'dz': initial_model.dz
                        }
                    else:
                        st.warning("请先在【旅行时计算】页面计算FMM旅行时")
                        observed_data = None
                else:
                    st.warning("请先在【旅行时计算】页面计算旅行时")
                    observed_data = None
                
                if st.checkbox("添加随机噪声到观测数据"):
                    noise_level = st.slider("噪声水平 (%)", 0, 20, 5)
                    if observed_data is not None:
                        observed_data['observed_times'] *= (1 + np.random.normal(0, noise_level / 100,
                                                                               observed_data['observed_times'].shape))
            
            else:
                if st.session_state.forward_result is not None:
                    observed_traces = st.session_state.forward_result['seismograms']
                    st.success(f"已加载波形数据: {observed_traces.shape[0]} 采样点 x {observed_traces.shape[1]} 道")
                    
                    observed_data = {'traces': observed_traces}
                    
                    if st.checkbox("添加随机噪声到观测数据"):
                        noise_level = st.slider("噪声水平 (%)", 0, 50, 5)
                        max_amp = np.max(np.abs(observed_traces))
                        noise = np.random.normal(0, max_amp * noise_level / 100, observed_traces.shape)
                        observed_data['traces'] = observed_traces + noise
                else:
                    if st.session_state.seismic_data is not None:
                        observed_traces = st.session_state.seismic_data['traces'].T
                        st.success(f"已加载导入数据: {observed_traces.shape[0]} 采样点 x {observed_traces.shape[1]} 道")
                        observed_data = {'traces': observed_traces}
                    else:
                        st.warning("请先进行正演模拟获取合成数据，或导入观测数据")
                        observed_data = None
        
        col_cmap, _ = st.columns(2)
        with col_cmap:
            result_cmap = st.selectbox("结果色标", ["viridis", "jet", "seismic"], 0)
        
        if st.button("开始反演", type="primary"):
            inv_params = InversionParams(
                max_iterations=max_iter,
                convergence_threshold=conv_thresh,
                regularization=regularization,
                inversion_type='traveltime' if inv_type == "旅行时反演" else 'waveform',
                frequency_scales=freq_scales,
                verbose=True
            )
            
            forward_params = None
            if inv_type == "波形反演" and st.session_state.forward_result is not None:
                forward_params = st.session_state.forward_result['params']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def callback(iteration, objective, change, model):
                progress = min((iteration + 1) / max_iter, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"迭代 {iteration + 1}: 目标函数 = {objective:.4e}, 相对变化 = {change:.2e}")
            
            inv_params.update_callback = callback
            
            try:
                with st.spinner("正在进行反演计算..."):
                    result = run_inversion(
                        initial_model.velocity,
                        inv_params,
                        forward_params=forward_params,
                        observed_data=observed_data
                    )
                    st.session_state.inversion_result = result
                
                st.success("反演完成！")
                progress_bar.progress(1.0)
                status_text.text("反演完成！")
                
            except Exception as e:
                st.error(f"反演错误: {str(e)}")
        
        if st.session_state.inversion_result is not None:
            result = st.session_state.inversion_result
            
            st.markdown("---")
            st.subheader("反演结果")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("迭代次数", result.iterations)
            with col_res2:
                st.metric("最终目标函数", f"{result.final_objective:.4e}")
            with col_res3:
                st.metric("是否收敛", "✅ 是" if result.converged else "❌ 否")
            
            with st.spinner("正在绘制结果对比图..."):
                fig = plot_inversion_result(
                    result.initial_model,
                    result.inverted_model,
                    initial_model.dx,
                    initial_model.dz,
                    true_model=result.true_model,
                    objective_history=result.objective_history,
                    cmap=result_cmap
                )
                buf = figure_to_bytes(fig)
                st.image(buf, use_column_width=True)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("下载PNG", buf, file_name="inversion_result.png", mime="image/png")
            with col_dl2:
                pdf_buf = figure_to_bytes(fig, format='pdf')
                st.download_button("下载PDF", pdf_buf, file_name="inversion_result.pdf", mime="application/pdf")
            
            if st.button("应用反演结果到速度模型"):
                new_model = GridModel(
                    initial_model.nx, initial_model.nz,
                    initial_model.dx, initial_model.dz
                )
                new_model.velocity = result.inverted_model.astype(np.float32)
                st.session_state.velocity_model = new_model
                st.success("已应用反演结果！")

elif page == "📈 频率域处理":
    st.header("📈 频率域处理")
    
    if st.session_state.seismic_data is None:
        st.warning("请先在【数据导入与管理】页面导入或生成地震数据")
    else:
        data = st.session_state.seismic_data
        
        traces = data['traces']
        dt = data['sample_interval']
        
        tab1, tab2, tab3 = st.tabs(["频谱分析", "带通滤波", "预测反褶积"])
        
        with tab1:
            st.subheader("频谱分析")
            
            trace_idx = st.slider("选择道号", 0, data['n_traces'] - 1, data['n_traces'] // 2)
            
            show_phase = st.checkbox("显示相位谱", True)
            
            with st.spinner("计算频谱..."):
                spec = compute_spectrum(traces[trace_idx, :], dt)
                
                fig = plot_spectrum(
                    spec['frequencies'],
                    spec['amplitude'],
                    phase=spec['phase'] if show_phase else None,
                    title=f"第 {trace_idx + 1} 道频谱"
                )
                buf = figure_to_bytes(fig)
                st.image(buf, use_column_width=True)
            
            st.subheader("平均频谱")
            
            with st.spinner("计算平均频谱..."):
                avg_spec = compute_average_spectrum(traces, dt)
                
                fig_avg, ax = plt.subplots(figsize=(12, 5))
                ax.plot(avg_spec['frequencies'], avg_spec['amplitude_mean'], 'b-',
                       label='平均振幅')
                ax.fill_between(avg_spec['frequencies'],
                               avg_spec['amplitude_mean'] - avg_spec['amplitude_std'],
                               avg_spec['amplitude_mean'] + avg_spec['amplitude_std'],
                               alpha=0.3, label='标准差范围')
                ax.set_xlabel('频率 (Hz)', fontsize=12)
                ax.set_ylabel('振幅', fontsize=12)
                ax.set_title('平均频谱', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                buf_avg = figure_to_bytes(fig_avg)
                st.image(buf_avg, use_column_width=True)
            
            peak_freq = avg_spec['frequencies'][np.argmax(avg_spec['amplitude_mean'])]
            st.info(f"频谱主频: {peak_freq:.1f} Hz")
        
        with tab2:
            st.subheader("带通滤波")
            
            col_filt1, col_filt2, col_filt3 = st.columns(3)
            with col_filt1:
                lowcut = st.number_input("低截止频率 (Hz)", 0.1, 100.0, 5.0, 1.0)
            with col_filt2:
                highcut = st.number_input("高截止频率 (Hz)", 1.0, 500.0, 50.0, 1.0)
            with col_filt3:
                order = st.slider("滤波器阶数", 1, 10, 4)
            
            filter_type = st.selectbox("滤波类型", ["bandpass", "lowpass", "highpass", "bandstop"],
                                      format_func=lambda x: {
                                          'bandpass': '带通',
                                          'lowpass': '低通',
                                          'highpass': '高通',
                                          'bandstop': '带阻'
                                      }[x])
            
            if st.button("应用滤波", type="primary"):
                if lowcut >= highcut and filter_type in ['bandpass', 'bandstop']:
                    st.error("低截止频率必须小于高截止频率！")
                else:
                    filter_params = FilterParams(
                        lowcut=lowcut,
                        highcut=highcut,
                        order=order,
                        filter_type=filter_type
                    )
                    
                    with st.spinner("正在进行滤波..."):
                        result = apply_filter_to_traces(traces, filter_params, dt)
                        st.session_state.process_result = {
                            'type': 'filter',
                            'original': traces,
                            'processed': result['filtered_traces'],
                            'dt': dt,
                            'params': filter_params
                        }
                    
                    st.success("滤波完成！")
            
            if st.session_state.process_result is not None and st.session_state.process_result['type'] == 'filter':
                result = st.session_state.process_result
                
                st.markdown("---")
                st.subheader("处理结果对比")
                
                compare_trace = st.slider("选择对比道号", 0, data['n_traces'] - 1, data['n_traces'] // 2,
                                         key="compare_trace_filter")
                
                with st.spinner("正在绘制对比图..."):
                    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    time = np.arange(data['n_samples']) * dt
                    
                    ax1[0].plot(time, result['original'][compare_trace, :], 'b-', label='原始')
                    ax1[0].plot(time, result['processed'][compare_trace, :], 'r--', label='滤波后')
                    ax1[0].set_xlabel('时间 (s)')
                    ax1[0].set_ylabel('振幅')
                    ax1[0].set_title(f'波形对比 - 第{compare_trace + 1}道')
                    ax1[0].legend()
                    ax1[0].grid(True, alpha=0.3)
                    
                    spec_orig = compute_spectrum(result['original'][compare_trace, :], dt)
                    spec_proc = compute_spectrum(result['processed'][compare_trace, :], dt)
                    
                    ax1[1].plot(spec_orig['frequencies'], spec_orig['amplitude'], 'b-', label='原始')
                    ax1[1].plot(spec_proc['frequencies'], spec_proc['amplitude'], 'r-', label='滤波后')
                    ax1[1].set_xlabel('频率 (Hz)')
                    ax1[1].set_ylabel('振幅')
                    ax1[1].set_title('频谱对比')
                    ax1[1].legend()
                    ax1[1].grid(True, alpha=0.3)
                    ax1[1].set_xlim(0, highcut * 2 if highcut < 200 else 100)
                    
                    avg_orig = compute_average_spectrum(result['original'], dt)
                    avg_proc = compute_average_spectrum(result['processed'], dt)
                    
                    ax2[0].plot(avg_orig['frequencies'], avg_orig['amplitude_mean'], 'b-', label='原始平均')
                    ax2[0].plot(avg_proc['frequencies'], avg_proc['amplitude_mean'], 'r-', label='滤波后平均')
                    ax2[0].set_xlabel('频率 (Hz)')
                    ax2[0].set_ylabel('平均振幅')
                    ax2[0].set_title('平均频谱对比')
                    ax2[0].legend()
                    ax2[0].grid(True, alpha=0.3)
                    ax2[0].set_xlim(0, highcut * 2 if highcut < 200 else 100)
                    
                    display_mode = st.radio("剖面显示", ["变面积", "Wiggle"], horizontal=True, key="display_filter")
                    
                    plt.tight_layout()
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                
                st.subheader("处理后剖面")
                with st.spinner("绘制剖面..."):
                    if display_mode == "变面积":
                        fig_section = plot_seismic_image(
                            result['processed'], time,
                            title='滤波后地震剖面'
                        )
                    else:
                        fig_section = plot_seismic_wiggle(
                            result['processed'], time,
                            title='滤波后地震剖面'
                        )
                    buf_section = figure_to_bytes(fig_section)
                    st.image(buf_section, use_column_width=True)
        
        with tab3:
            st.subheader("预测反褶积")
            
            col_deconv1, col_deconv2 = st.columns(2)
            with col_deconv1:
                pred_lag = st.number_input("预测步长 (采样点)", 1, 50, 1, 1)
                white_noise = st.number_input("白噪系数", 0.0001, 0.1, 0.001, format="%.4f")
            with col_deconv2:
                op_length = st.number_input("算子长度 (采样点)", 10, 200, 60, 10)
            
            if st.button("应用反褶积", type="primary"):
                deconv_params = DeconvolutionParams(
                    prediction_lag=pred_lag,
                    operator_length=op_length,
                    white_noise=white_noise
                )
                
                with st.spinner("正在进行反褶积..."):
                    result = apply_deconvolution_to_traces(traces, deconv_params)
                    st.session_state.process_result = {
                        'type': 'deconvolution',
                        'original': traces,
                        'processed': result['deconvolved_traces'],
                        'dt': dt,
                        'params': deconv_params
                    }
                
                st.success("反褶积完成！")
            
            if st.session_state.process_result is not None and st.session_state.process_result['type'] == 'deconvolution':
                result = st.session_state.process_result
                
                st.markdown("---")
                st.subheader("处理结果对比")
                
                compare_trace = st.slider("选择对比道号", 0, data['n_traces'] - 1, data['n_traces'] // 2,
                                         key="compare_trace_deconv")
                
                with st.spinner("正在绘制对比图..."):
                    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    time = np.arange(data['n_samples']) * dt
                    
                    ax1[0].plot(time, result['original'][compare_trace, :], 'b-', label='原始')
                    ax1[0].plot(time, result['processed'][compare_trace, :], 'r--', label='反褶积后')
                    ax1[0].set_xlabel('时间 (s)')
                    ax1[0].set_ylabel('振幅')
                    ax1[0].set_title(f'波形对比 - 第{compare_trace + 1}道')
                    ax1[0].legend()
                    ax1[0].grid(True, alpha=0.3)
                    
                    spec_orig = compute_spectrum(result['original'][compare_trace, :], dt)
                    spec_proc = compute_spectrum(result['processed'][compare_trace, :], dt)
                    
                    ax1[1].plot(spec_orig['frequencies'], spec_orig['amplitude'], 'b-', label='原始')
                    ax1[1].plot(spec_proc['frequencies'], spec_proc['amplitude'], 'r-', label='反褶积后')
                    ax1[1].set_xlabel('频率 (Hz)')
                    ax1[1].set_ylabel('振幅')
                    ax1[1].set_title('频谱对比')
                    ax1[1].legend()
                    ax1[1].grid(True, alpha=0.3)
                    
                    avg_orig = compute_average_spectrum(result['original'], dt)
                    avg_proc = compute_average_spectrum(result['processed'], dt)
                    
                    ax2[0].plot(avg_orig['frequencies'], avg_orig['amplitude_mean'], 'b-', label='原始平均')
                    ax2[0].plot(avg_proc['frequencies'], avg_proc['amplitude_mean'], 'r-', label='反褶积后平均')
                    ax2[0].set_xlabel('频率 (Hz)')
                    ax2[0].set_ylabel('平均振幅')
                    ax2[0].set_title('平均频谱对比')
                    ax2[0].legend()
                    ax2[0].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)

elif page == "📡 叠加处理":
    st.header("📡 叠加处理")
    
    if st.session_state.seismic_data is None:
        st.warning("请先在【数据导入与管理】页面导入或生成地震数据")
    else:
        data = st.session_state.seismic_data
        
        traces = data['traces']
        dt = data['sample_interval']
        offsets = data.get('offsets', np.arange(data['n_traces']) * 10)
        
        st.subheader("CDP道集提取")
        
        gather_type = st.radio("道集类型", ["整个剖面作为CDP", "按CDP号提取"], horizontal=True)
        
        if gather_type == "整个剖面作为CDP":
            cdp_traces = traces
            cdp_offsets = offsets
            cdp_number = "All"
        else:
            if 'cdp_numbers' in data:
                cdp_numbers = np.unique(data['cdp_numbers'])
                cdp_idx = st.selectbox("选择CDP号", cdp_numbers, index=len(cdp_numbers) // 2)
                cdp_mask = data['cdp_numbers'] == cdp_idx
                cdp_traces = traces[cdp_mask, :]
                cdp_offsets = offsets[cdp_mask]
                cdp_number = cdp_idx
            else:
                st.warning("数据中没有CDP号信息，使用整个剖面")
                cdp_traces = traces
                cdp_offsets = offsets
                cdp_number = "All"
        
        st.info(f"CDP {cdp_number}: {cdp_traces.shape[0]} 道, 每道 {cdp_traces.shape[1]} 采样点")
        
        col_stack1, col_stack2 = st.columns(2)
        
        with col_stack1:
            st.subheader("速度谱参数")
            v_min = st.number_input("最小速度 (m/s)", 500.0, 3000.0, 1000.0, 100.0)
            v_max = st.number_input("最大速度 (m/s)", 2000.0, 8000.0, 5000.0, 100.0)
            dv = st.number_input("速度间隔 (m/s)", 10.0, 200.0, 50.0, 10.0)
            semblance_win = st.slider("相干时窗 (采样点)", 1, 21, 5, 2)
        
        with col_stack2:
            st.subheader("NMO参数")
            stretch_limit = st.slider("最大拉伸比 (%)", 0, 100, 30, 5) / 100
            mute_near = st.number_input("近偏移距切除 (m)", 0.0, 500.0, 0.0, 10.0)
            mute_far = st.number_input("远偏移距切除 (m)", 0.0, 10000.0, 0.0, 100.0)
            if mute_far == 0:
                mute_far = None
        
        if st.button("计算速度谱和NMO", type="primary"):
            with st.spinner("正在计算..."):
                spectrum_params = VelocitySpectrumParams(
                    v_min=v_min,
                    v_max=v_max,
                    dv=dv,
                    semblance_window=semblance_win
                )
                
                nmo_params = NMOParams(
                    stretch_limit=stretch_limit,
                    mute_near_offsets=mute_near,
                    mute_far_offsets=mute_far
                )
                
                result = process_cdp_gather(
                    cdp_traces, cdp_offsets, dt,
                    nmo_params, spectrum_params
                )
                
                st.session_state.stacking_result = result
            
            st.success("处理完成！")
        
        if st.session_state.stacking_result is not None:
            result = st.session_state.stacking_result
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["原始道集", "速度谱", "NMO校正", "叠加结果"])
            
            with tab1:
                st.subheader("原始CDP道集")
                time = np.arange(data['n_samples']) * dt
                
                display_mode = st.radio("显示方式", ["变面积", "Wiggle"], horizontal=True, key="orig_display")
                
                with st.spinner("绘制道集..."):
                    if display_mode == "变面积":
                        fig = plot_seismic_image(cdp_traces, time, title=f"CDP {cdp_number} 原始道集")
                    else:
                        fig = plot_seismic_wiggle(cdp_traces, time, title=f"CDP {cdp_number} 原始道集")
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                
                fig_offset, ax = plt.subplots(figsize=(12, 4))
                ax.plot(cdp_offsets, 'o-')
                ax.set_xlabel('道号')
                ax.set_ylabel('炮检距 (m)')
                ax.set_title('炮检距分布')
                ax.grid(True, alpha=0.3)
                buf_offset = figure_to_bytes(fig_offset)
                st.image(buf_offset, use_column_width=True)
            
            with tab2:
                st.subheader("速度谱")
                
                spectrum_result = result['velocity_spectrum']
                
                with st.spinner("绘制速度谱..."):
                    fig = plot_velocity_spectrum(
                        spectrum_result['spectrum'],
                        spectrum_result['velocities'],
                        spectrum_result['times'],
                        picks=result['velocity_picks'],
                        title=f"CDP {cdp_number} 速度谱"
                    )
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                
                st.subheader("速度拾取")
                picks_df = pd.DataFrame(result['velocity_picks'])
                if not picks_df.empty:
                    picks_df = picks_df[['time', 'velocity', 'semblance']]
                    picks_df.columns = ['时间 (s)', '速度 (m/s)', '相干值']
                    st.dataframe(picks_df, hide_index=True)
                
                fig_vfunc, ax = plt.subplots(figsize=(10, 4))
                ax.plot(result['velocity_function'], np.arange(data['n_samples']) * dt, 'b-', linewidth=2)
                ax.set_xlabel('速度 (m/s)')
                ax.set_ylabel('时间 (s)')
                ax.set_title('NMO速度函数')
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()
                buf_vfunc = figure_to_bytes(fig_vfunc)
                st.image(buf_vfunc, use_column_width=True)
            
            with tab3:
                st.subheader("NMO校正结果")
                
                nmo_result = result['nmo_result']
                
                display_mode = st.radio("显示方式", ["变面积", "Wiggle"], horizontal=True, key="nmo_display")
                
                with st.spinner("绘制NMO校正道集..."):
                    if display_mode == "变面积":
                        fig = plot_seismic_image(
                            nmo_result['corrected_gather'],
                            time,
                            title='NMO校正后道集'
                        )
                    else:
                        fig = plot_seismic_wiggle(
                            nmo_result['corrected_gather'],
                            time,
                            title='NMO校正后道集'
                        )
                    buf = figure_to_bytes(fig)
                    st.image(buf, use_column_width=True)
                
                st.subheader("拉伸切除")
                fig_mute, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(nmo_result['mute_masks'].T, extent=[0, len(cdp_offsets), time[-1], time[0]],
                          aspect='auto', cmap='gray')
                ax.set_xlabel('道号')
                ax.set_ylabel('时间 (s)')
                ax.set_title('切除掩模 (白色=有效)')
                buf_mute = figure_to_bytes(fig_mute)
                st.image(buf_mute, use_column_width=True)
            
            with tab4:
                st.subheader("CDP叠加结果")
                
                stacked_trace = result['stacked_trace']
                
                fig_stack, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(stacked_trace, time, 'k-', linewidth=1)
                ax1.fill_betweenx(time, 0, stacked_trace, where=stacked_trace > 0, color='black')
                ax1.set_xlabel('振幅')
                ax1.set_ylabel('时间 (s)')
                ax1.set_title('CDP叠加道')
                ax1.grid(True, alpha=0.3)
                ax1.invert_yaxis()
                
                max_amp = np.max(np.abs(stacked_trace))
                ax2.plot(np.zeros_like(time), time, 'k-', linewidth=0.5)
                for i in range(0, len(time), 5):
                    amp = stacked_trace[i] / max_amp * 2
                    ax2.plot([-amp, amp], [time[i], time[i]], 'k-', linewidth=1)
                ax2.set_xlim(-2.5, 2.5)
                ax2.set_xlabel('变面积显示')
                ax2.set_ylabel('时间 (s)')
                ax2.set_title('变面积叠加道')
                ax2.grid(True, alpha=0.3)
                ax2.invert_yaxis()
                ax2.set_yticklabels([])
                
                plt.tight_layout()
                buf_stack = figure_to_bytes(fig_stack)
                st.image(buf_stack, use_column_width=True)
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button("下载PNG", buf_stack, file_name="cdp_stack.png", mime="image/png")
                with col_dl2:
                    pdf_buf = figure_to_bytes(fig_stack, format='pdf')
                    st.download_button("下载PDF", pdf_buf, file_name="cdp_stack.pdf", mime="application/pdf")

elif page == "🎨 可视化导出":
    st.header("🎨 可视化导出")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("可用数据")
        
        data_options = []
        if st.session_state.seismic_data is not None:
            data_options.append("地震数据")
        if st.session_state.velocity_model is not None:
            data_options.append("速度模型")
        if st.session_state.forward_result is not None:
            data_options.append("正演结果")
        if st.session_state.travel_time_result is not None:
            data_options.append("旅行时结果")
        if st.session_state.inversion_result is not None:
            data_options.append("反演结果")
        if st.session_state.process_result is not None:
            data_options.append("频率域处理结果")
        if st.session_state.stacking_result is not None:
            data_options.append("叠加处理结果")
        
        if not data_options:
            st.info("请先在其他页面生成数据")
        else:
            selected_data = st.multiselect("选择要导出的内容", data_options, default=data_options)
            
            st.subheader("导出设置")
            export_format = st.selectbox("图片格式", ["PNG", "PDF"])
            export_dpi = st.slider("分辨率 (DPI)", 72, 300, 150)
            colormap = st.selectbox("色标", ["viridis", "jet", "seismic", "plasma", "inferno", "magma"])
    
    with col_vis2:
        st.subheader("批量导出")
        
        if st.button("生成所有图片", type="primary") and data_options:
            with st.spinner("正在生成图片..."):
                export_dir = tempfile.mkdtemp()
                generated_files = []
                
                if "速度模型" in selected_data and st.session_state.velocity_model is not None:
                    model = st.session_state.velocity_model
                    fig = plot_velocity(
                        model.velocity, model.dx, model.dz,
                        cmap=colormap,
                        title="速度模型"
                    )
                    fname = os.path.join(export_dir, f"velocity_model.{export_format.lower()}")
                    fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                    generated_files.append(("速度模型", fname))
                
                if "地震数据" in selected_data and st.session_state.seismic_data is not None:
                    data = st.session_state.seismic_data
                    traces = data['traces']
                    time = np.arange(data['n_samples']) * data['sample_interval']
                    
                    fig = plot_seismic_image(traces, time, cmap='seismic', title="地震剖面")
                    fname = os.path.join(export_dir, f"seismic_section.{export_format.lower()}")
                    fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                    generated_files.append(("地震剖面", fname))
                    
                    fig = plot_seismic_wiggle(traces, time, title="地震剖面 (Wiggle)")
                    fname = os.path.join(export_dir, f"seismic_wiggle.{export_format.lower()}")
                    fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                    generated_files.append(("地震剖面(Wiggle)", fname))
                
                if "正演结果" in selected_data and st.session_state.forward_result is not None:
                    result = st.session_state.forward_result
                    fig = plot_seismic_image(
                        result['seismograms'], result['time'],
                        cmap='seismic',
                        title="合成地震记录"
                    )
                    fname = os.path.join(export_dir, f"synthetic_seismic.{export_format.lower()}")
                    fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                    generated_files.append(("合成地震记录", fname))
                    
                    if result['snapshots']:
                        mid_idx = len(result['snapshots']) // 2
                        fig = plot_wavefield_snapshot(
                            result['snapshots'][mid_idx],
                            st.session_state.velocity_model.dx,
                            st.session_state.velocity_model.dz,
                            result['snapshot_times'][mid_idx],
                            cmap='seismic'
                        )
                        fname = os.path.join(export_dir, f"wavefield_snapshot.{export_format.lower()}")
                        fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                        generated_files.append(("波场快照", fname))
                
                if "反演结果" in selected_data and st.session_state.inversion_result is not None:
                    result = st.session_state.inversion_result
                    fig = plot_inversion_result(
                        result.initial_model,
                        result.inverted_model,
                        st.session_state.velocity_model.dx,
                        st.session_state.velocity_model.dz,
                        true_model=result.true_model,
                        objective_history=result.objective_history,
                        cmap=colormap
                    )
                    fname = os.path.join(export_dir, f"inversion_result.{export_format.lower()}")
                    fig.savefig(fname, dpi=export_dpi, bbox_inches='tight')
                    generated_files.append(("反演结果", fname))
                
                st.success(f"已生成 {len(generated_files)} 个图片文件！")
                
                st.subheader("下载文件")
                for name, fpath in generated_files:
                    with open(fpath, 'rb') as f:
                        st.download_button(
                            f"下载 {name}",
                            f,
                            file_name=os.path.basename(fpath),
                            mime=f"image/{export_format.lower()}"
                        )
                
                if st.button("下载模型JSON") and st.session_state.velocity_model is not None:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        st.session_state.velocity_model.save(f.name)
                        with open(f.name, 'rb') as f_read:
                            st.download_button(
                                "下载速度模型JSON",
                                f_read,
                                file_name="velocity_model.json",
                                mime="application/json"
                            )
    
    st.markdown("---")
    st.subheader("使用说明")
    
    with st.expander("📖 快速入门指南"):
        st.markdown("""
        ### 地震波形反演与地层结构成像工具
        
        **工作流程推荐:**
        
        1. **数据导入**: 在【数据导入与管理】页面导入SEG-Y或CSV格式的地震数据，或生成教学演示数据
        
        2. **建立模型**: 在【速度模型定义】页面创建初始速度模型，支持三种建模方式:
           - 层状模型: 指定每层深度和速度
           - 渐变模型: 指定顶底速度，线性或指数渐变
           - 网格模型: 逐像素编辑速度值
        
        3. **正演模拟**: 在【正演模拟】页面设置参数进行波场正演
           - 自动检查CFL稳定性和数值色散条件
           - 支持PML和Mur吸收边界
           - 支持Ricker和Gauss子波
        
        4. **旅行时计算**: 在【旅行时计算】页面计算初至旅行时
           - 快速行进法(FMM)求解Eikonal方程
           - 直射线追踪（层状介质）
        
        5. **反演**: 在【反演算法】页面进行速度反演
           - 旅行时反演: LSQR方法
           - 波形反演: 伴随状态法 + L-BFGS优化，多尺度策略
        
        6. **数据处理**: 在【频率域处理】和【叠加处理】页面进行常规处理
        
        7. **结果导出**: 在【可视化导出】页面批量导出结果
        """)
    
    with st.expander("⚙️ 数值稳定性说明"):
        st.markdown("""
        ### 有限差分正演的数值稳定性
        
        **CFL条件 (Courant-Friedrichs-Lewy condition):**
        - 保证数值求解稳定的必要条件
        - CFL = v_max × dt / min(dx, dz) < 1/√2 ≈ 0.707
        
        **数值色散条件:**
        - 每波长至少需要10个网格点才能有效抑制数值色散
        - 最小速度 / 最高频率 / min(dx, dz) ≥ 10
        
        **自动检查:**
        - 程序会自动计算并显示CFL数和每波长点数
        - 如果不满足条件，会给出建议的参数调整值
        - 不满足稳定性条件时无法启动正演计算
        """)
    
    with st.expander("📊 反演算法说明"):
        st.markdown("""
        ### 反演算法
        
        **旅行时反演 (Traveltime Tomography):**
        - 基于FMM计算的初至旅行时
        - 用LSQR迭代求解线性化的速度扰动
        - 射线覆盖不均匀区域自动正则化
        
        **波形反演 (Full Waveform Inversion - FWI):**
        - 以波形残差平方和为目标函数
        - 伴随状态法计算梯度
        - L-BFGS算法优化迭代
        - 多尺度策略: 先低频后高频，避免周期跳跃
        - 支持收敛阈值和最大迭代次数双重终止条件
        """)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌍 地震波形反演与地层结构成像工具 | 基于Python + Streamlit开发</p>
        <p style="font-size: 0.8em;">支持SEG-Y数据导入、速度建模、有限差分正演、FMM旅行时计算、波形反演等完整功能</p>
    </div>
    """,
    unsafe_allow_html=True
)