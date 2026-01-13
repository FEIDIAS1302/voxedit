import streamlit as st
import numpy as np
import pyworld as pw
import soundfile as sf
import io
import base64
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from scipy.interpolate import interp1d
import os
import requests
import uuid

# ==========================================
# ★設定エリア：Secretsからキーを読み込む
# ==========================================
import os

# Streamlit CloudのSecrets、または環境変数から読み込む安全な設計
try:
    DEFAULT_API_KEY = st.secrets["FISH_AUDIO_API_KEY"]
    DEFAULT_MODEL_ID = st.secrets["FISH_AUDIO_MODEL_ID"]
except FileNotFoundError:
    # ローカル実行時などSecretsがない場合は空文字（または手入力）にする
    DEFAULT_API_KEY = "" 
    DEFAULT_MODEL_ID = ""


# ==========================================
# 0. ページ設定 & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Intonation Editor", page_icon="🎼")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Arial, sans-serif; color: #444; background: #fff; }
    [data-testid="stSidebar"] { background-color: #fcfcfc; border-right: 1px solid #eee; }
    .stButton > button { background-color: #333; color: #fff; border: none; border-radius: 0; width: 100%; }
    .stButton > button:hover { background-color: #555; }
    div[data-testid="stHorizontalBlock"] { overflow-x: auto; }
    
    [data-testid="stImage"] {
        margin-top: 30px;
        padding-bottom: 20px;
        display: flex;
        justify-content: center;
    }
    
    .header-title {
        text-align: center;
        font-family: "Times New Roman", serif;
        letter-spacing: 0.1em;
        font-size: 2.5rem;
        margin-top: 40px;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

if 'init' not in st.session_state:
    st.session_state['init'] = False
    st.session_state['segments'] = []
    st.session_state['update_count'] = 0
    st.session_state['current_audio_x'] = None
    st.session_state['current_fs'] = None
    st.session_state['current_source_id'] = None

# ==========================================
# 1. 関数定義
# ==========================================

# ★修正：PCM（生データ）として取得する関数
def call_fish_audio_api_pcm(text, reference_id, api_key):
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # formatを "pcm" に指定
    payload = {
        "text": text,
        "reference_id": reference_id,
        "format": "pcm", 
        "latency": "normal",
        "sample_rate": 44100 # サンプリングレートを明示
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            st.error(f"❌ API Request Failed (Status: {response.status_code})")
            return None, None

        # 生バイナリデータを取得
        raw_data = response.content
        if len(raw_data) == 0:
            st.error("❌ API returned 0 bytes.")
            return None, None

        # PCM (16bit int) を numpy配列 (float -1.0~1.0) に変換
        # ※Fish AudioのPCMは通常 16bit Little Endian
        audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float64) / 32768.0
        
        return audio_float, 44100 # 指定したレートを返す
        
    except Exception as e:
        st.error(f"❌ System Error: {e}")
        return None, None

def test_fish_api_connection(api_key, model_id):
    url = "https://api.fish.audio/v1/tts"
    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
    # テスト時もmp3で軽めに確認
    payload = { "text": "T", "reference_id": model_id, "format": "mp3", "latency": "normal" }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200 and len(response.content) > 100:
            return True, "Connection Successful!"
        elif response.status_code == 401: return False, "❌ Error 401: Invalid API Key."
        elif response.status_code == 404: return False, "❌ Error 404: Model ID not found."
        else: return False, f"❌ Error {response.status_code}"
    except Exception as e: return False, f"Connection Error: {e}"

def find_voiced_segments(f0, fs, min_duration_ms=50, max_note_ms=300):
    is_voiced = f0 > 0
    diff = np.diff(is_voiced.astype(int), prepend=0, append=0)
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    segments = []
    min_frames, max_frames = int(min_duration_ms/5.0), int(max_note_ms/5.0)
    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= min_frames:
            if duration > max_frames:
                num_splits = int(np.ceil(duration / max_frames))
                chunk = duration / num_splits
                for k in range(num_splits):
                    sub_s, sub_e = int(s + k*chunk), int(s + (k+1)*chunk)
                    if k == num_splits-1: sub_e = e
                    avg = np.mean(f0[sub_s:sub_e])
                    segments.append({"start": sub_s, "end": sub_e, "val": avg, "original_val": avg, "center": (sub_s+sub_e)/2})
            else:
                avg = np.mean(f0[s:e])
                segments.append({"start": s, "end": e, "val": avg, "original_val": avg, "center": (s+e)/2})
    return segments

def generate_waveform_bg(audio_data, width_px, height_px):
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    step = max(1, len(audio_data)//int(width_px*2))
    ax.plot(audio_data[::step], color='#708090', alpha=0.3, linewidth=1)
    ax.set_ylim(-1, 1); ax.axis('off')
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ==========================================
# 2. UI
# ==========================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300) 
    else: 
        st.markdown("""
        <h1 class='header-title'>VOX EDITOR</h1>
        <p style='text-align: center; color: #999; margin-top: 10px;'>AI-Powered Intonation Control</p>
        """, unsafe_allow_html=True)

st.sidebar.markdown("### 1. Input Source")
tab1, tab2 = st.sidebar.tabs(["📁 Upload", "🐟 Fish Audio"])

with tab1:
    uf = st.file_uploader("Upload WAV", type=["wav"])
    if uf:
        fid = f"up_{uf.name}"
        if st.session_state.get('current_source_id') != fid:
            d, f = sf.read(uf)
            if d.ndim>1: d=d.mean(axis=1)
            st.session_state.update({'current_audio_x': d.astype(np.float64), 'current_fs': f, 'current_source_id': fid, 'segments': []})
            st.rerun()

with tab2:
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    model_id = st.text_input("Model ID", value=DEFAULT_MODEL_ID)
    
    if st.button("📡 Test Connection"):
        if not api_key or not model_id:
            st.sidebar.error("Please enter Key and ID.")
        else:
            with st.sidebar.status("Testing..."):
                success, msg = test_fish_api_connection(api_key, model_id)
                if success: st.sidebar.success(msg)
                else: st.sidebar.error(msg)
    
    st.markdown("---")

    # ▼▼▼ 追加・変更箇所ここから ▼▼▼
    
    # 感情タグの定義
    emotion_map = {
        "指定なし (Default)": "",
        "楽しい (Happy)": "(happy)",
        "悲しい (Sad)": "(sad)",
        "怒り (Angry)": "(angry)",
        "興奮 (Excited)": "(excited)",
        "穏やか (Calm)": "(calm)",
        "ささやき (Whispering)": "(whispering)",
        "叫び (Shouting)": "(shouting)"
    }

    # ドロップダウンの表示
    selected_label = st.selectbox(
        "Emotion / Tone (※オプション)",
        options=list(emotion_map.keys())
    )
    
    # 選択されたタグを取得 (例: "(happy)")
    selected_tag = emotion_map[selected_label]

    # テキスト入力エリア
    txt = st.text_area("Text", "こんにちは")
    
    if st.button("Generate Audio", type="primary"):
        if not api_key or not model_id: 
            st.error("Key/ID required")
        else:
            with st.spinner("Generating PCM Audio..."):
                # タグ結合処理: タグがある場合のみ先頭に付与して半角スペースを入れる
                final_text = f"{selected_tag} {txt}" if selected_tag else txt
                
                # コンソール等での確認用（不要なら削除可）
                print(f"Generating with text: {final_text}")

                # API呼び出し (引数を final_text に変更)
                audio_data, sample_rate = call_fish_audio_api_pcm(final_text, model_id, api_key)
                
                if audio_data is not None:
    # ▲▲▲ 追加・変更箇所ここまで ▲▲▲
                    # データが空でないか最終チェック
                    if audio_data.size == 0:
                        st.error("❌ Generated PCM data is empty.")
                    else:
                        st.session_state.update({
                            'current_audio_x': audio_data, 
                            'current_fs': sample_rate, 
                            'current_source_id': f"fish_{uuid.uuid4()}", 
                            'segments': []
                        })
                        st.success("Done! Audio loaded.")
                        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Clear All State"):
    st.session_state.clear()
    st.rerun()

zoom = st.sidebar.slider("Zoom", 1.0, 5.0, 1.5)
p_min = st.sidebar.number_input("Min Hz", 0, 400, 0, 50)
p_max = st.sidebar.number_input("Max Hz", 200, 1000, 600, 50)
st.sidebar.markdown("---")
max_ms = st.sidebar.slider("Split ms", 50, 1000, 200, 50)
min_ms = st.sidebar.slider("Min ms", 10, 200, 50)
if st.sidebar.button("Reset Edits"):
    for s in st.session_state.get('segments', []): s['val'] = s['original_val']
    st.session_state['update_count'] += 1
    st.rerun()

# ==========================================
# 3. メイン処理
# ==========================================
x = st.session_state.get('current_audio_x')
fs = st.session_state.get('current_fs')

if x is None or x.size == 0:
    st.info("👈 左のサイドバーからファイルをアップロードするか、音声を生成してください。")
else:
    mx = np.max(np.abs(x))
    if mx > 1.0 or (mx > 0 and mx < 0.1): x = x / mx * 0.9

    aid = f"{st.session_state['current_source_id']}_{min_ms}_{max_ms}"
    if st.session_state.get('last_aid') != aid:
        with st.spinner("Analyzing..."):
            try:
                _f0, t = pw.harvest(x, fs, frame_period=5.0)
                f0 = pw.stonemask(x, _f0, t, fs)
                sp = pw.cheaptrick(x, f0, t, fs)
                ap = pw.d4c(x, f0, t, fs)
                st.session_state.update({'f0': f0, 'sp': sp, 'ap': ap, 'fs': fs, 'audio_raw': x})
                segs = find_voiced_segments(f0, fs, min_ms, max_ms)
                if not segs: st.warning("音声区間が見つかりませんでした。")
                st.session_state.update({'segments': segs, 'last_aid': aid, 'init': True, 'update_count': st.session_state['update_count']+1})
            except Exception as e: st.error(f"Analysis Error: {e}")

    if st.session_state.get('segments'):
        st.markdown("---")
        base_w = 800; w = int(base_w * zoom); h = 400
        objs = []
        
        objs.append({"type": "image", "src": generate_waveform_bg(st.session_state['audio_raw'], w, h), "left": 0, "top": 0, "width": w, "height": h, "selectable": False})
        
        ghz = 50
        shz = (p_min // ghz) * ghz
        for hz in range(shz, p_max+1, ghz):
            if hz < p_min: continue
            yp = h - ((hz - p_min)/(p_max - p_min) * h)
            objs.append({"type": "line", "x1": 0, "y1": yp, "x2": w, "y2": yp, "stroke": "#f0f0f0", "strokeWidth": 1, "selectable": False})
        
        dur = len(st.session_state['audio_raw']) / fs
        gms = 100
        for ms in range(0, int(dur*1000), gms):
            xp = (ms / (dur*1000)) * w
            objs.append({"type": "line", "x1": xp, "y1": 0, "x2": xp, "y2": h, "stroke": "#f5f5f5", "strokeWidth": 1, "selectable": False})

        tf = len(st.session_state['f0'])
        for i, s in enumerate(st.session_state['segments']):
            spx = (s['start']/tf)*w; epx = (s['end']/tf)*w; wid = max(2, epx-spx)
            oy = h - ((s['original_val']-p_min)/(p_max-p_min)*h) - 4
            objs.append({"type": "rect", "left": int(spx), "top": int(oy), "width": int(wid), "height": 8, "fill": "#e0e0e0", "selectable": False})
            ny = h - ((s['val']-p_min)/(p_max-p_min)*h) - 4
            col = "#2C3E50" if abs(s['val']-s['original_val']) > 1 else "#B0B0B0"
            objs.append({"type": "rect", "left": int(spx), "top": int(ny), "width": int(wid), "height": 8, "fill": col, "selectable": False})

        ckey = f"cv_{st.session_state['update_count']}_{w}_{aid}"
        st.caption(f"Range: {p_min}-{p_max}Hz")
        with st.container():
            st.markdown(f'<div style="overflow-x: scroll; border: 1px solid #eee;">', unsafe_allow_html=True)
            res = st_canvas(fill_color="#00000000", stroke_width=0, background_color="#FFF", initial_drawing={"objects": objs}, update_streamlit=True, height=h, width=w, drawing_mode="point", point_display_radius=0, key=ckey)
            st.markdown('</div>', unsafe_allow_html=True)

        if res.json_data:
            clicks = [o for o in res.json_data["objects"] if o["type"] == "circle"]
            if clicks:
                lx, ly = clicks[-1]["left"], clicks[-1]["top"]
                cf = (lx/w)*tf
                tidx = -1; mind = float('inf')
                for i, s in enumerate(st.session_state['segments']):
                    if s['start']-10 <= cf <= s['end']+10: tidx = i; break
                    d = min(abs(cf-s['start']), abs(cf-s['end']))
                    if d < mind: mind = d; tidx = i
                if tidx != -1:
                    nv = p_min + ((h-ly)/h)*(p_max-p_min)
                    st.session_state['segments'][tidx]['val'] = max(0, nv)
                    st.session_state['update_count'] += 1
                    st.rerun()

        of0 = st.session_state['f0']; ol = len(of0)
        xp = [0]; yd = [0]
        for s in st.session_state['segments']:
            xp.append(s['center']); yd.append(s['val'] - s['original_val'])
        xp.append(ol-1); yd.append(0)
        
        if len(xp) > 2:
            dc = interp1d(xp, yd, kind='linear', fill_value=0, bounds_error=False)(np.arange(ol))
            dc = np.convolve(dc, np.ones(30)/30, mode='same')
            nf0 = np.where(of0>0, np.maximum(0, of0+dc), 0.0)
            syn = pw.synthesize(nf0, st.session_state['sp'], st.session_state['ap'], st.session_state['fs'])
            st.markdown("### Preview")
            c1, c2 = st.columns([2, 1])
            c1.audio(syn, sample_rate=st.session_state['fs'])
            buf = io.BytesIO()
            sf.write(buf, syn, st.session_state['fs'], format='WAV')
            c2.download_button("Download", buf, "edited.wav", "audio/wav", use_container_width=True)