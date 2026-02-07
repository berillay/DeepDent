import streamlit as st
from ultralytics import YOLO
from PIL import Image
import datetime
import pandas as pd

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Dental Radyoloji Sistemi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- KURUMSAL TASARIM (CSS) ---
# Burada yazı renklerini SİYAH (Black) olarak zorluyoruz ki okunmama sorunu olmasın.
st.markdown("""
    <style>
    /* Genel Sayfa Yapısı */
    .stApp {
        background-color: #f4f6f9; /* Çok açık gri (Hastane duvarı gibi) */
        color: #212529; /* Koyu antrasit yazı rengi */
    }
    
    /* Input (Giriş) Etiketleri */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stTextArea > label {
        color: #2c3e50 !important; /* Lacivert tonlu siyah */
        font-weight: 700 !important; /* Kalın yazı */
        font-size: 14px !important;
    }
    
    /* Üst Header Bandı */
    .header-bar {
        background-color: #1a252f; /* Çok koyu lacivert */
        color: white;
        padding: 20px;
        margin-bottom: 25px;
        border-bottom: 4px solid #3498db; /* Mavi ince çizgi */
    }
    
    /* Rapor Kutuları */
    .report-box {
        background-color: white;
        border: 1px solid #ced4da;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buton Tasarımı - Ciddi */
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 4px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #1c5980;
        color: white;
    }
    
    /* Tablo Başlıkları */
    thead tr th {
        background-color: #ecf0f1 !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
st.markdown("""
    <div class="header-bar">
        <h2 style='margin:0; font-family: Arial, sans-serif;'>DENTAL DIAGNOSTIC AI SYSTEM</h2>
        <p style='margin:0; font-size: 14px; opacity: 0.8;'>Radyolojik Görüntü İşleme ve Karar Destek Modülü</p>
    </div>
""", unsafe_allow_html=True)

# --- BÖLÜM 1: HASTA KAYIT PANELİ ---
st.markdown("#### 1. HASTA KİMLİK VE PROTOKOL BİLGİLERİ")
st.markdown("---")

# Beyaz bir kutu içinde form
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # label= parametresi ile kutunun üstünde ne yazdığını netleştiriyoruz
        p_name = st.text_input(label="HASTA ADI SOYADI", placeholder="Örn: Ahmet Yılmaz")
        p_protocol = st.text_input(label="PROTOKOL / DOSYA NO", placeholder="Örn: 2025-0012")

    with col2:
        p_age = st.number_input(label="HASTA YAŞI", min_value=0, max_value=120, step=1, value=0)
        p_doctor = st.text_input(label="SORUMLU HEKİM", value="Dt. Berilay Kaplan", disabled=True) # Değiştirilemez alan

    with col3:
        p_date = st.date_input(label="İŞLEM TARİHİ", value=datetime.date.today())
        p_notes = st.text_input(label="KLİNİK ENDİKASYON / ŞİKAYET", placeholder="Örn: Sağ alt molar bölgede ağrı")

# --- BÖLÜM 2: GÖRÜNTÜLEME VE ANALİZ ---
st.write("") # Boşluk
st.markdown("#### 2. RADYOLOJİK GÖRÜNTÜ ANALİZİ")
st.markdown("---")

# Dosya yükleyici
uploaded_file = st.file_uploader(label="PANORAMİK RÖNTGEN DOSYASI SEÇİNİZ (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Model Yükleme
try:
    model = YOLO('best.pt')
except:
    st.error("SİSTEM HATASI: Model dosyası (best.pt) dizinde bulunamadı.")
    st.stop()

if uploaded_file is not None:
    # Form Kontrolü
    if not p_name:
        st.error("HATA: Analize başlamak için lütfen 'Hasta Adı Soyadı' alanını doldurunuz.")
    else:
        # Resmi Yükle
        image = Image.open(uploaded_file)
        
        col_img, col_act = st.columns([3, 1])
        
        with col_img:
            st.image(image, caption="Yüklenen Ham Görüntü", use_column_width=True)
            
        with col_act:
            st.info("Sistem Hazır.")
            st.write("Görüntü işleme algoritması başlatılacaktır.")
            # Ciddi bir buton
            start_btn = st.button("ANALİZİ BAŞLAT", type="primary", use_container_width=True)

        if start_btn:
            with st.spinner('Görüntü taranıyor ve lezyonlar işaretleniyor...'):
                # Tahmin (Confidence 0.25)
                results = model.predict(image, conf=0.25)
                res_plotted = results[0].plot(line_width=2, font_size=12)
                
                # --- SONUÇ RAPORU ---
                st.markdown("### 3. ANALİZ SONUÇ RAPORU")
                st.markdown("---")
                
                r_col1, r_col2 = st.columns([1, 1])
                
                with r_col1:
                    st.image(res_plotted[:, :, ::-1], caption="AI İşlenmiş Görüntü", use_column_width=True)
                
                with r_col2:
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(f"**Hasta:** {p_name} | **Protokol:** {p_protocol}")
                    st.markdown(f"**Tarih:** {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}")
                    st.markdown("---")
                    
                    detections = results[0].boxes
                    count = len(detections)
                    
                    if count > 0:
                        st.markdown(f"<h3 style='color:#c0392b;'>TESPİT EDİLEN BULGU SAYISI: {count}</h3>", unsafe_allow_html=True)
                        
                        # Pandas ile şık bir tablo oluşturalım
                        data = []
                        for i, box in enumerate(detections):
                            conf_score = float(box.conf)
                            # Koordinatları al (Opsiyonel, doktor bölgeyi bilsin diye)
                            coords = box.xyxy[0].tolist() 
                            
                            data.append({
                                "No": i+1,
                                "Sınıflandırma": "Dental Caries (Çürük)",
                                "Güven Skoru (%)": f"%{round(conf_score * 100, 1)}",
                                "Durum": "İnceleme Önerilir"
                            })
                            
                        df = pd.DataFrame(data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        
                        st.warning("⚠️ DİKKAT: Yukarıdaki bulgular yapay zeka destekli ön tanıdır. Kesin teşhis için klinik muayene esastır.")
                    else:
                        st.success("✅ Görüntüde aktif patolojik bulgu saptanmamıştır.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)