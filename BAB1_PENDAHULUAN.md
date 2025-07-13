# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Sepak bola modern telah berkembang menjadi olahraga yang sangat mengandalkan data dan analisis taktik untuk meningkatkan performa tim. Salah satu pendekatan analisis yang kini banyak digunakan adalah *passing network*, yaitu representasi pola umpan antar pemain dalam bentuk graf. Melalui *passing network*, pelatih dan analis dapat memahami struktur permainan, menemukan kelemahan lawan, serta mengidentifikasi pemain kunci dalam distribusi bola (Gudmundsson & Horton, 2017).

Di Liga Inggris (*English Premier League* / EPL), analisis passing network semakin penting karena tingginya intensitas dan variasi taktik yang diterapkan oleh setiap tim (Lucey et al., 2013). Namun, proses analisis manual membutuhkan waktu dan keahlian khusus, serta seringkali tidak mampu menangkap pola dinamis dan variasi strategi yang terjadi sepanjang musim.

Seiring berkembangnya teknologi kecerdasan buatan (*Artificial Intelligence* / AI), khususnya *Generative Adversarial Networks* (GAN) dan turunannya *Conditional GAN* (CGAN), kini dimungkinkan untuk melakukan simulasi dan prediksi passing network secara otomatis dan inovatif (Goodfellow et al., 2014; Mirza & Osindero, 2014). Dengan memanfaatkan CGAN, pola passing dapat dihasilkan berdasarkan kondisi tertentu seperti formasi, gaya taktik, dan tingkat kreativitas tim, sehingga analisis taktik menjadi lebih fleksibel dan adaptif.

Penelitian ini berfokus pada pengembangan dashboard visualisasi passing network berbasis AI (CGAN) untuk tim-tim EPL musim 2024/2025. Dengan dashboard ini, diharapkan pelatih, analis, maupun peneliti dapat melakukan eksplorasi dan simulasi taktik secara interaktif, serta memperoleh insight yang lebih mendalam terhadap pola permainan tim.

## 1.2 Rumusan Masalah

Rumusan masalah dalam penelitian ini adalah sebagai berikut:
1. Bagaimana membangun model *Conditional GAN* (CGAN) untuk menghasilkan passing network tim-tim EPL berdasarkan data pertandingan musim 2024/2025?
2. Bagaimana mengembangkan dashboard visualisasi passing network berbasis AI yang interaktif dan informatif?
3. Bagaimana membandingkan hasil passing network yang dihasilkan AI dengan data aktual pertandingan untuk menilai efektivitas dan realisme model?

## 1.3 Batasan Masalah

Agar penelitian lebih terarah, batasan masalah yang diterapkan adalah:
- Data yang digunakan terbatas pada pertandingan *English Premier League* musim 2024/2025, yang diambil dari sumber open data seperti [StatsBomb Open Data](https://github.com/statsbomb/open-data) dan [Football-Data.co.uk](https://www.football-data.co.uk/).
- Analisis hanya difokuskan pada passing network, tidak membahas aspek lain seperti dribbling, shooting, atau defensive action secara mendalam.
- Model AI yang digunakan adalah *Conditional GAN* (CGAN), tanpa membandingkan dengan arsitektur AI lain.
- Dashboard dikembangkan menggunakan bahasa pemrograman Python dengan library utama *Streamlit*, *PyTorch*, dan *Matplotlib*.

## 1.4 Tujuan Penelitian

Penelitian ini bertujuan untuk:
1. Mengembangkan model *Conditional GAN* (CGAN) yang mampu menghasilkan passing network tim-tim EPL berdasarkan data pertandingan aktual.
2. Membangun dashboard visualisasi passing network berbasis AI yang interaktif dan mudah digunakan.
3. Melakukan evaluasi terhadap hasil passing network AI dengan membandingkannya pada data aktual, serta menganalisis kelebihan dan keterbatasan model.

## 1.5 Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini adalah:
- **Manfaat Teoritis:** Menambah referensi dan literatur terkait penerapan AI, khususnya CGAN, dalam analisis taktik sepak bola berbasis passing network.
- **Manfaat Praktis:** Memberikan alat bantu analisis yang inovatif bagi pelatih, analis, dan peneliti sepak bola untuk mengeksplorasi dan mensimulasikan pola passing serta strategi tim secara interaktif.

## 1.6 Sistematika Penulisan

Sistematika penulisan skripsi ini adalah sebagai berikut:
- **Bab I Pendahuluan:** Berisi latar belakang, rumusan masalah, batasan masalah, tujuan, manfaat, dan sistematika penulisan.
- **Bab II Tinjauan Pustaka:** Membahas teori-teori dasar, penelitian terdahulu, serta kerangka pemikiran yang mendasari penelitian.
- **Bab III Metodologi Penelitian:** Menjelaskan metode, data, tahapan penelitian, serta tools yang digunakan.
- **Bab IV Hasil dan Pembahasan:** Menyajikan hasil implementasi, visualisasi dashboard, serta analisis dan evaluasi model.
- **Bab V Kesimpulan dan Saran:** Berisi kesimpulan dari penelitian dan saran untuk pengembangan lebih lanjut.

---

**Referensi:**
- Gudmundsson, J., & Horton, M. (2017). Spatio-Temporal Analysis of Team Sports. *ACM Computing Surveys*, 50(2), 1–34.
- Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2013). Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic Features from Spatiotemporal Data. *MIT Sloan Sports Analytics Conference*.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*.
- Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv preprint arXiv:1411.1784*.
- StatsBomb Open Data. https://github.com/statsbomb/open-data
- Football-