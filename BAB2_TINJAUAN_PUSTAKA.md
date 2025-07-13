# BAB II TINJAUAN PUSTAKA

## 2.1 Passing Network dalam Sepak Bola

*Passing network* adalah representasi grafis dari pola umpan antar pemain dalam sebuah tim sepak bola. Setiap pemain direpresentasikan sebagai *node* (titik), sedangkan umpan antar pemain digambarkan sebagai *edge* (garis) yang menghubungkan node-node tersebut. Analisis passing network dapat membantu pelatih dan analis dalam memahami struktur permainan, menemukan pemain kunci, serta mengidentifikasi pola distribusi bola yang efektif (Gudmundsson & Horton, 2017).

![Gambar 2.1 Contoh Visualisasi Passing Network](https://raw.githubusercontent.com/statsbomb/open-data/master/img/passing_network_example.png)
**Gambar 2.1** Contoh visualisasi *passing network* pada tim sepak bola.  
*Sumber: StatsBomb Open Data*

Passing network dapat dianalisis menggunakan berbagai metrik graf, seperti *degree centrality*, *betweenness centrality*, dan *clustering coefficient* untuk mengukur peran dan pengaruh setiap pemain dalam distribusi bola.

---

## 2.2 Analisis Taktik Sepak Bola Modern

Analisis taktik sepak bola modern tidak hanya mengandalkan pengamatan visual, tetapi juga memanfaatkan data spatio-temporal untuk memahami pergerakan dan interaksi pemain di lapangan (Lucey et al., 2013). Dengan data event dan tracking, pelatih dapat mengidentifikasi pola serangan, pertahanan, serta transisi yang terjadi selama pertandingan.

---

## 2.3 Kecerdasan Buatan (*Artificial Intelligence*) dan *Conditional GAN*

Kecerdasan buatan (*Artificial Intelligence* / *AI*) telah banyak digunakan dalam analisis olahraga, termasuk sepak bola. Salah satu pendekatan AI yang populer adalah *Generative Adversarial Networks* (*GAN*), yang mampu menghasilkan data baru yang menyerupai data asli (Goodfellow et al., 2014).

*Conditional GAN* (*CGAN*) merupakan pengembangan dari GAN yang memungkinkan proses generasi data dilakukan berdasarkan kondisi tertentu, seperti formasi tim, gaya bermain, atau tingkat kreativitas (Mirza & Osindero, 2014).

![Gambar 2.2 Arsitektur Conditional GAN](https://miro.medium.com/v2/resize:fit:720/format:webp/1*1Qw2vQKpQnXl8n5nqk6QJw.png)
**Gambar 2.2** Arsitektur *Conditional GAN* (*CGAN*).  
*Sumber: Mirza & Osindero, 2014*

Pada gambar di atas, *generator* menghasilkan data passing network berdasarkan kondisi (*condition*), sedangkan *discriminator* bertugas membedakan antara data asli dan data hasil generator.

---

## 2.4 Visualisasi Data Olahraga

Visualisasi data sangat penting dalam analisis sepak bola modern. Dengan visualisasi yang tepat, pola dan insight yang tersembunyi dalam data dapat lebih mudah dipahami dan diinterpretasikan. Beberapa visualisasi yang umum digunakan antara lain *passing network*, *heatmap* posisi pemain, *radar chart* performa tim, dan *pie chart* distribusi formasi.

![Gambar 2.3 Contoh Visualisasi Dashboard Passing Network](https://user-images.githubusercontent.com/123456789/football-dashboard-example.png)
**Gambar 2.3** Contoh dashboard visualisasi passing network dan metrik performa tim.  
*Sumber: Dokumentasi pribadi (hasil pengembangan penelitian)*

---

## 2.5 Penelitian Terdahulu

Penelitian terkait passing network dan penerapan AI dalam sepak bola telah banyak dilakukan. Tabel berikut merangkum beberapa penelitian terdahulu yang relevan:

**Tabel 2.1 Ringkasan Penelitian Terdahulu Passing Network dan AI dalam Sepak Bola**

| No | Penulis (Tahun)         | Judul Penelitian                                      | Metode                | Data                | Hasil Utama                                   | Relevansi/Kritik                |
|----|------------------------|-------------------------------------------------------|-----------------------|---------------------|-----------------------------------------------|----------------------------------|
| 1  | Gudmundsson & Horton (2017) | Spatio-Temporal Analysis of Team Sports             | Network Analysis      | EPL, La Liga        | Passing network efektif untuk analisis taktik | Belum menggunakan AI             |
| 2  | Lucey et al. (2013)    | Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic Features from Spatiotemporal Data | Spatiotemporal Analysis | EPL                | Fitur passing network meningkatkan prediksi   | Belum ada generative model       |
| 3  | Goodfellow et al. (2014) | Generative Adversarial Nets                          | GAN                   | MNIST, CIFAR        | GAN efektif menghasilkan data baru            | Belum diterapkan di sepak bola   |
| 4  | Mirza & Osindero (2014) | Conditional Generative Adversarial Nets              | CGAN                  | MNIST               | CGAN dapat menghasilkan data terkontrol       | Potensi untuk passing network    |
| 5  | Decroos et al. (2019)  | Actions Speak Louder than Goals: Valuing Player Actions in Soccer | VAEP Model           | Eredivisie          | Penilaian aksi pemain berbasis data event     | Fokus pada valuasi aksi, bukan passing network |

*Sumber: Diolah dari berbagai literatur*

---

## 2.6 Kerangka Pemikiran

Kerangka pemikiran penelitian ini digambarkan pada diagram berikut:

![Gambar 2.4 Diagram Alur Penelitian](https://user-images.githubusercontent.com/123456789/flowchart-passing-network-cgan.png)
**Gambar 2.4** Diagram alur penelitian:  
Akuisisi data → Preprocessing → Training CGAN → Generasi passing network → Visualisasi dashboard → Evaluasi hasil.

Diagram ini menunjukkan tahapan utama penelitian, mulai dari pengumpulan data pertandingan EPL, preprocessing data, pelatihan model CGAN, hingga visualisasi passing network dalam dashboard interaktif dan evaluasi hasil.

---

**Referensi:**
- Gudmundsson, J., & Horton, M. (2017). Spatio-Temporal Analysis of Team Sports. *ACM Computing Surveys*, 50(2), 1–34.
- Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2013). Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic Features from Spatiotemporal Data. *MIT Sloan Sports Analytics Conference*.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*.
- Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv preprint arXiv:1411.1784*.
- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions Speak Louder than Goals: Valuing Player Actions in Soccer. *Data Mining and Knowledge Discovery*, 33(4), 1310–1335.
- StatsBomb Open Data. https://github.com/statsbomb/open-data
- Football-Data.co.uk. https://www.football