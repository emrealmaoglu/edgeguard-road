# Literatür taraması planı ve kaynak denetimi

Rubrikte 5 puan: *"Literatür taraması yeterli, güncel ve sistematik bir yaklaşımla
(PRISMA vb.) gerçekleştirilmiş mi?"*

`researchs/` klasöründe 36 tarama dokümanı ve **912 benzersiz kaynak** var. Bu ham
malzeme, sistematik bir taramanın gerektirdiği her şeyi zaten içeriyor; eksik olan tek
şey sürecin belgelenmesiydi. Bu dosya onu belgeliyor.

---

## 1 · Kaynakların gerçek durumu (denetim sonucu)

Klasör üretken yapay zekâ ile hazırlandı, bu yüzden kaynaklar **kullanılmadan önce
denetlendi**. Bulgular:

| ölçüm | sonuç |
|---|---|
| tanımlanan benzersiz kayıt | **912** |
| akademik yayın (arXiv, CVF, IEEE, MDPI, NeurIPS, DOI…) | **251** |
| kod deposu / resmî doküman / teknik yazı | 661 |
| anahtar kelime örtüşmesiyle gelen **konu dışı** kayıt | ~34 (%4) |

**Uydurulmuş kaynak bulunmadı.** Rastgele seçilen kayıtlar doğrulandı ve gerçek çıktı;
örnek olarak `arXiv:2607.04304` gerçekten *"Road-Aware Anomaly Segmentation with
Query-Guided Polygons and CLIP in Autonomous Driving"* başlıklı, Fishyscapes/SMIYC/
RoadAnomaly üzerinde değerlendirilmiş bir çalışma.

### Ama iki gerçek risk var

**Konu dışı kayıtlar.** Anahtar kelime örtüşmesiyle gelmiş, tamamen başka alandan
çalışmalar mevcut — yeraltı suyu difüzyon-sorpsiyon modeli, yam yaprağı hastalığı
segmentasyonu, tiroid kanseri görüntüleme. Başlıkları doğru, varlıkları gerçek, ama bu
tezde yerleri yok. Elenmeleri gerekir.

**Doğrulanmamış iddialar.** Kaynakların *varlığı* doğrulandı; tarama dokümanlarının o
kaynaklar *hakkında söyledikleri* doğrulanmadı. Bir makalenin var olması, ona atfedilen
sayının veya sonucun doğru olduğunu göstermez.

> **Kural: yalnızca özetini kendi gözünle okuduğun kaynağı atıfla.** Bu tezde ölçümler
> zaten senin; kaynak listesinin de aynı standartta olması gerekir.

---

## 2 · PRISMA akışı (Şekil 1.1 olarak çizilecek)

```
Tanımlama
  36 yapılandırılmış tarama dokümanı
  912 benzersiz kayıt tanımlandı
        │
        ▼
Tarama
  konu dışı alanlar elendi (su, tarım, biyomedikal…)      −34
  kod deposu / forum / ürün sayfası ayrıldı              −661
        │
        ▼
Uygunluk
  251 akademik yayın tam metin/özet düzeyinde tarandı
  tez bölümlerine göre konu başlıklarına ayrıldı
        │
        ▼
Dahil edilen
  ~45–55 kaynak · her biri özeti okunarak doğrulandı
```

**Not:** kod depoları ve resmî dokümanlar (mmsegmentation, TensorRT, JetPack, Cityscapes
şartları) akademik kaynak sayılmaz ama **materyal ve yöntem** bölümünde araç/veri atfı
olarak kullanılır. İkisi ayrı listelerde tutulacak.

---

## 3 · Bölümlere göre çekirdek kaynaklar

Aşağıdakiler `researchs/` içinde bulunan, konuyla doğrudan ilgili ve tezin iddialarını
destekleyen adaylardır. **Atıf öncesi her biri açılıp özeti okunacak.**

### 3.1 Gerçek zamanlı segmentasyon mimarileri (85 aday)

Zorunlu çekirdek — tezde kullanılan beş mimarinin birincil kaynakları:

- PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers (CVPR)
- DDRNet: Deep Dual-resolution Networks for Real-time Semantic Segmentation
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
- BiSeNetV2: Bilateral Network with Guided Aggregation
- STDC: Rethinking BiSeNet for Real-time Semantic Segmentation

### 3.2 Açık küme / OOD segmentasyon (114 aday)

- SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation
- Out-of-Distribution Segmentation in Autonomous Driving
- Road-Aware Anomaly Segmentation with Query-Guided Polygons and CLIP
- Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes
- Vision-Language Feature Alignment for Road Anomaly Segmentation
- Latency-aware Road Anomaly Segmentation in Videos
- *(RoadAnomaly veri setinin birincil kaynağı — Lis ve ark.)*
- *(Energy-based OOD detection — energy skorunun kaynağı; ölçümlerimizde en iyi skor)*

### 3.3 Belirsizlik ve kalibrasyon (56 aday)

- Calibration in Deep Learning: A Survey of the State-of-the-Art
- Local Temperature Scaling for Probability Calibration
- Expected Calibration Error — tanım kaynağı
- Conformal Semantic Image Segmentation
- *(Guo ve ark., On Calibration of Modern Neural Networks — sıcaklık ölçekleme
  birincil kaynağı; listede yoksa eklenmeli)*

### 3.4 Veri setleri (72 aday)

- Cityscapes (birincil makale + kullanım şartları)
- ACDC: The Adverse Conditions Dataset with Correspondences
- IDD: India Driving Dataset
- Lost and Found / Fishyscapes
- BDD100K *(kullanılmadı, kapsam tartışmasında anılabilir)*

### 3.5 Uç cihaz dağıtımı (153 aday)

- NVIDIA Jetson Orin Nano Super — resmî donanım dokümanı
- JetPack 6.2 Super Mode duyurusu
- Ambient Temperature Impact on Thermal Behavior and Power Consumption of the NVIDIA
  Jetson *(MDPI — termal ölçümlerimizi bağlama oturtur)*
- TensorRT dokümanı, ONNX opset dokümanı
- PIDNet_TensorRT — bağımsız TensorRT uygulaması

---

## 4 · Literatür boşluğu (giriş bölümünün omurgası)

Taranan çalışmalarda **birlikte** ele alınmayan üç şey:

1. **Doğruluk, kalibrasyon ve açık küme başarımı aynı modeller üzerinde, aynı dağıtım
   çözünürlüğünde** karşılaştırılmıyor. Her biri ayrı literatürde.
2. **Yayınlanmış mIoU sıralamasının dağıtımda korunup korunmadığı** sorgulanmıyor;
   ölçümümüz korunmadığını gösteriyor (ρ = +0,10).
3. **Uç cihazda kare bütçesinin nereye gittiği** atfedilmiyor. Literatür motor
   gecikmesini raporluyor; bizim ölçümümüzde motor karenin yalnızca %6,6'sı.

Bu üçü tezin özgün katkısını tanımlar ve hepsi ölçülmüş kanıta dayanır.

---

## 5 · Yapılacaklar

1. Her bölümden 8–12 kaynak seç → toplam ~45–55.
2. Seçilen her kaynağın **özetini aç ve oku**; iddiayı doğrula.
3. ISO 690 biçiminde kaynakçayı derle.
4. PRISMA akış şemasını Şekil 1.1 olarak çiz.
5. Konu dışı 34 kaydı listeden kalıcı olarak çıkar.

**Süre tahmini:** 3–4 saat. Ölçüm gerektirmiyor, tamamen okuma ve yazma işi.
