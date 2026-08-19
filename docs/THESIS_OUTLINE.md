# Tez iskeleti · Düzce Üniversitesi BM498

Şablon: `bcb84bf8-…​.docx`. Bu belge şablonun bölüm yapısını, elimizdeki kanıtla ve
üretilmiş figürlerle eşler. Her satırda hangi kaydın kullanılacağı yazılıdır; kanıtı
olmayan hiçbir bölüm doldurulmayacaktır.

Referans uzunluk şablonda **30 sayfa**. Kaynak biçimi **ISO 690**. Şekiller `Şekil X.Y.`,
çizelgeler `Çizelge X.Y.` olarak numaralanır ve **şekle atıf şekilden önce** verilir.

---

## Ön kısım (şablonda hazır, doldurulacak)

| sayfa | not |
|---|---|
| Kapak | başlık, ad, öğrenci no, ders sorumlusu |
| Değerlendirme tutanağı | jüri dolduracak |
| **BEYAN** | intihal beyanı — imzalanacak |
| **ÜRETKEN YAPAY ZEKA KULLANIM BEYANI** | **zorunlu** · aşağıya bakınız |
| TEŞEKKÜR | serbest |
| İÇİNDEKİLER / ŞEKİL / ÇİZELGE LİSTESİ | Word otomatik üretir |
| KISALTMALAR | mIoU, ECE, OOD, AUROC, AP, FPR95, TensorRT, ONNX, FP16, ACDC |
| SİMGELER | ρ (Spearman), J (joule), W (watt) |
| **ÖZET** | tek paragraf, en fazla 1 sayfa · mevcut özet paragrafın temel alınacak |
| **ABSTRACT** | aynısının İngilizcesi |

### Yapay zekâ beyanı

Şablon bunu **zorunlu tutuyor**, yani gizlenecek bir şey yok — beyan edilecek bir şey var.
`docs/AI_USAGE_LOG.md` her materyal değişikliğin tarihini, dosyasını, testini ve
gerekçesini taşıyor; beyanı bu loga dayandır. Yazılan metin senin olmalı: sayılar zaten
senin ölçümlerinden geliyor, cümleleri de sen kur.

---

## 1 · GİRİŞ (≈3 sayfa)

- Problem: uç cihazda algı sistemi yalnızca tahmin üretmemeli; ne kadar emin olduğunu,
  bilinmeyenle karşılaşıp karşılaşmadığını ve hangi bölgenin önemli olduğunu bildirmeli.
- Motivasyon: yayınlanmış doğruluk sıralamalarının dağıtımda geçerli olup olmadığı
  bilinmiyor; kalibrasyon ve açık küme başarımı model seçiminde kullanılmıyor.
- Katkılar (dört madde, hepsi ölçülmüş):
  1. Beş gerçek zamanlı mimarinin **dağıtım çözünürlüğünde** doğruluk, kalibrasyon ve
     açık küme karşılaştırması
  2. Yayınlanmış sıralamanın dağıtım sıralamasını öngörmediğinin gösterilmesi
  3. Uç cihazda kare bütçesinin hızlandırıcıda değil CPU tarafında harcandığının
     ölçümle atfedilmesi ve iki hedefli optimizasyonla 1,88× hızlanma
  4. Gerçek olumsuz koşullarda (ACDC) kalibrasyonun çöktüğünün ve gecede sessiz
     başarısızlık oluştuğunun gösterilmesi
- **Literatür taraması (rubrik: 5 puan, PRISMA vb. sistematik yaklaşım).** ⚠ **Henüz
  yapılmadı.** Yapılacak: gerçek zamanlı semantik segmentasyon · belirsizlik/kalibrasyon ·
  açık küme segmentasyon · uç cihaz dağıtımı başlıkları, PRISMA akış şemasıyla.

## 2 · MATERYAL VE YÖNTEM (≈8 sayfa)

### 2.1 Veri setleri (rubrik: FAIR — 5 puan)

| veri | rol | boyut | kaynak |
|---|---|---|---|
| Cityscapes val | doğruluk, sınıf bazlı IoU, ECE | 500 kare | resmî |
| ACDC val | gerçek olumsuz koşul | 406 kare (sis 100 · gece 106 · yağmur 100 · kar 100) | resmî |
| RoadAnomaly | açık küme yol tehlikesi | 60 kare, piksel etiketli | Lis ve ark., EPFL CVLab |
| Cityscapes+IDD20K | kendi eğitimimiz | dondurulmuş manifest | resmî |
| Cityscapes demoVideo | zamansal/nitel gösterim | 180 ardışık kare | resmî |

**FAIR argümanı:** her veri seti dondurulmuş manifestle (`*.frozen.json`), sha256
özetleriyle ve rol ayrımıyla (`train_fit`/`train_select`/`train_calibration`) yönetildi;
etiket dönüşümü `cityscapesscripts` tablosundan alınıp ACDC'nin kendi trainId'lerine karşı
40 karede doğrulandı (`scripts/prepare_cityscapes_trainids.py`).

### 2.2 Modeller

Beş gerçek zamanlı mimari, mmsegmentation model zoo referans checkpoint'leri (Apache-2.0),
Cityscapes'te 120k–160k adım eğitilmiş. Ayrıca kendi sınırlı bütçeli çok-domainli
eğitimimiz (2.500 adım). **İki hat ayrı tutulur, karıştırılmaz.**

### 2.3 Sistem mimarisi (rubrik: blok şeması — 5 puan)

→ **Şekil 2.1** = `D1_system_architecture.pdf`

### 2.4 Ölçüm protokolü

→ **Şekil 2.2** = `D2_measurement_protocol.pdf`
→ **Şekil 2.3** = `D3_three_axis_framework.pdf`

Donanım: Jetson Orin Nano Super · 25 W · TensorRT 10.3.0 FP16 · JetPack 6 (L4T 36) ·
600 sn sürdürülen yük · 200 kare ısınma · tam telemetri.

### 2.5 Metrikler

mIoU · sınıf bazlı IoU · piksel doğruluğu · ECE ve güvenilirlik eğrisi · AUROC/AP/FPR95 ·
bootstrap %95 GA · gecikme (medyan/p95/p99) · sürdürülen FPS · ortalama/tepe güç ·
joule/kare · tepe RAM · sıcaklık.

## 3 · UYGULAMA VE KARŞILAŞILAN SORUNLAR (≈5 sayfa)

**Rubrik: "uyumsuzluklar ve aksaklıklar belirtilerek çözüm yöntemleri tartışılmış mı?"
(5 puan).** Bu bölüm projenin en güçlü kanıt havuzu — gerçek, belgelenmiş hatalar:

| sorun | kök neden | çözüm |
|---|---|---|
| Değerlendirme rastgele ağırlık ölçüyordu | `Runner.from_cfg()` `cfg.load_from`'u yüklemiyor | `load_evaluation_weights()` + bayt doğrulaması |
| ONNX pariteni geçemiyordu | TF32 CUDA ↔ FP32 CPU karşılaştırması | parite CPU'da, sınıf haritası eşdeğerliği |
| HPO erken pes ediyordu | PRUNED denemeler tamamlanmış sayılıyordu | `complete_trials_within_budget` |
| Log paketi sessizce üretilmiyordu | ZIP 1980 öncesi zaman damgası taşıyamıyor | mtime ileri kırpma |
| AUROC gerçek runtime'da çökerdi | `np.trapezoid` numpy ≥2.0'a özel, runtime 1.26.4 | yamuk kuralı doğrudan yazıldı |
| Eşik politikaları 90 dakika sürüyordu | O(n²) tarama | kümülatif sayımla vektörleştirme |
| Jetson benchmark 10 dakikayı boşa harcıyordu | telemetri kontrolü ölçümden *sonra* | ön koşula alındı |
| Optimizasyon kararı yanlış makinede verildi | Mac ↔ ARM'de Python/NumPy oranı farklı | hedef cihazda A/B ölçümü |

**Rubrik: proje/risk/değişiklik yönetimi (4 puan)** → `docs/AI_USAGE_LOG.md` (append-only),
ADR'ler, git geçmişi, CI.

**Rubrik: yeniden üretilebilirlik (3 puan)** → sabit tohum (20260728), dondurulmuş
manifestler, sha256 bağlı kayıtlar, 626 birim testi, pinlenmiş çalışma zamanı.

## 4 · OPTİMİZASYON (≈4 sayfa)

Ölç → ata → düzelt → yeniden ölç döngüsü, üç kez:

→ **Şekil 4.1** = `D4_frame_budget_flow.pdf`
→ **Şekil 4.2** = `07_frame_budget.pdf`

1. Mesafe dönüşümü ayrılabilir hâle getirildi: **34×** (9,20 → 0,27 ms)
2. Bileşen etiketleme vektörleştirildi: Jetson'da **1,59×** *(karar hedef cihazda verildi)*
3. On bir sınıf geçişi tek geçişe indirildi: **3,90×** (15,45 → 3,96 ms)

Sonuç: kare 151,67 → 80,77 ms (**1,88×**), enerji 1,084 → 0,665 J/kare (**1,63×**),
motora dokunulmadan, çıktılar birebir aynı.

## 5 · BULGULAR VE TARTIŞMA (≈7 sayfa)

| içerik | şekil/çizelge |
|---|---|
| Yayın vs dağıtım doğruluğu, sıralama değişimi | **Şekil 5.1** = `01_published_vs_measured_miou` |
| Sınıf bazlı IoU | **Şekil 5.2** = `02_per_class_iou` |
| Güvenilirlik diyagramları | **Şekil 5.3** = `03_reliability_diagrams` |
| Gerçek olumsuz koşullar | **Şekil 5.4** = `04_acdc_conditions` |
| Açık küme skorları ve tehlike kırılımı | **Şekil 5.5** = `05_open_set` |
| Pareto | **Şekil 5.6** = `06_pareto` |
| Kesin yanlış piksellerde güven | **Şekil 5.7** = `09_overconfidence` |
| Sentetik bozulmaya tepki | **Şekil 5.8** = `08_synthetic_shift` |
| Nitel: beş mimari aynı karede | **Şekil 5.9** = `qualitative/models_same_frame` |
| Nitel: temiz/sis/gece | **Şekil 5.10** = `qualitative/conditions_pidnet_s` |
| Üç eksenli karşılaştırma | **Çizelge 5.1** |
| Jetson ölçümleri | **Çizelge 5.2** |
| ACDC koşulları | **Çizelge 5.3** |

### Ana bulgular

1. **Yayınlanmış sıralama dağıtımı öngörmüyor** (ρ = +0,10, n=5). Dağıtım
   çözünürlüğünde SegFormer-B0 sınıf-ortalamalı mIoU'da öne geçiyor, yayında sonuncuydu.
2. **Doğrulukta kazanan yok.** Aynı 500 karede eşleştirilmiş karşılaştırma ilk iki modeli
   **ayırt edemiyor** (+0,0000 [−0,0050, +0,0049]); on çiftin sekizi ayrışıyor, ilk ikisi
   ayrışmıyor. SegFormer-B0'ın sınıf-ortalamalı 0,0085'lik üstünlüğü 19 sınıfın 14'ünde
   biriken küçük kazançlardan, çoğu **ince yapıdan** geliyor (direk +0,057, trafik ışığı
   +0,035, insan +0,033). *(§9a)*
3. **Çıktı stride'ı beş ayrı ölçümde aynı yere çıkıyor** — sınıf bazlı IoU, sınır F1,
   sınıf-bazlı ECE, bileşen parçalanması ve sürülebilir alan. Bedeli **giderilemez**:
   logitleri stride-8'e indirmek post-processing'i 3,98× hızlandırıyor ama 2,03 mIoU'ya
   mal oluyor. *(§2b, §3a, §4, §8)*
4. **Rakip açıklama ölçüldü ve elendi.** Nadirlik gerçek bir etken (ρ(pay, IoU) = +0,68)
   ama yeterli değil: `pole`, `motorcycle`'dan 18 kat daha sık olmasına rağmen kalibrasyon
   hatası 9 kat kötü. Ayıran şey nadirlik değil **şekil**. *(§2b-2)*
5. **Gecede sessiz başarısızlık**: mIoU 0,1497, piksel doğruluğu %45,2, ortalama güven
   %71,9, ECE 0,2694 (en iyi koşulun 8,7 katı). Model en çok yanıldığı koşulda
   yanıldığını bilmiyor. *(§6a)*
6. **Kayıp yük körlüğü**: açık küme AUROC 0,4805 — rastgeleden kötü. *(§5)*
7. **Havuzlanmış ECE yanıltıyor**: sınıf-bazlı hesapta hata **1,5–3 kat** büyüyor; `road`
   piksellerin %39'u ve havuzlanmış sayıyı o taşıyor. *(§4)*
8. **Motor kare bütçesinin %5,4'ü**; gerçek zamanlılık CPU tarafında kazanılıyor. *(§8)*
9. **FP16 dağıtımın ölçülebilir bedeli yok** (eşleştirilmiş fark sıfırdan ayırt edilemiyor,
   piksel uyumu %99,97) — ve eşleştirme yapılmasaydı DDRNet için **var olmayan** bir
   −0,0253 cezası raporlanacaktı. *(§7a)*
10. **Sürülebilir alanda ego koridoru adımının karşılığı ölçüldü**: yanlış-sürülebilir
    piksellerin %18,8–26,2'si eleniyor, ~0,55 puan IoU karşılığında. *(§3a)*
11. **Tek-kare dikkat baskın davranış olarak titriyor**: izlerin %50,1'i tek kare yaşıyor,
    medyan iz ömrü 1 kare. *(§3b)*

### Tartışma sınırları (dürüstlük)

- n = 5 mimari; bu ölçekte hiçbir korelasyon kesin kanıt değildir. Nitekim ρ = −0,90
  bulgusu bu yüzden yanlış okundu ve geri çekildi (bkz. THESIS_EVIDENCE §9).
- Doğrulukta ilk iki model **kare düzeyinde ayrışmıyor** (+0,0000 [−0,0050, +0,0049],
  500 eşleştirilmiş kare); "en doğru model" ifadesi kullanılmayacak.
- Referans checkpoint'ler farklı reçetelerle eğitilmiştir; kontrollü ablasyon değildir.
- Kendi eğitimimiz yayınların %0,7'si kadar örnek görmüştür.
- Sentetik bozulmalar algoritmiktir; gerçek ACDC ölçümleri ayrıca verilmiştir.

## 6 · SONUÇLAR VE ÖNERİLER (≈2 sayfa)

- Uç cihazda model seçimi yayınlanmış mIoU'ya göre yapılamaz.
- Sistem maliyetini FLOP değil segmentasyon başının **çıktı stride'ı** belirler.
- Gece koşulu ayrı ele alınmalı ya da belirsizlik sinyali aydınlatmaya duyarlı hâle
  getirilmelidir.
- **Ölçüldü:** SegFormer logitlerini stride-8'e indirmek post-processing'i 3,98×
  hızlandırır ama 2,03 mIoU'ya mal olur. Yüksek çözünürlüklü logit gerçek doğruluk
  taşıyor; uç cihaz için soru "maliyet kaldırılabilir mi" değil, "bu doğruluk bu enerjiye
  değer mi".
- **Sürdürülebilirlik (rubrik 4 puan):** joule/kare doğrudan ölçüldü; DDRNet seçimi
  PIDNet-M'e göre kare başına %22 enerji tasarrufu sağlar.
- **Girişimcilik/yenilikçilik (rubrik 4 puan):** üç eksenli seçim yöntemi ve stride
  bulgusu, uç cihaz dağıtımı yapan herkes için doğrudan uygulanabilir.

## 7 · KAYNAKLAR (ISO 690)

Asgari: Cityscapes · IDD · ACDC · RoadAnomaly (Lis ve ark.) · PIDNet · DDRNet ·
SegFormer · BiSeNetV2 · mmsegmentation · TensorRT · enerji tabanlı OOD · kalibrasyon
(Guo ve ark.) · ECE.

## 8 · EKLER

- EK 1: Ölçüm kayıtlarının şeması ve sha256 listesi
- EK 2: Karşılaşılan hatalar ve regresyon testleri
- EK 3: Yeniden üretim talimatları

---

## Rubrik karşılığı

| kriter | puan | durum |
|---|---|---|
| Problem tanımı | 5 | ✅ |
| **Literatür taraması (PRISMA)** | 5 | ✅ Şekil 1.1, sayılar türetilir |
| **Blok şeması** | 5 | ✅ D1 |
| Materyal ve metot | 5 | ✅ |
| **FAIR veri yönetimi** | 5 | ✅ güçlü |
| **Aksaklıklar ve çözümler** | 5 | ✅ çok güçlü |
| Zorluk derecesi | 5 | ✅ |
| Etik / intihal / AI beyanı | 5 | ✅ log mevcut |
| Mühendislik etiği | 5 | ✅ |
| **Standartlar (ISO/IEEE)** | 5 | ✅ THESIS_SECTIONS §C4 |
| **Proje/risk yönetimi** | 4 | ✅ log + ADR |
| Girişimcilik | 4 | ✅ THESIS_SECTIONS §C5 |
| Sürdürülebilirlik | 4 | ✅ THESIS_SECTIONS §C6, ölçülmüş joule |
| **Yeniden üretilebilirlik** | 3 | ✅ çok güçlü |

**Rubrikte açık kalan boşluk yok.** PRISMA akışı Şekil 1.1'dir ve sayıları
`audit_literature_corpus.py` türetir; standartlara atıf, girişimcilik, sürdürülebilirlik
ve özet/abstract `THESIS_SECTIONS.md`'dedir.
