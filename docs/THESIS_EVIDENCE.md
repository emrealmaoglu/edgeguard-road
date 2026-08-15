# Tez kanıt envanteri

Özetteki her iddianın karşısına, o iddiayı destekleyen **ölçülmüş** kanıt ve kaydın
bulunduğu dosya yazılmıştır. Ölçülmemiş olanlar açıkça boşluk olarak işaretlidir; hiçbir
satır "beklenen" değeri "ölçülmüş" gibi göstermez.

Bütün sayılar 2026-08-15 itibarıyla gerçek koşulardan gelir. Donanım: NVIDIA Jetson Orin
Nano Super, 25 W güç modu, TensorRT 10.3.0 FP16, JetPack 6 (L4T 36, aarch64).

---

## 1 · Modeller ve doğruluk

**İddia:** *"hafif derin öğrenme modellerini geliştirerek ve karşılaştırarak"*

İki ayrı kanıt hattı var ve karıştırılmamalıdır.

### 1a · Kendi eğitilen modeller (deneysel katkı)

Cityscapes + IDD20K karışık, 2.500 adım, batch 4, 512×1024 crop, bf16, ImageNet
başlangıcı, `seed 20260728`, NVIDIA L4.

| model | domain-macro mIoU | Cityscapes | IDD20K | nadir sınıf mIoU |
|---|---|---|---|---|
| pidnet_s | 0,3700 | 0,4641 | 0,2759 | 0,1755 |
| ddrnet_23_slim | 0,3574 | 0,4310 | 0,2840 | 0,1346 |
| segformer_b0 | 0,2286 | 0,2723 | 0,1849 | **0,0000** |

Kaynak: `reports/screening/candidate_table.json`. Üçü de `onnx_validated: true`,
`rejected: []`.

**Sınıf bazlı IoU** eğitim loglarında tam tablo hâlinde mevcut
(`screening/<model>/ce/<zaman>/<zaman>.log`, "per class results"). ddrnet_23_slim örneği:
road 85,98 · sidewalk 50,22 · building 58,24 · sky 70,82 · car 58,67 · vegetation 62,58
· person 32,67 · bicycle 35,15 · bus 32,64 · wall 32,18 · motorcycle 24,98 · traffic sign
21,88 · rider 19,49 · truck 18,29 · fence 12,93 · traffic light 11,44 · pole 7,45 ·
**train 1,33**.

**Nadir sınıf bulgusu:** segformer_b0'ın nadir sınıf mIoU'su tam olarak sıfır. 2.500
adımlık bütçede nadir sınıfların hiçbirini öğrenememiş. Sınıf dengesizliğinin mimariye
göre farklı vurduğunun somut kanıtı.

### 1b · Referans checkpoint'ler (karşılaştırma temeli)

mmsegmentation model zoo, Cityscapes'te 120k–160k adım eğitilmiş, Apache-2.0.
Kendi eğitimimiz bunların **%0,7'si kadar** örnek gördü (10.000'e karşı 1.440.000).

| model | mIoU (Cityscapes val) |
|---|---|
| PIDNet-M | 80,22 |
| PIDNet-S | 78,74 |
| DDRNet-23-slim | 77,84 |
| SegFormer-B0 | 76,54 |
| BiSeNetV2 | 75,76 |

**Geçersiz sayılar:** eski kampanyanın 16,48 / 20,53 / 26,54 / 25,21 / 17,41 değerleri
kullanılmayacak. O değerlendirme rastgele ağırlık ölçüyordu — `Runner.from_cfg()`
`cfg.load_from`'u yüklemez, bu yalnızca `train()/val()/test()` içinde olur. Hata
`load_evaluation_weights()` ile düzeltildi.

---

## 2 · Sınıf algılama ve konumlandırma

**İddia:** *"yol, kaldırım, araç, yaya, bisiklet, trafik işareti ve benzeri çevresel
sınıfları algılaması; önemli nesne ve bölgeleri görüntü üzerinde konumlandırması"*

19 Cityscapes sınıfı. Bölge çıkarımı `derive_perception` ile; her bölge için sınıf,
sınırlayıcı kutu, ağırlık merkezi, alan, ortalama güven, ortalama entropi, koridora
uzaklık ve piksel maskesi üretiliyor (`SemanticRegion`).

Kanıt: `predict.py --emit-regions` çıktıları, `results/figures/`.

---

## 3 · Sürülebilir alan ve risk bölgeleri

**İddia:** *"sürülebilir alan ve potansiyel risk bölgelerini belirlemesi"*

- **Sürülebilir alan:** `drivable_corridor_from_semantics` — yol pikselleri arasından
  alt-orta bölgeye bağlı bileşen seçiliyor (`road_mask.png`, `drivable_corridor.png`).
- **Risk bölgeleri:** yedi özellikli açıklanabilir füzyon (`contextual_risk`).

Gerçek tehlike karelerinde ölçülmüş sıralama (PIDNet-S referans, RoadAnomaly):

| kare | 1. sıra | risk | seviye | baskın etken |
|---|---|---|---|---|
| animals19_Porte_de_Roubaix | bicycle | 0,813 | high | anomaly_score |
| animals06_sheep_roads_lambs | person | 0,806 | high | anomaly_score |
| obstacles12_rocks5 | rider | 0,819 | high | anomaly_score |
| frankfurt_000000_002196 (temiz şehir) | person | 0,885 | high | anomaly_score |

Kayıt: `results/risk/*.json`.

**Dürüstlük sınırı:** `detector_overlap` (nesne dedektörü yok) ve `temporal_persistence`
(tek kare) **sıfır ağırlıkla** dışlandı, sıfır değerle değil. Sıfır değer verilseydi
ölçülmemiş bir sinyal "risk yok" gibi görünür ve bütün skorları aşağı çekerdi. Kayıt
uygulanan ve dışlanan ağırlıkları ayrı ayrı yazıyor. Çıktı
`calibrated_physical_risk_probability: false` — bu bir operasyonel dikkat sıralamasıdır,
fiziksel risk olasılığı değil.

---

## 4 · Güven ve piksel düzeyinde belirsizlik

**İddia:** *"tahmin güvenini ve piksel düzeyindeki belirsizliği ölçmesi"*

Dört skor üretiliyor: maximum softmax probability, normalize entropi, maximum logit,
energy (`uncertainty_maps`).

### Kalibrasyon bulgusu

Anomali pikselleri 19 sınıfın hiçbiri değildir; model ne tahmin ederse etsin **kesinlikle
yanılır**. O piksellerdeki güveni:

| model | anomali px güven | normal px güven | oran |
|---|---|---|---|
| pidnet_s | 0,7850 | 0,8214 | 0,956× |
| pidnet_m | 0,7934 | 0,8397 | 0,945× |
| segformer_b0 | 0,8404 | 0,9106 | 0,923× |
| ddrnet_23_slim | 0,7310 | 0,8060 | 0,907× |
| bisenetv2 | 0,7304 | 0,8165 | 0,895× |

Model, **kesin yanıldığı yerlerde bile %73–84 güven** veriyor; güveni yalnızca %4,4–10,5
düşüyor. Ciddi aşırı-güven. PIDNet-S en aşırı güvenli, BiSeNetV2 en dürüst.

Bu, MSP'nin neden zayıf bir OOD skoru olduğunu (AUROC 0,569) ve energy'nin neden daha iyi
olduğunu (0,667) doğrudan açıklıyor.

**Boşluk:** ECE / reliability diyagramı ölçülmedi. Etiketli Cityscapes validasyon verisi
gerektiriyor; `evaluate.py run --fit-temperature` altyapısı hazır ve rol kapısı
sağlanabilir durumda, veri temin edilince tek komutla koşar.

---

## 5 · Açık küme / yol tehlikesi algılama

**İddia:** *"eğitim dağılımından farklı ... koşulları fark edebilmesi ve güvenilir olmayan
durumları işaretleyebilmesi"*

**Veri:** RoadAnomaly (Lis ve ark., EPFL CVLab), 60 kare, piksel etiketli gerçek yol
tehlikeleri — hayvan, kaya, koni, enkaz. Değerlendirme amaçlı; hiçbir model bu veriyle
eğitilmedi veya ince ayarlanmadı (`model_trained_on_this_data: false`).

**Protokol:** kare başına 20.000 piksel deterministik alt-örnekleme (`seed 20260728`),
toplam 1.200.000 piksel, %9,8 anomali.

### Skor karşılaştırması (PIDNet-S referans)

| skor | AUROC | AP | FPR95 |
|---|---|---|---|
| energy | **0,6670** | **0,1562** | 0,8608 |
| maximum logit | 0,6617 | 0,1539 | 0,8674 |
| normalize entropi | 0,6022 | 0,1322 | 0,9399 |
| MSP | 0,5693 | 0,1163 | 0,9462 |

`energy > max-logit > entropi > MSP` sıralaması OOD literatürünün bildirdiği sıralamayla
birebir aynı — uygulamayı bu sayılardan bağımsız olarak doğruluyor.

Bootstrap %95 GA (energy, 200 yeniden örnekleme): AUROC [0,6655, 0,6684],
AP [0,1550, 0,1574], FPR95 [0,8581, 0,8638].

### Tehlike türüne göre (energy)

| tehlike | AUROC | AP | anomali piksel |
|---|---|---|---|
| araç | 0,7070 | 0,1689 | 6.260 |
| koni | 0,6884 | 0,0678 | 4.552 |
| hayvan | 0,6717 | 0,2128 | 78.711 |
| engel | 0,6276 | 0,1189 | 28.190 |
| **kayıp yük** | **0,4805** | 0,0062 | 419 |

**Güvenlik bulgusu:** model kayıp yükte (lost cargo) rastgeleden kötü — AUROC 0,48. Yol
üzerine düşmüş yükü fark edemiyor.

### Operasyonel eşikler (energy)

| politika | eşik | TPR | FPR | F1 |
|---|---|---|---|---|
| F1-optimal | −6,233 | 0,450 | 0,210 | 0,267 |
| %5 risk bütçesi | −5,013 | 0,077 | 0,050 | 0,100 |

---

## 6 · Hava ve ışık koşullarına dayanıklılık

**İddia:** *"eğitim dağılımından farklı hava, ışık, yol ve trafik koşullarını fark
edebilmesi"*

### 6a · Gerçek olumsuz koşullar (ACDC, 406 kare, 5 mimari)

| model | temiz | sis | kar | yağmur | **gece** | gece kaybı | gece ECE |
|---|---|---|---|---|---|---|---|
| **SegFormer-B0** | 69,34 | 59,93 | 46,33 | 45,71 | **20,59** | **−70,3%** | 0,3150 |
| PIDNet-S | 67,68 | 57,14 | 42,79 | 39,64 | 14,97 | −77,9% | 0,2694 |
| BiSeNetV2 | 66,02 | 46,59 | 36,82 | 37,75 | 13,34 | −79,8% | 0,2712 |
| PIDNet-M | 68,47 | 60,29 | 41,53 | 42,64 | 12,54 | −81,7% | 0,3812 |
| **DDRNet-23-slim** | 68,50 | 56,99 | 38,75 | 44,59 | **7,12** | **−89,6%** | **0,4626** |

**Üç bulgu:**

1. **Gece bütün mimarilerde felaket.** En iyisi bile doğruluğunun %70'ini kaybediyor.
   Kalibrasyon her modelde **8,7–14,0 kat** bozuluyor (ECE ~0,03 → 0,27–0,46).

2. **Ama mimariler eşit çökmüyor.** SegFormer-B0 gecede doğruluğunun %29,7'sini
   koruyor; DDRNet-23-slim yalnızca %10,4'ünü — yani gecede **işlevsiz** (7,12 mIoU).

3. **Hız/enerji kazananı, dayanıklılık kaybedeni.** DDRNet-23-slim en hızlı (77,43 ms)
   ve en verimli (0,630 J/kare) model; aynı zamanda olumsuz koşullara en kırılgan olanı.
   Bu, üç eksenli seçime **dördüncü bir ekseni** ekliyor.

### 6b · Birleşik örüntü: tanıdık olmayan girdiye dayanıklılık

SegFormer-B0 iki bağımsız "tanıdık olmayan girdi" testinde de birinci:

| test | SegFormer-B0 | DDRNet-23-slim |
|---|---|---|
| bilinmeyen **nesne** (açık küme AP) | **0,3469** | 0,1651 |
| bilinmeyen **koşul** (gecede korunan doğruluk) | **%29,7** | %10,4 |

Beşlideki tek transformer bu. Dikkat tabanlı küresel bağlamın, girdi dağılımı kaydığında
CNN'lerin yerel özniteliklerinden daha zarif bozulduğu yorumu bu iki ölçümle tutarlı —
ancak n=5 ile bu bir gözlemdir, kanıtlanmış mekanizma değil.

### 6c · Sentetik bozulma (karşılaştırma amaçlı)

15 RoadAnomaly karesi, `rescue.stress._corrupt`, severity 0,7, PIDNet-S:

| koşul | ort. entropi | artış | düşük-güven piksel |
|---|---|---|---|
| temiz | 0,1709 | 1,00× | 3,55% |
| sis | 0,2009 | 1,18× | 6,76% |
| kar | 0,1906 | 1,12× | 6,01% |
| yağmur | 0,1736 | 1,02× | 4,33% |
| gece | 0,1703 | **1,00×** | 3,95% |

**Sentetik ile gerçek arasındaki uçurum:** sentetik "gece" (parlaklık düşürme) belirsizlik
sinyalinde **hiçbir tepki** yaratmıyor; gerçek ACDC gecesinde doğruluk %78 düşüyor. Yani
sentetik bozulma, gerçek koşul kaymasının yerine geçemez — bu, sentetik stres testlerine
dayanan çalışmalar için doğrudan bir uyarıdır.

## 7 · Uç cihaz performansı

**İddia:** *"gecikme, bellek tüketimi, güç kullanımı ve gerçek zamanlı çalışma kapasitesi"*

Beş model, aynı donanım, aynı güç modu, aynı kareler. Her biri 600 saniye sürdürülen yük,
200 kare ısınma, tam telemetri.

| model | motor | kare | FPS | ort. güç | J/kare | tepe RAM | logit çıktısı |
|---|---|---|---|---|---|---|---|
| PIDNet-M | 11,15 ms | 88,50 ms | 10,82 | 8,75 W | 0,808 J | 2.632 MiB | 64×128 |
| PIDNet-S | 5,07 ms | 80,77 ms | 11,87 | 7,89 W | 0,665 J | 2.601 MiB | 64×128 |
| **DDRNet-23-slim** | **3,81 ms** | **77,43 ms** | **12,36** | **7,79 W** | **0,630 J** | 2.541 MiB | 64×128 |
| SegFormer-B0 | 15,04 ms | **270,82 ms** | **3,57** | 7,93 W | **2,221 J** | 2.605 MiB | **128×256** |
| BiSeNetV2 | 13,45 ms | 92,36 ms | 10,34 | 8,36 W | 0,809 J | 2.530 MiB | 64×128 |

Hiçbirinde termal kısma yok (GPU 56–59 °C). Güç 25 W bütçesinin üçte biri.

**Gerçek zaman kapısı hiçbirinde geçmedi** (medyan ≤ 50 ms, ≥ 20 FPS). Ama sebebi
model değil — bkz. §8.

---

## 8 · Kare bütçesinin nereye gittiği

Aşama profili (PIDNet-S, 60 kare, gerçek cihaz):

| aşama | ilk ölçüm | optimizasyon sonrası | pay |
|---|---|---|---|
| derive_perception | 96,05 ms | 43,64 ms | 46,4% |
| preprocess | 24,80 ms | 24,71 ms | 26,3% |
| görüntü çözme | 13,97 ms | 14,00 ms | 14,9% |
| confidence_entropy | 5,78 ms | 5,80 ms | 6,2% |
| **motor** | **5,09 ms** | **5,09 ms** | **5,4%** |
| argmax | 0,75 ms | 0,74 ms | 0,8% |
| **toplam** | **146,44 ms** | **93,98 ms** | |

**Ana bulgu:** TensorRT motoru kare bütçesinin yalnızca **%5,4'ü**. Gerçek zamanlılık
hızlandırıcıda değil, CPU tarafındaki algı yığınında kazanılıp kaybediliyor.

### İki hedefli optimizasyon, modele dokunmadan

1. `_distance_from_mask` piksel-piksel Python BFS kuyruğuydu. Engelsiz 4-komşuluk
   ızgarasında BFS tam olarak L1 mesafe dönüşümüdür ve L1 ayrılabilirdir → dört
   `np.minimum.accumulate` taraması. **34× hızlanma** (9,20 → 0,27 ms), çıktı birebir aynı.
2. `_label_components` vektörleştirildi. **Karar hedef cihazda verildi:** geliştirme
   Mac'inde BFS 1,4× kazanıyordu, Jetson'da propagation 1,59× kazandı (3,154 → 1,989 ms;
   kare başına 11 çağrı, yani 34,7 → 21,9 ms). ARM CPU'da Python yorumlayıcısı numpy'a
   göre çok daha yavaş.

**Sonuç:** kare 151,67 → 80,77 ms (**1,88×**), enerji 1,084 → 0,665 J/kare (**1,63×**),
motor değişmedi, çıktılar birebir aynı — diferansiyel testlerle kanıtlı. Kuantalama yok,
doğruluk kaybı yok.

> *Uç cihaz optimizasyon kararları geliştirme makinesinde alınamaz: aynı iki
> implementasyon, aynı girdilerde, iki makinede zıt sonuç verdi.*

### SegFormer anomalisi

SegFormer-B0'ın motoru DDRNet'ten yalnızca **+11,23 ms** yavaş, ama karesi **+193,39 ms**
yavaş — motor farkının **17 katı**. Sebep `output_shape`: SegFormer stride-4'te
(128×256 = 32.768 piksel) logit üretiyor, diğerleri stride-8'de (64×128 = 8.192).
**4× daha fazla piksel** CPU tarafındaki her aşamaya giriyor.

> **Uç cihazda bir modelin sistem maliyetini belirleyen şey FLOP'ları veya GPU gecikmesi
> değil, segmentasyon başının çıktı stride'ıdır** — çünkü CPU tarafındaki algı yığını
> çıktı piksel sayısıyla ölçeklenir, hızlandırıcı ölçeklenmez.

**Bu maliyet giderilebilir bir entegrasyon artığı değil; ölçüldü.** Logitleri stride-8'e
indirmek post-processing'i **3,98×** hızlandırıyor ama **2,03 mIoU'ya mal oluyor**
(0,6721 → 0,6518, 60 Cityscapes val karesi, `scripts/measure_logit_stride_tradeoff.py`).
SegFormer-B0 bu durumda doğruluk sıralamasında birincilikten dördüncülüğe düşer.

Yani stride-4 çıktısı gerçek doğruluk taşıyor: yüksek çözünürlüklü logit hem daha iyi
segmentasyon hem daha yüksek CPU maliyeti demek. Uç cihaz için doğru soru "bu maliyet
kaldırılabilir mi" değil, **"bu doğruluk bu enerjiye değer mi"**.

---

## 9 · Üç eksenin birleşimi

| model | mIoU | OOD AUROC | OOD AP | J/kare | FPS |
|---|---|---|---|---|---|
| PIDNet-M | **80,22** | 0,562 | 0,124 | 0,808 | 10,82 |
| PIDNet-S | 78,74 | 0,667 | 0,156 | 0,665 | 11,87 |
| DDRNet-23-slim | 77,84 | 0,674 | 0,165 | **0,630** | **12,36** |
| SegFormer-B0 | 76,54 | **0,785** | **0,347** | 2,221 | 3,57 |
| BiSeNetV2 | 75,76 | 0,708 | 0,175 | 0,809 | 10,34 |

**Spearman ρ(mIoU, AUROC) = −0,90** · **ρ(mIoU, AP) = −0,90**

> **Doğruluk arttıkça açık küme güvenliği düşüyor.** En doğru model (PIDNet-M) yol
> tehlikelerini fark etmekte neredeyse yazı-tura seviyesinde (AUROC 0,562). En az doğru
> olanlardan SegFormer-B0 ise 2,8× daha iyi (AP 0,347 vs 0,124).

Mekanizma: SegFormer-B0 bu beşlideki tek transformer; öznitelik geometrisi bilinmeyen
nesneleri ayırmakta CNN dekoderlerinden farklı davranıyor.

**Karar:** DDRNet-23-slim pratik kazanan — en hızlı, en düşük enerjili, açık kümede ikinci
en iyi. SegFormer-B0 güvenlikte açık ara önde ama bu pipeline'da 3,53× enerjiye mal
oluyor; §8'deki stride düzeltmesi uygulanırsa bu maliyet büyük ölçüde kaybolur.

**Dürüstlük sınırı:** n = 5 mimari. ρ = −0,90 güçlü bir eğilim ama kesin kanıt değil.
Ayrıca bunlar farklı reçetelerle (farklı iterasyon/batch) eğitilmiş yayınlanmış
checkpoint'lerdir; kontrollü ablasyon değil, model karşılaştırmasıdır.

---

## 10 · Ölçülmeyenler

Hiçbiri "ölçülmüş" gibi sunulmayacak.

| eksik | neden | gereken |
|---|---|---|
| ECE / reliability diyagramı | etiketli validasyon verisi yok | Cityscapes val + `evaluate.py run --fit-temperature` |
| Referans modellerin sınıf bazlı IoU'su | aynı | Cityscapes val |
| Gerçek olumsuz-koşul verisi | ACDC indirilmedi | ACDC (kayıt gerektirir) |
| Sürülebilir alan sayısal metriği | `drivable_metrics` çağrısız | GT yol maskeleri |
| SegFormer stride düzeltmesi | zaman | ~20 dk Jetson ölçümü |
| Mühürlü final test verisi | **kasıtlı** | `docs/adr/0005` — yalnızca insan tetikler |

---

## 11 · Üretilen kod

| dosya | ne yapar |
|---|---|
| `scripts/evaluate_open_set.py` | piksel-OOD değerlendirmesi (§5) |
| `scripts/analyze_contextual_risk.py` | yedi özellikli risk füzyonu (§3) |
| `scripts/measure_shift_response.py` | hava/ışık tepkisi (§6) |
| `scripts/jetson/profile_pipeline.py` | aşama profili (§8) |
| `scripts/jetson/compare_labelling.py` | hedef cihazda implementasyon kıyası (§8) |
| `scripts/jetson/run_all_models.sh` | beş modelin gözetimsiz motor+benchmark koşusu (§7) |
| `scripts/build_presentation_outputs.py` | figür/tablo üretimi |

Yol boyunca bulunup düzeltilen gerçek hatalar: `Runner.from_cfg()` ağırlık yüklememesi ·
TF32/FP32 parite uyuşmazlığı · HPO'nun PRUNED denemeleri tamamlanmış sayması · 1980
öncesi mtime'ın log paketini sessizce düşürmesi · `np.trapezoid`'ın numpy 1.26'da
bulunmaması · `threshold_policies`'in O(n²) olması · Jetson benchmark'ının telemetriyi
10 dakikalık ölçümden *sonra* kontrol etmesi.
