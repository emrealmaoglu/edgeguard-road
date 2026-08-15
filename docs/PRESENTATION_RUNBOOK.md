# Sunum ve tez çıktıları — uygulama kılavuzu

**Durum (2026-08-15).** Sunuma 2 gün var. Kampanya zincirinin `hpo` sonrası kısmı
tekrar tekrar düşüyor ve bitmesi için gereken süre elde yok. Bu kılavuz, **bitmiş
`screening` koşusunun gerçek checkpoint'lerinden** sunum ve tez için gereken bütün somut
çıktıların nasıl üretileceğini anlatır.

Kampanyanın `hpo` / `final` / `selection` / `ablation` / `accept` / `evaluate` / `export` /
`report` / `package` zinciri bu yol için **gerekli değil**. Sunumun beş katkısının hiçbiri
o zincire bağlı değil ve buradaki hiçbir adım kabul kapısına dokunmuyor: `predict.py`,
`evaluate.py`, `audit_dataset.py`, `analyze_training_results.py`, `build_tensorrt.py`,
`benchmark.py` ve `app.py`'nin geliştirici modu kapısızdır. Kabul kapısı yalnızca
`thesis_bundle.py`, `colab_release.py` ve `generate_release_gallery.py` içindedir; bu
kılavuz o üçünü hiç çağırmaz.

## Elde hâlihazırda ölçülmüş olan

2026-08-14 screening koşusu üç modeli de eğitti, değerlendirdi ve ONNX'e çıkardı
(`scientific_evidence: true`, NVIDIA L4, bf16, 2.500 adım, ImageNet başlangıcı,
Cityscapes + IDD20K tekdüze örnekleme, `seed 20260728`).

**Ana sonuç tablosu** — kaynak: `reports/screening/candidate_table.json`
(`edgeguard-screening-logs (1).zip`), `rejected: []`, üçü de `onnx_validated: true`:

| model | domain-macro mIoU | Cityscapes | IDD20K | nadir sınıf mIoU | ONNX medyan gecikme (CPU) | ONNX boyut |
|---|---|---|---|---|---|---|
| **pidnet_s** | **0,3700** | 0,4641 | 0,2759 | **0,1755** | 43,1 ms | 30,6 MB |
| ddrnet_23_slim | 0,3574 | 0,4310 | 0,2840 | 0,1346 | 52,6 ms | 22,9 MB |
| segformer_b0 | 0,2286 | 0,2723 | 0,1849 | **0,0000** | 200,7 ms | 15,0 MB |

Bağımsız doğrulama — eğitim loglarındaki validasyon mIoU'ları
(`screening/<model>/ce/<zaman>/<zaman>.log`), aynı sıralamayı veriyor:

| model | val mIoU (iter 2.000) | val mIoU (iter 2.500) | eğitim süresi |
|---|---|---|---|
| pidnet_s | 34,16 | 35,48 | 41 dk |
| ddrnet_23_slim | 33,40 | 34,93 | 33 dk |
| segformer_b0 | 19,32 | 19,97 | 15 dk |

Sunumda söylenebilecek gerçek bulgular:

- **pidnet_s hem en doğru hem en hızlı** — segformer_b0'dan 4,7 kat düşük gecikmeyle
  1,6 kat yüksek mIoU. Gerçek zamanlı edge dağıtımı için transformer tabanlı modelin
  bu bütçede rekabet edemediği, ölçülmüş bir sonuçtur.
- **segformer_b0'ın nadir sınıf mIoU'su tam olarak 0,0000** — 2.500 adımlık bütçede
  nadir sınıfların hiçbirini öğrenememiş. Bu, sınıf dengesizliğinin mimariye göre farklı
  vurduğunun somut kanıtı ve rapordaki "nadir sınıf başarımı" başlığının doğrudan cevabı.
- **Her üç model de IDD20K'da Cityscapes'ten belirgin biçimde kötü** (pidnet_s'te 0,464 →
  0,276). Domain kaymasının ölçülmüş etkisi budur; sunumdaki "farklı koşullara genelleme"
  iddiasının sayısal karşılığı.

**Eski kampanyanın mIoU sayıları (16,48 / 20,53 / 26,54 / 25,21 / 17,41) kullanılmayacak.**
O değerlendirme rastgele ağırlık ölçüyordu: `Runner.from_cfg()` `cfg.load_from`'u
yüklemez, bu yalnızca `Runner.train()/val()/test()` içinde olur
(`mmengine/runner/runner.py:1765/1798/1821`). Hata bu oturumda bulunup
`load_evaluation_weights()` ile düzeltildi ve yukarıdaki tablo düzeltme sonrası ölçümdür.
O sayılar geçersizdir.

## 1 · Colab oturumu A — sunum çıktıları

`notebooks/EdgeGuard_10_Sunum_Ciktilari.ipynb` → **L4 GPU + Yüksek RAM** → *Tümünü çalıştır*.

Notebook sırayla: Drive'dan screening kaydını geri yükler (yeniden eğitim yok), sonra
`scripts/build_presentation_outputs.py`'yi çalıştırır ve `EdgeGuard_Sunum_Ciktilari.zip`
indirir.

Üretilenler:

| Çıktı | Nereden | Sunumdaki karşılığı |
|---|---|---|
| `overlay.png`, `mask.png` | `predict.py` | Semantik segmentasyon |
| `confidence.png`, `entropy.png` | `predict.py` | Piksel düzeyinde belirsizlik |
| `road_mask.png`, `drivable_corridor.png` | `predict.py --emit-regions` | Sürülebilir alan |
| `unreliable_mask.png` | `predict.py --emit-regions` | Güvenilmez bölge işaretleme |
| `attention_map.png`, `regions_overlay.png`, `regions/*.png` | `predict.py --emit-risk` | Operasyonel dikkat |
| `evaluation.json` → `reliability.before/after`, `temperature.json` | `evaluate.py run --fit-temperature` | ECE / kalibre edilmiş güven |
| `frame_uncertainty.json` | aynı koşu | Alan-başına belirsizlik karşılaştırması |
| `class_distribution_by_domain.*`, `pooled_imbalance_and_weights.*` | `audit_dataset.py` | Veri seti figürleri |
| eğitim eğrileri + sınıf tablosu | `analyze_training_results.py` | Eğitim davranışı |
| `candidate_table.json` | mevcut screening raporu | Model karşılaştırması |

Sıcaklık ölçekleme **meşru yoldan** yapılır: `evaluate_model` sıcaklık uydurmayı yalnızca
kaynak-domain `train_calibration` rolünde kabul eder
(`src/edgeguard/rescue/mmseg_runtime.py:1585`) ve her dondurulmuş eğitim manifesti bu rolü
zaten taşır (`ROLE_RATIOS`, `%5`). Kapı aşılmaz, sağlanır. İlk domainde sıcaklık uydurulur,
kalan domainler o sıcaklıkla ölçülür — böylece domainler arası karşılaştırma tek ölçekte
kalır ve "kalibrasyon domain kayması altında ne oluyor" sorusu ölçülmüş bir cevap alır.

Görsel kareler `train_calibration` diliminden seçilir, yani modellerin **fit edilmediği**
veriden; sample id'ye göre sıralı seçildiği için tekrar koşuda aynı kareler gelir.

### Adım kaydı — koşmayan hiçbir şey ölçülmüş görünmez

Sürücü her adımı tek tek çalıştırır ve tek tek kaydeder. Bir adım patlarsa **diğerleri yine
de üretilir** — kampanyanın baştan beri düşürdüğü şey buydu. `presentation_outputs.json`
her adımı `produced` / `reused` / `skipped` / `failed` olarak, atlananları gerekçesiyle
yazar; `scientific_status` ancak gerçekten üretilmiş bir adım varsa `measured` olur.
Notebook'un son hücresi başarısız ve atlanan adımları ayrıca listeler — **çıktıyı
kullanmadan önce oraya bakın.**

## 2 · Daha uzun pidnet_s eğitimi (opsiyonel)

**Ayrı bir notebook yok ve olmamalı.** Aynı `EdgeGuard_10_Sunum_Ciktilari.ipynb`
notebook'unun 4. hücresi bu işi yapar; varsayılan olarak kapalıdır.

Ayrı ikinci bir Colab oturumu açmak **bilerek reddedildi**: o oturumun da Drive'daki
kampanya durum deposunu geri yüklemesi gerekirdi ve iki oturumun aynı depoya eşzamanlı
yazması, sunumun tamamen üzerine kurulu olduğu bitmiş screening kaydını bozabilirdi.
İkinci oturumun kazandıracağı şey (belki 0,37 → 0,45 mIoU), o riski karşılamıyor.

Bunun yerine: **A oturumu bitip zip indikten sonra**, aynı oturumda 4. hücrede
`RUN_LONGER_TRAINING = True` yapıp hücreyi elle çalıştırın. Çalışma zamanı, veri ve
manifestler zaten hazır olduğu için ikinci bir geri yükleme yapılmaz.

Bu hücrenin çalıştırdığı komut `--recovery-root` **almaz**, yani Drive kurtarma deposuna
hiç yazmaz: uzun koşu, sunumun dayandığı kanıta zarar veremez. Kendi kaydını
`training_run.json` olarak yazar, `presentation_outputs.json`'ı ezmez.

`--max-steps` LR programının ufkunu da kısaltır: bu, 2.500'lük koşunun devamı değil,
kendi içinde tutarlı 10.000 adımlık ayrı bir koşudur. Bittiğinde `evaluate.py run` ile
ölçüp yeni sayıyı kullanabilirsiniz; bitmezse hiçbir şey kaybedilmez.

## 3 · Jetson Orin Nano Super — sunumun en güçlü kısmı

`build_tensorrt.py` düz `.onnx` alır; kabul edilmiş sürüm, deployment bundle veya release
ZIP istemez. Screening ONNX'i zaten `[1,3,512,1024]` — yeniden ihraç gerekmez.

| # | Adım | Kim |
|---|---|---|
| 0-2 | JetPack doğrula, güç modunu 25W'a sabitle (`nvpmodel`), NVIDIA'nın Jetson PyTorch wheel'ini kur (PyPI'dan **değil**) | **Sahip — sudo** |
| 3 | `pip install -e .` + `onnxruntime opencv-python-headless`; `torch.cuda.is_available()` ve `import tensorrt` doğrula | güvenli |
| 4 | `build_tensorrt.py` **dry-run** (bayraksız) → çalıştırılacak `trtexec` komutunu yazdırır | güvenli |
| 5 | Aynı komut `--execute` ile → FP16 engine + `engine_sha256` manifesti | Sahip |
| 6 | Benchmark karelerini NVMe'ye koy (disk I/O ölçüme dahildir) | güvenli |
| 7 | `tegrastats --interval 1000 --logfile …&` — benchmark'tan **hemen önce** | **Sahip — sudo** |
| 8 | `benchmark.py --power-profile 25W --warmup 200 --minimum-iterations 30000` | Sahip |
| 9 | `pkill tegrastats` — **hemen sonra** | **Sahip — sudo** |

`scripts/jetson/AGENTS.md` gereği otomatik ajanlar `sudo`, flashing, güç-modu değişimi ve
Jetson SSH çalıştırmaz; yukarıdaki sudo satırlarını sahip kendi elleriyle koşar.

### Koddan doğrulanmış üç tuzak

1. **Süre eşiği beklendiği gibi çalışmıyor.** `benchmark.py:200` döngüsü
   `while index < minimum_iterations and (elapsed < minimum_duration_seconds)` — hangisi
   önce dolarsa durur, ikisi birden değil. 20 FPS'te varsayılan 5.000 kare ≈ 250 sn, yani
   runbook'un vaat ettiği 600 sn'lik soak gerçekleşmez. Gerçek 10 dakika için
   `--minimum-iterations 30000` verin.
2. **Güç ortalaması bütün telemetri logunu kapsar.** `tegrastats`'ı dar pencerede tutun;
   boşta geçen süre watt ortalamasını aşağı çeker ve `joule_per_frame` olduğundan iyi görünür.
3. **`throttling_warning_detected` neredeyse boş bir kontrol** — stok `tegrastats`
   çıktısında o kelimeler geçmez. Termal kanıtı sıcaklık değerlerinden okuyun.

**Kabul kriteri (25W):** medyan ≤ 50 ms · p95 ≤ 66,7 ms · ≥ 20 FPS · telemetri tam.

## 4 · Sunum paneli — Jetson üzerinde

Kaydedilecek panel `presentation_app.py`'dir ve **anlattığı cihazın üzerinde** çalışır.
Canlı çıkarım yapmaz: gösterdiği her sayı diskteki bir ölçüm kaydından okunur. Sebep
pratik — kayıt sırasında bir modelin yüklenmesi ya da bir kareye takılması, anlatılan
şeyle ilgisi olmayan bir risktir.

*(Not: `app.py` ayrı bir geliştirici arayüzüdür ve sunumda kullanılmaz.)*

### 4a · Önce: Jetson kayıtlarını cihazdan geri al ⚠️

**Bu adım yapılmadan panelin 6. sayfası boş kalır** — sunumun en güçlü slaytı odur.
`run_all_models.sh` ölçümleri cihazda **tek düz klasöre** yazar ve o klasör bugüne kadar
hiç Mac'e kopyalanmadı; §7'deki gecikme/güç/joule tablosu şu an yalnızca bu dokümanda
duruyor, panelde değil.

Mac'te, tek komut:

```bash
scp -r emre@100.102.153.67:~/edgeguard-jetson-runs .local/presentation/jetson-run
```

*(Yol farklıysa: `run_all_models.sh`'e üçüncü argüman olarak verdiğin klasör hangisiyse
odur; cihazda `ls ~/*jetson*` ile bulunur.)*

Klasörde model başına şunlar olmalı: `<model>_benchmark.json`,
`<model>_tegrastats.log`, `<model>_stage_profile.json`. `.plan` motor dosyaları ve
`_build.json` kayıtları da orada olacak — paketleyici onları **almaz**, gerek yok.

### 4b · Paketi kur (Mac'te)

```bash
.venv/bin/python scripts/build_jetson_bundle.py \
  --results .local/presentation/results \
  --figures .local/presentation/figures \
  --video .local/presentation/video \
  --jetson-run .local/presentation/jetson-run \
  --output .local/presentation/bundle \
  --archive .local/presentation/eg_presentation_bundle.tgz \
  --require accuracy --require drivable --require open_set \
  --require jetson --require telemetry
```

`--jetson-run` düz klasörü panelin okuduğu üç gruba dağıtır: `*_benchmark.json` →
`jetson/`, `*_stage_profile.json` → `profile/`, `*_tegrastats.log` → `telemetry/`.

`--require`, sunumun onsuz verilemeyeceği sonuçlar içindir. Sebebi: panel eksik bir gruba
hata vermez, sadece o tabloyu göstermez — yani unutulan bir ölçüm sessizce kaybolur.
`--require` o sessizliği hataya çevirir. Paketin içindeki `bundle_manifest.json` neyin
bulunduğunu ve neyin bulunamadığını ayrı ayrı yazar; kopyalamadan önce ona bakın.

### 4c · Cihaza kopyala

```bash
scp .local/presentation/eg_presentation_bundle.tgz emre@100.102.153.67:~/
```

Jetson'da:

```bash
tar xzf ~/eg_presentation_bundle.tgz -C ~/ && ls ~/bundle
```

### 4d · Paneli çalıştır

```bash
EDGEGUARD_RESULTS=~/bundle streamlit run ~/edgeguard-road/presentation_app.py
```

Kenar çubuğundaki anlık sıcaklık/güç/RAM cihazın **o andaki** durumudur; asıl telemetri
sayfa 6'daki, sonuçlar üretilirken kaydedilmiş zaman serileridir. İkisi ayrı şeydir ve
panel bunu kendi üstünde yazar.

Kayıt sırası, sayfa numaralarıyla aynı: **1 Problem ve yöntem** → **2 Model
karşılaştırması** → **3 Açık küme** → **4 Belirsizlik ve kalibrasyon** → **5 Bağlamsal
risk** → **6 Uç cihaz telemetrisi** → **7 Sınırlar**. Toplam 2–3 dakika; kalan 2–3 dakika
konuşma.

## Sunumda söylenmeyecekler

- **Piksel düzeyinde OOD sayısı (AUPR/FPR95) yok.** `evaluation/ood.py::pixel_ood_metrics`
  tam ve testlidir ama tek besleyeni sentetik rastgele logittir
  (`campaign/stages.py:358`, `run_local_closure.py:233`). Gerçek ölçüm için piksel-etiketli
  anomali verisi (Fishyscapes Lost&Found) gerekir; `data/fishyscapes.py` adaptörü vardır
  ama hiç çağrılmaz ve veri indirilmemiştir.

  Bu **final** sunum olduğu için bunu "sıradaki adım" diye geçiştirmek yanlış olur —
  sahiplenilmiş bir kapsam sınırı olarak, gerekçesiyle söylenir: *"Açık küme tespitini
  belirsizlik tabanlı olarak uyguladım ve niteliksel olarak gösteriyorum; sayısal
  AUPR/FPR95 için piksel düzeyinde anomali etiketli bir veri seti (Fishyscapes
  Lost&Found) gerekiyor, bu da altyapı ve hesaplama bütçesi nedeniyle kapsam dışında
  bırakıldı."* Bitirme projesinde gerekçelendirilmiş kapsam daraltması normaldir; uydurma
  sayı vermek değildir. Gösterilen şey gerçek: entropi/energy haritaları ve
  `unreliable_mask` ile güvenilmez bölge işaretleme çalışıyor.
- **`--emit-risk` fiziksel risk olasılığı üretmez.** `context/risk.py::contextual_risk`
  çağrılmaz; üretilen şey deterministik bir *dikkat* skorudur ve `summary.json` zaten
  `"physical_risk_probability": false` der. Slaytta "operasyonel dikkat göstergesi" denir.
- **`fast_scnn` ve `bisenetv2` screening kapsamı dışındaydı** — sunumda "hesaplama bütçesi
  nedeniyle 5 aday 3'e indirildi" diye açıkça yazılır.
- **Mühürlü final test verisi açılmaz** (`docs/adr/0005`). Bu kapı yalnızca insan
  tarafından tetiklenir ve bu kılavuzdaki hiçbir adım ona dokunmaz.

## Doğrulama

```bash
.venv/bin/python -m pytest tests/unit/test_presentation_outputs.py tests/unit/test_delivery_notebooks.py -q
```

Koşu sonrası elle bakılacaklar:

- `presentation_outputs.json` içinde `failed` / `skipped` adım var mı, varsa gerekçesi ne?
- Her `predict.py` çıktısında `summary.json` var ve `regions[]` dolu mu?
- `evaluation.json` → `reliability.after.ece` sayısal mı (null değil)? Null ise
  `--fit-temperature` rol kapısına takılmıştır.
- Jetson `benchmark.py` çıktısında `telemetry.telemetry_complete: true` ve
  `joule_per_frame` null **değil** mi? Null ise `tegrastats` penceresi kaçmıştır.
