# EKLER

> **Şablon notu:** Ekler şablonda `EK 1: BAŞLIK ADI` biçiminde verilmektedir ve ek
> şekilleri `Şekil 10.1` diye numaralanır (şablon ekleri 10. bölüm saymaktadır).

---

## EK 1: ÖLÇÜM KAYITLARININ ŞEMASI

Bu raporda geçen bütün sayılar, makine tarafından okunabilir ölçüm kayıtlarından
gelmektedir. Kayıtlar `reports/measurements/` dizininde bulunmaktadır.

Çizelge 10.1. Ölçüm kayıtları ve üretildikleri betikler.

| Kayıt | İçerik | Üreten |
|---|---|---|
| `drivable/*.json` | 5 mimari × 200 kare: yol IoU, iki toleransta sınır F1, yanlış-sürülebilir oran | `evaluate_drivable_area.py` |
| `calibration/*.json` | 3 mimari × 200 kare: havuzlanmış ve sınıf bazlı ECE, sınıf kırılımı | `measure_classwise_calibration.py` |
| `components/*.json` | 3 mimari × 100 kare: bileşen kapsama, bileşen IoU, parçalanma | `evaluate_component_localization.py` |
| `temporal/*.json` | 150 ardışık kare: iz ömürleri, titreme, risk sıralamasına etki | `measure_temporal_persistence.py` |
| `paired_comparison.json` | 5 mimari × aynı 500 kare: kare başına mIoU, eşleştirilmiş bootstrap | `compare_models_paired.py` |
| `class_distribution.json` | 500 kare, 917 M piksel: sınıf dağılımı ve başarım korelasyonları | `measure_class_distribution.py` |
| `leakage_audit.json` | 4 küme × 446 kare: yarıçap taramalı yakın-kopya denetimi | `audit_split_leakage.py` |
| `literature_audit.json` | 36 tarama dokümanı: PRISMA aşama sayıları | `audit_literature_corpus.py` |
| `*_fp16_accuracy.json` | Cihazda 3 mimari × 100 kare: FP16 ve FP32 eşleştirilmiş | `jetson/evaluate_engine.py` |

### Kayıt Şeması

Her kayıt aşağıdaki alanları taşımaktadır:

| Alan | Anlamı |
|---|---|
| `schema_version` | Kayıt biçiminin sürümü |
| `record_type` | Ölçümün türü |
| `generated_at` | UTC üretim zamanı |
| `model_sha256` | Ölçülen ağırlık dosyasının özeti |
| `frames` | Kullanılan kare sayısı |
| `seed` | Rastgelelik tohumu (varsa) |
| `scientific_measurement` | Kaydın gerçek bir ölçümden geldiğini bildirir |

`model_sha256` alanı, bir sayının hangi ağırlıkla üretildiğinin kayıttan doğrulanmasını
sağlamaktadır.

### Sayısal Tutarlılık Denetimi

Raporda geçen dört ondalıklı her sayının bir ölçüm kaydında karşılığı bulunup bulunmadığı
otomatik olarak denetlenmektedir (`scripts/verify_reported_numbers.py`). Denetim, elle
yazılmış veya güncelliğini yitirmiş bir sayıyı test aşamasında yakalamaktadır. Kayıtlarda
doğrudan bulunmayan türetilmiş değerler (iki kaydın farkı gibi) ayrı bir listede,
türetiliş biçimiyle birlikte belirtilmektedir.

---

## EK 2: ÜRETKEN YAPAY ZEKÂ KULLANIM KAYDI

Şablonun üretken yapay zekâ kullanım beyanına dayanak oluşturmak üzere, proje boyunca
yapılan her materyal değişiklik **ekle-sadece** bir kayıtta tutulmuştur
(`docs/AI_USAGE_LOG.md`).

Kayıt her satırda şunları içermektedir: tarih, değişikliği yapan, değişikliğin konusu,
dokunulan dosyalar, yapılan doğrulama (test sonucu dâhil), yetkinin dayanağı, sonucu ve
gerekçesi.

Kaydın iki özelliği belirtilmelidir:

- **Geriye dönük düzenlenmez.** Bir karar değiştiğinde eski satır silinmez; düzeltme yeni
  bir satır olarak eklenir. Bu raporda geri çekilen iki sonuç, kendi düzeltmeleriyle
  birlikte bu kayıtta bulunmaktadır.
- **Doğrulama içerir.** Her satır, o değişikliğin hangi testle doğrulandığını yazmaktadır.

---

## EK 3: YENİDEN ÜRETİM TALİMATLARI

### Ölçümlerin Yeniden Üretilmesi

Doğruluk, kalibrasyon, sürülebilir alan, bileşen ve open-set ölçümleri host makinede
yeniden üretilebilir. Gereken girdiler: Cityscapes doğrulama kümesi, ACDC doğrulama kümesi,
RoadAnomaly ve karşılaştırılan mimarilerin ONNX grafikleri.

Bütün betikler ölçüm kaydını `--output` ile belirtilen yola yazmakta ve rastgelelik
tohumunu kayda dâhil etmektedir. Aynı girdi ve aynı tohumla çalıştırıldığında aynı kayıt
üretilir.

### Uç Cihaz Ölçümlerinin Yeniden Üretilmesi

Uç cihaz ölçümleri Jetson üzerinde yapılmalıdır. Sıra şöyledir: güç modunun 25 W'a
sabitlenmesi, ONNX grafiğinden TensorRT motorunun kurulması, telemetri kaydının
başlatılması, sürdürülen yük ölçümü ve telemetrinin durdurulması.

Motor kurulumu, üretilen motorun sha256 özetini içeren bir manifest yazmaktadır; ölçüm
kayıtları bu özeti taşır, dolayısıyla hangi motorun ölçüldüğü doğrulanabilir.

### Sürüm Sabitleme

| Bileşen | Sürüm |
|---|---|
| TensorRT | 10.3.0 |
| JetPack | 6 (L4T 36, aarch64) |
| Güç modu | 25 W |
| Rastgelelik tohumu | 20260728 |
| Dağıtım çözünürlüğü | 512×1024 |

### Doğrulama

Depo, 790 birim testi içermektedir. Testlerin bir bölümü doğrudan karşılaşılmış hatalara
karşılık gelmekte ve hatanın koşulunu sabitlemektedir. Ayrıca `scripts/` altındaki 69
betiğin tamamı, belgelenen çağrı biçimiyle çalıştırılarak içe aktarma bütünlüğü açısından
sınanmaktadır.
