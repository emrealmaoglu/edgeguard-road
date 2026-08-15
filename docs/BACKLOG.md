# Yapılacaklar · öncelikli

Kalan süre: sunum videosu **~21 saat** içinde Drive'a yüklenmiş olmalı. Tez sonrasında.
Sıralama buna göre: videoya giren her şey önce.

Durum kodları: ✅ bitti · 🔵 sırada · ⚪ tez için, video sonrası · ❌ kapsam dışı

---

## A · Video için kritik yol

| # | iş | durum | kim | süre |
|---|---|---|---|---|
| A1 | Panel Jetson'da çalışıyor | ✅ | — | — |
| A2 | 9 sonuç figürü + 4 şema | ✅ | — | — |
| A3 | Demo videosu (180 kare) | ✅ | — | — |
| A4 | GPU ön işleme ölçümü + argmax uyumu | 🔵 | **sen** | 2 dk |
| A5 | Panelin dil/tipografi geçişi | 🔵 | ben | 1 sa |
| A6 | Son paket → Jetson | 🔵 | ikimiz | 15 dk |
| A7 | Prova: 7 sayfa, 2-3 dk | 🔵 | sen | 30 dk |
| A8 | **Kayıt + Drive'a yükleme** | 🔵 | sen | 1 sa |

## B · Optimizasyonlar

| # | iş | beklenen | durum |
|---|---|---|---|
| B1 | Mesafe dönüşümü ayrılabilir | 34× (9,20→0,27 ms) | ✅ |
| B2 | Bileşen etiketleme vektörleştirme | Jetson'da 1,59× | ✅ |
| B3 | Tek geçiş çoklu sınıf etiketleme | 3,90× (15,45→3,96 ms) | ✅ |
| B4 | GPU letterbox + normalizasyon | 24,7 → ~5 ms bekleniyor | 🔵 ölçüm bekliyor |
| B5 | Pinned memory (H2D asenkron) | ~3-5 ms | ⚪ |
| B6 | **SegFormer stride düşürme** | 270,8 → ~85 ms (3,2×) | 🔵 en yüksek değer |
| B7 | softmax/energy paylaşımı | ~3 ms | ⚪ |
| B8 | D2H'yi uint8 maskeye indirme | araştırma öneriyor (29) | ⚪ |
| B9 | INT8 kuantalama | motor 5,1 → ~3 ms | ❌ doğruluk riski, motor zaten %6,6 |
| B10 | NVDEC donanım decode | 14 ms | ❌ dosya tabanlı benchmark, dağıtımda yok |

**Kümülatif (ölçülmüş):** kare 146,44 → 76,39 ms (**1,92×**), enerji 1,084 → 0,665 J/kare,
motora hiç dokunulmadan, çıktılar birebir aynı.

## C · Tez — ölçüm gerektirmeyen yazı işleri

| # | iş | rubrik puanı | durum |
|---|---|---|---|
| C1 | PRISMA akış şeması (Şekil 1.1) | 5 | 🔵 |
| C2 | ~50 kaynağın doğrulanması + ISO 690 listesi | 5 | 🔵 14 doğrulandı, bekleyen yok |
| C3 | Konu dışı 34 kaydın elenmesi | — | 🔵 |
| C4 | Standartlara atıf (ISO/IEC 25010, IEEE 829, ISO 690) | 5 | ⚪ |
| C5 | Girişimcilik/yenilikçilik bölümü | 4 | ⚪ |
| C6 | Sürdürülebilirlik bölümü (joule/kare üzerinden) | 4 | ⚪ |
| C7 | ÖZET + ABSTRACT | — | ⚪ |
| C8 | Beyan + AI kullanım beyanı | — | ⚪ sen imzalayacaksın |

## D · Tez — ek ölçümler (varsa zaman)

| # | iş | değer | durum |
|---|---|---|---|
| D1 | FP16 dağıtım sadakati (Jetson'da mIoU) | `numerical_equivalence_pending` kapanır | ⚪ script hazır |
| D2 | 5 modelin ACDC ölçümü (şu an sadece PIDNet-S) | koşul × mimari etkileşimi | ⚪ ~40 dk |
| D3 | Zamansal kalıcılık (riskin 7. özelliği) | 150 kare: izlerin %50,1'i tek karelik | ✅ |
| D4 | Sürülebilir alan sayısal metriği | Cityscapes val, 200 kare × 5 mimari | ✅ |
| D5 | Kendi modellerimizin ACDC/RoadAnomaly ölçümü | referansla karşılaştırma | ⚪ |
| D6 | Veri seti dağılım figürleri | rubrik: FAIR | ⚪ |

## E · Araştırma dokümanlarından çıkan, henüz değerlendirilmemiş fikirler

`researchs/` 36 dokümanın taranmasından. Hiçbiri video için kritik değil.

| kaynak | fikir | değerlendirme |
|---|---|---|
| 05 (kalibrasyon) | Local Temperature Scaling, conformal segmentation | ⚪ ECE'yi düşürebilir; sıcaklık ölçekleme altyapısı hazır |
| 06 (bağlamsal risk) | 7 özellikli füzyon | ✅ uygulandı |
| 07 (zamansal tutarlılık) | N-of-M persistence, hysteresis | ⚪ D3 ile birlikte |
| 04 (OOD) | energy/max-logit/MSP karşılaştırması | ✅ uygulandı, literatür sıralamasıyla örtüştü |
| 19 (cross-dataset) | görülmemiş veri setinde test | ✅ ACDC + RoadAnomaly ile karşılandı |
| 25 (failure analysis) | hata modu sınıflandırması | ⚪ gece bulgusu için yapılabilir |
| 26 (metrik/istatistik) | bootstrap güven aralığı | ✅ uygulandı |
| 29 (Jetson) | pinned memory, D2H küçültme | ⚪ B5, B8 |
| 36 (tez yazımı) | yapı önerileri | ⚪ şablonla karşılaştırılacak |

## F · Kapalı riskler

- Panel tek bozuk dosyada çökmüyor ✅
- Benchmark telemetri eksikse 10 dakika harcamadan duruyor ✅
- Kayıtlarda `scientific_status` yalnızca gerçekten üretilmiş adım için `measured` ✅
- Mühürlü test verisi açılmadı, açılmayacak ✅
- Araştırma dokümanlarındaki sayılar doğrudan alıntılanmayacak ✅ (`RESEARCH_VERIFICATION.md`)

---

## Şimdi ne yapılıyor

**Sen:** A4 (GPU ön işleme ölçümü)
**Ben:** B6 (SegFormer stride) → A5 (panel tipografi) → C1/C2 (PRISMA + kaynak doğrulama)
