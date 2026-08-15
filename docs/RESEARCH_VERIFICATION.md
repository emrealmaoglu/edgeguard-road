# Araştırma dokümanlarının ölçümle doğrulanması

`researchs/` klasöründeki 36 doküman üretken yapay zekâ ile hazırlandı. Kaynakların
gerçek olduğu ayrıca denetlendi (`docs/LITERATURE_PLAN.md`); bu dosya farklı bir soruyu
sorar: **dokümanlardaki teknik iddialar doğru mu?**

Elimizde bunu yapacak şey var — gerçek donanımda alınmış ölçümler. Uyuşan iddialar hem
araştırmayı hem ölçümü güçlendirir; uyuşmayanlar tezde tartışılacak bulgudur.

Ölçüm kaynağı: Jetson Orin Nano Super · 25 W · TensorRT 10.3.0 FP16 · 600 sn sürdürülen
yük, ve Cityscapes val (500 kare) üzerinde ONNX FP32.

---

## 1 · Doğrulanan iddialar

| iddia (kaynak) | ölçümümüz | sonuç |
|---|---|---|
| *"67 TOPS değerinin doğrudan FPS'ye lineer yansıyacağını varsaymak akademik bir yanılgıdır"* (08) | Motor 5,08 ms ama kare 76,39 ms — TOPS ile FPS arasında hiçbir doğrusal ilişki yok | ✅ **kuvvetle doğrulandı** |
| *"Ön işleme CPU üzerinde yapılmamalıdır"* (08) | `preprocess` 24,72 ms = karenin %32,4'ü | ✅ doğrulandı |
| *"Ortalama gecikme yerine yüzdelik raporlanmalı"* (08) | p99 204,3 ms iken medyan 80,8 ms — 2,5× fark | ✅ doğrulandı |
| *"Warm-up zorunludur"* (08) | 200 kare ısınma uygulandı | ✅ uyumlu |
| *"Sürekli 25 W'ta soğutma aşılırsa saat hızları düşer"* (03) | 7,79–8,75 W ölçüldü, 57–59 °C, **kısma yok** | ✅ mekanizma doğru, bizim yükte tetiklenmedi |
| *"8 GB LPDDR5, 102 GB/s"* (03, 27, 29) | Telemetri `ram_total_mib: 7620` (≈7,44 GiB kullanılabilir) | ✅ tutarlı |
| *"INT8, kalibrasyon veri seti gerektirir"* (08) | Uygulanmadı; motor zaten karenin %6,6'sı | ✅ doğru, ama önceliksiz |

## 2 · Düzeltilen veya eksik iddialar

| iddia | ölçümümüz | değerlendirme |
|---|---|---|
| *"BiSeNetV2 … %71,9 mIoU"* (03) | Kullandığımız resmî checkpoint **75,76** (mmseg model zoo) | ⚠ Makalenin bildirdiği değer ile model zoo checkpoint'i farklı. Tezde **hangi checkpoint** olduğu belirtilmeli. |
| *"PIDNet-S %78,6 mIoU Cityscapes **test** setinde"* (03) | Model zoo **val** için 78,74 bildiriyor | ⚠ test/val karışıklığı. Bizim ölçümümüz val'de. |
| *"DDRNet … %77,4 mIoU"* (03) | Model zoo 77,84 | ⚠ küçük sapma, muhtemelen farklı sürüm |
| *"p95 gecikme 100 ms'yi aşıyorsa mimari hantaldır"* (27) | PIDNet-S p95 **97,80 ms** (v2) | ⚠ **eşiğin hemen altındayız** — bu kriter tezde tartışılmalı |
| *"Hedef bütçe 30 FPS için kare başına 33 ms"* (29) | 76,39 ms → 13,1 FPS | ❌ **hedefe ulaşılamadı**; sebep atfedildi (CPU tarafı) |
| *"GStreamer/NVDEC ile donanım decode"* (29) | Dosya tabanlı benchmark; decode 14,00 ms CPU'da | ❌ uygulanmadı, kapsam dışı bırakıldı |

## 3 · Araştırmanın öngörmediği, ölçümün bulduğu

Bu üçü hiçbir dokümanda geçmiyor ve tezin özgün katkısını oluşturuyor:

1. **Çıktı stride'ı sistem maliyetini belirliyor.** SegFormer-B0'ın motoru DDRNet'ten
   +11,2 ms yavaş ama karesi **+193,4 ms** yavaş — çünkü stride-4 çıktısı (128×256) CPU
   tarafına 4× piksel veriyor. Araştırma dokümanları mimari seçimini FLOP ve motor
   gecikmesi üzerinden tartışıyor; bu maliyet kalemi hiç anılmıyor.

2. **Yayınlanmış mIoU sıralaması dağıtımda korunmuyor** (ρ = +0,10). Dokümanlar model
   seçimini yayın değerlerine göre öneriyor; ölçüm bunun dağıtım çözünürlüğünde
   geçersiz olduğunu gösteriyor.

3. **Optimizasyon kararları geliştirme makinesinde alınamıyor.** Aynı iki bileşen
   etiketleme implementasyonu Mac'te BFS lehine 1,4×, Jetson'da vektörleştirme lehine
   1,59× sonuç verdi. Hiçbir doküman bu tuzağı anmıyor.

## 4 · Sonuç

Dokümanların **yöntem çerçevesi güvenilir** — ölçüm metodolojisi (yüzdelik raporlama,
ısınma, sürdürülen test, güç örnekleme) bizim bağımsız olarak uyguladığımızla örtüşüyor.

**Sayısal iddialar ise dikkatle kullanılmalı.** Model mIoU değerleri makale/checkpoint/
split ayrımı yapılmadan verilmiş; tezde her sayının **hangi checkpoint, hangi split,
hangi çözünürlük** olduğu açıkça yazılacak. Bizim tablolarımız bunu zaten yapıyor.

> Tezde kullanılacak kural: araştırma dokümanları **yöntem ve literatür haritası** olarak
> kullanılır; **hiçbir sayı doğrudan alıntılanmaz.** Her sayı ya bizim ölçümümüzdür ya da
> birincil kaynağından doğrulanmıştır.
