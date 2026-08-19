# ÖZET ve ABSTRACT

> **Şablon notu:** Özet **tek paragraf** olmalıdır. Aşağıdaki metin bu kurala göre
> yazılmıştır; okunabilirlik için burada cümleler ayrı satırlarda görünse de Word'e tek
> paragraf olarak yapıştırılmalıdır.
>
> Şablonun özet sayfası ayrıca şu satırları ister: yazar adı, kurum, "Bitirme Tezi",
> danışman adı, ay-yıl ve **sayfa sayısı**. Sayfa sayısı rapor tamamlandıktan sonra
> yazılacaktır.

---

## ÖZET

**KAYNAK KISITLI UÇ CİHAZLARDA BELİRSİZLİK FARKINDALIKLI AÇIK KÜME YOL TEHLİKESİ ALGILAMA
VE BAĞLAMSAL RİSK ANALİZİ**

Otonom sürüş algı sistemleri, eğitim sırasında görmedikleri nesnelerle karşılaştıklarında
sessizce yanılır: bir tahmin üretir ve o tahminden yüksek güven bildirirler. Bu tez, kaynak
kısıtlı bir uç cihazda çalışan, belirsizliğinin farkında ve açık kümeye dayanıklı bir yol
tehlikesi algılama hattı tasarlamakta ve bu hattı doğruluk, belirsizlik, açık küme ve uç
cihaz maliyeti eksenlerinde ölçmektedir. Beş gerçek zamanlı semantik bölütleme mimarisi
(PIDNet-M, PIDNet-S, DDRNet-23-slim, SegFormer-B0, BiSeNetV2) ortak bir protokolle, dağıtım
çözünürlüğünde ve aynı kareler üzerinde karşılaştırılmış; sistem NVIDIA Jetson Orin Nano
Super üzerinde 25 W güç modunda, TensorRT FP16 ile ve 600 saniyelik sürdürülen yük altında
profillenmiştir. Belirsizlik dört skorla (maksimum softmax olasılığı, entropi, maksimum
logit, enerji) ölçülmüş, açık küme başarımı 60 kare piksel etiketli gerçek yol tehlikesi
üzerinde değerlendirilmiştir. Dört bulgu elde edilmiştir: birincisi, yayımlanmış model
sıralaması dağıtım koşuluna taşınmamaktadır (Spearman ρ = +0,10) ve uç cihaz için model
seçimi yayımlanmış doğruluğa bakılarak yapılamaz; ikincisi, sistem maliyetini model boyutu
değil bölütleme başının çıktı stride'ı belirlemektedir — bir modelin hızlandırıcı gecikmesi
11,23 ms fazlayken uçtan uca kare süresi 193,39 ms fazladır, çünkü daha ince çıktı ızgarası
CPU tarafına dört kat piksel vermektedir; üçüncüsü, doğrulukta ayırt edilebilir bir kazanan
yoktur — aynı 500 karede eşleştirilmiş karşılaştırma ilk iki modeli ayırt edememektedir
(+0,0000, %95 aralık [−0,0050, +0,0049]) ve seçim enerji ile dayanıklılığa kalmaktadır;
dördüncüsü, havuzlanmış kalibrasyon hatası yanıltıcıdır ve sınıf bazlı hesaplandığında
1,5 ile 3 kat büyümektedir. Ayrıca sistemin yola düşmüş yükü ayırt edemediği (AUROC 0,4805,
rastgeleden kötü) ve gece koşulunda bütün mimarilerin doğruluğunun en az %70'ini
kaybederken kalibrasyon hatasının 8,7 ile 14,0 kat arttığı ölçülmüştür. Sistem 12,36 FPS
ile gerçek zaman hedefini karşılamamaktadır; darboğazın model değil CPU tarafındaki algı
yığını olduğu ölçülmüş ve kod tarafı optimizasyonuyla kare süresi modele dokunulmadan
151,67 ms'den 80,77 ms'ye indirilmiştir.

**Anahtar sözcükler:** Semantik bölütleme, Açık küme tanıma, Belirsizlik kestirimi,
Kalibrasyon, Uç bilişim, Jetson, TensorRT, Otonom sürüş.

---

## ABSTRACT

**UNCERTAINTY-AWARE OPEN-SET ROAD HAZARD DETECTION AND CONTEXTUAL RISK ANALYSIS ON
RESOURCE-CONSTRAINED EDGE DEVICES**

Autonomous driving perception systems fail silently when they encounter objects absent from
their training distribution: they produce a prediction and report high confidence in it.
This thesis designs an uncertainty-aware, open-set road hazard detection pipeline running
on a resource-constrained edge device and measures it along four axes: accuracy,
uncertainty, open-set performance and edge cost. Five real-time semantic segmentation
architectures (PIDNet-M, PIDNet-S, DDRNet-23-slim, SegFormer-B0, BiSeNetV2) are compared
under a common protocol, at deployment resolution and on identical frames; the system is
profiled on an NVIDIA Jetson Orin Nano Super in its 25 W power mode with TensorRT FP16
under a 600-second sustained load. Uncertainty is quantified with four scores (maximum
softmax probability, entropy, maximum logit, energy) and open-set performance is evaluated
on 60 pixel-labelled frames of real road hazards. Four findings emerge: first, published
model rankings do not transfer to the deployment condition (Spearman ρ = +0.10), so edge
model selection cannot be made from published accuracy; second, system cost is determined
by the segmentation head's output stride rather than model size — one model's accelerator
latency is 11.23 ms higher while its end-to-end frame time is 193.39 ms higher, because a
finer output grid feeds four times as many pixels to the CPU-side stack; third, there is no
distinguishable accuracy winner — a paired comparison on the same 500 frames cannot
separate the top two models (+0.0000, 95% interval [−0.0050, +0.0049]), leaving the choice
to energy and robustness; fourth, pooled calibration error is misleading and grows by a
factor of 1.5 to 3 when computed per class. The system is further shown to be unable to
distinguish lost cargo on the road (AUROC 0.4805, worse than chance), and under night
conditions every architecture loses at least 70% of its accuracy while calibration error
grows by a factor of 8.7 to 14.0. The system reaches 12.36 FPS and does not meet the
real-time target; the bottleneck was measured to be the CPU-side perception stack rather
than the model, and code-side optimisation reduced frame time from 151.67 ms to 80.77 ms
without touching the network.

**Keywords:** Semantic segmentation, Open-set recognition, Uncertainty estimation,
Calibration, Edge computing, Jetson, TensorRT, Autonomous driving.
