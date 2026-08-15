# Doğrulanmış kaynakça (ISO 690)

Aşağıdaki her kaynak **açılıp doğrulandı**: başlık, yazarlar, yayın yeri ve yıl birincil
kaynağından teyit edildi. `researchs/` klasöründen gelen ama doğrulanmamış hiçbir kayıt
bu listede yok.

Şablon ISO 690 istiyor: *YAZAR, Ad. Başlık. Yayın yeri, yıl, sayfa. DOI/URL.*

---

## A · Kullanılan mimariler (birincil kaynaklar)

**[1]** XU, Jiacong, XIONG, Zixiang ve BHATTACHARYYA, Shankar P. *PIDNet: A Real-Time
Semantic Segmentation Network Inspired by PID Controllers.* Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html
✔ doğrulandı · üç dallı mimari, sınır dikkatiyle detay/bağlam füzyonu

**[2]** XIE, Enze, WANG, Wenhai, YU, Zhiding, ANANDKUMAR, Anima, ALVAREZ, Jose M. ve
LUO, Ping. *SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers.* Advances in Neural Information Processing Systems 34 (NeurIPS), 2021.
https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
✔ doğrulandı · hiyerarşik transformer kodlayıcı + hafif MLP kod çözücü, konum kodlaması yok

> **Not:** SegFormer'ın konum kodlaması kullanmaması, bizim ölçtüğümüz çözünürlük
> dayanıklılığıyla (dağıtım çözünürlüğünde en düşük kayıp, −%9,4) doğrudan tutarlı.
> Makale bunu Cityscapes-C üzerinde "zero-shot robustness" olarak bildiriyor; bizim
> ACDC gece ölçümümüz (en yüksek dayanıklılık, %29,7) bağımsız bir doğrulaması.

**[3]** HONG, Yuanduo, PAN, Huihui, SUN, Weichao ve JIA, Yisong. *Deep Dual-resolution
Networks for Real-time and Accurate Semantic Segmentation of Road Scenes.* arXiv preprint
arXiv:2101.06085, 2021. https://arxiv.org/abs/2101.06085
✔ doğrulandı · iki dallı omurga + Deep Aggregation Pyramid Pooling Module (DAPPM)

> **Sayı ayrımı — tezde önemli:** makale DDRNet-23-slim için **Cityscapes test setinde
> %77,4 mIoU / 102 FPS (2080Ti)** bildiriyor. mmsegmentation model zoo checkpoint'i ise
> **val setinde 77,84** veriyor. İkisi de doğru, farklı split ölçüyorlar. Bizim
> ölçümümüz (68,50) **dağıtım çözünürlüğünde val**'dir — üçü karıştırılmamalıdır.

**[4]** YU, Changqian, GAO, Changxin, WANG, Jingbo, YU, Gang, SHEN, Chunhua ve SANG, Nong.
*BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic
Segmentation.* International Journal of Computer Vision, 2021, cilt 129, s. 3051–3068.
https://doi.org/10.1007/s11263-021-01515-2
✔ doğrulandı · Detail Branch + Semantic Branch + Guided Aggregation Layer

## B · Veri setleri

**[5]** SAKARIDIS, Christos, DAI, Dengxin ve VAN GOOL, Luc. *ACDC: The Adverse
Conditions Dataset with Correspondences for Semantic Driving Scene Understanding.*
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Ekim
2021, s. 10765–10775. https://arxiv.org/abs/2104.13395
✔ doğrulandı · 4006 görüntü, sis/gece/yağmur/kar arasında eşit dağılmış, piksel düzeyinde
anotasyon + normal koşul eşleniği + belirsiz bölge maskesi

**[6]** LIS, Krzysztof, NAKKA, Krishna, FUA, Pascal ve SALZMANN, Mathieu. *Detecting the
Unexpected via Image Resynthesis.* Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), 2019.
https://openaccess.thecvf.com/content_ICCV_2019/html/Lis_Detecting_the_Unexpected_via_Image_Resynthesis_ICCV_2019_paper.html
✔ doğrulandı · **RoadAnomaly veri setinin kaynağı** — bu tezde açık küme değerlendirmesi
için kullanıldı

**[7]** CORDTS, Marius, OMRAN, Mohamed, RAMOS, Sebastian, REHFELD, Timo, ENZWEILER,
Markus, BENENSON, Rodrigo, FRANKE, Uwe, ROTH, Stefan ve SCHIELE, Bernt. *The Cityscapes
Dataset for Semantic Urban Scene Understanding.* Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2016.
https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html
✔ doğrulandı · 50 şehir, 5000 ince anotasyonlu görüntü, 1024×2048, **19 sınıf** — bu
tezdeki bütün ontolojinin temeli

**[8]** VARMA, Girish, SUBRAMANIAN, Anbumani, NAMBOODIRI, Anoop, CHANDRAKER, Manmohan ve
JAWAHAR, C. V. *IDD: A Dataset for Exploring Problems of Autonomous Navigation in
Unconstrained Environments.* IEEE Winter Conference on Applications of Computer Vision
(WACV), 2019. arXiv:1811.10200. https://arxiv.org/abs/1811.10200
✔ doğrulandı · 182 sürüş dizisinden 10.004 görüntü, 34 sınıf, **dört düzeyli etiket
hiyerarşisi**

> **Tezdeki rolü:** IDD'nin varlık nedeni, Cityscapes'in dayandığı varsayımların —
> şeritli yol, az sayıda iyi tanımlı nesne sınıfı, trafik kurallarına uyum —
> yapılandırılmamış yollarda geçerli olmamasıdır. Bizim çok-domainli eğitim kararımızın
> gerekçesi tam olarak budur; makale ayrıca "geleneksel yol dışındaki sürülebilir alan"
> gibi yeni sınıflar tanımlıyor, ki bu da sürülebilir alan ölçümümüzle doğrudan ilgili.

## C · Belirsizlik, kalibrasyon ve açık küme

**[9]** LIU, Weitang, WANG, Xiaoyun, OWENS, John ve LI, Yixuan. *Energy-based
Out-of-distribution Detection.* Advances in Neural Information Processing Systems 33
(NeurIPS), 2020.
https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
✔ doğrulandı · **energy skorunun kaynağı** — makale energy'nin softmax skorlarından daha
iyi ayırt ettiğini savunuyor; bizim ölçümümüz bunu bağımsız olarak doğruladı
(AUROC 0,667 vs 0,569)

**[10]** GUO, Chuan, PLEISS, Geoff, SUN, Yu ve WEINBERGER, Kilian Q. *On Calibration of
Modern Neural Networks.* Proceedings of the 34th International Conference on Machine
Learning (ICML), PMLR 70, 2017. https://proceedings.mlr.press/v70/guo17a.html
✔ doğrulandı · **aşırı-güven olgusunun ve sıcaklık ölçeklemenin kaynağı** — modern
ağların kötü kalibre olduğu bulgusu; bizim ACDC gece ölçümümüz (güven %71,9, doğruluk
%45,2) bunun uç bir örneği

**[11]** CHAN, Robin, LIS, Krzysztof ve ark. *SegmentMeIfYouCan: A Benchmark for Anomaly
Segmentation.* Proceedings of the NeurIPS Track on Datasets and Benchmarks, 2021.
https://arxiv.org/abs/2104.14812
✔ doğrulandı · iki görev: anomali nesne segmentasyonu ve **yol engeli segmentasyonu**;
100 görüntülük piksel etiketli değerlendirme seti

> Bu kaynak bizim **kayıp yük körlüğü** bulgumuzun (AUROC 0,4805) literatürdeki
> karşılığını veriyor: benchmark, yol üzerindeki engelleri ayrı bir görev olarak
> tanımlıyor çünkü genel anomali yöntemleri orada zayıf kalıyor.

## D · Uç cihaz dağıtımı

**[12]** SHESHADRI, Suhas Hariharapura, KARUMBUNATHAN, Leela Subramaniam ve FRANKLIN,
Dustin. *NVIDIA Jetson Orin Nano Developer Kit Gets a "Super" Boost.* NVIDIA Technical
Blog, 17 Aralık 2024.
https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/
✔ doğrulandı · 67 seyrek / 33 yoğun INT8 TOPS, 8 GB 128-bit LPDDR5, 102 GB/s bant
genişliği, 1024 CUDA + 32 Tensor çekirdeği @ 1020 MHz, 6 çekirdekli Arm Cortex-A78AE @
1,7 GHz, **7 W | 15 W | 25 W** güç modları

> **Neden bu kaynak:** ölçümlerimizin tamamı `--power-profile 25W` ile alındı ve 25 W
> modu tam olarak bu duyuruyla gelen "super" moddur (GPU 635 → 1020 MHz, bellek 68 → 102
> GB/s). Cihazın hangi yapılandırmada ölçüldüğü, sayıların kendisi kadar tezin parçasıdır.

**[13]** NVIDIA. *Accuracy Considerations.* NVIDIA TensorRT Documentation, Inference
Library. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html
✔ doğrulandı · FP16'nın 5 bit üs + 10 bit mantis yapısının hız/bellek kazancı sağladığını
ama "azaltılmış hassasiyet gösterebileceğini" ve indirgenmiş hassasiyetin "anlamlı
doğruluk kaybına yol açabileceğini" belirtiyor

> **Dikkat — bu kaynağa yükleyebileceğimizden fazlasını yüklemeyelim:** doküman FP16
> sonrası bağımsız doğrulamayı *zorunlu kılan* bir ifade içermiyor; doğruluk kaybının
> mümkün olduğunu söylüyor. Bizim `numerical_equivalence_pending: true` bayrağımızın
> gerekçesi budur: motor FP16'da kuruldu, dağıtım doğruluğu henüz cihazda ölçülmedi, ve
> "olabilir"i "olmadı"ya çevirecek tek şey ölçümün kendisidir (bkz. yapılacaklar D1).

**[14]** KRIŠLAURKS, Rihards, TISCENKO, Deniss, MEDVEDEVS, Vladislavs, ORMANIS, Juris ve
JUDVAITIS, Janis. *Ambient Temperature Impact on the Thermal Behavior and Power
Consumption of the NVIDIA Jetson AGX Orin in an Outdoor Enclosure.* Electronics, 2026,
cilt 15, sayı 11, makale 2467. https://doi.org/10.3390/electronics15112467
✔ doğrulandı · −20 °C ile +40 °C arası ortam sıcaklığında AGX Orin karakterizasyonu;
+40 °C'de stres testi **GPU +95,6 °C / CPU +99,0 °C**'de termal kısıtlamayı tetikliyor;
YOLOv8s çıkarım yükü 19,1 W ortalama güçte 108,8 FPS

> **Tezdeki rolü — iki kayıt, bir uyarı.** Birincisi: kısıtlamanın gerçekleştiği eşik
> *sıcaklık* değerlerinden okunuyor, bir uyarı bayrağından değil. Bizim
> `benchmark.py`'deki `throttling_warning_detected` kontrolü stok `tegrastats`
> çıktısında hiç geçmeyen kelimeleri aradığı için pratikte boş bir kontroldür; termal
> kanıtı biz de sıcaklık serisinden okuyoruz. İkincisi: ölçülen cihaz **AGX Orin**'dir,
> bizimki **Orin Nano Super** — farklı güç zarfı, farklı soğutma. Sayılar bizim cihazımıza
> aktarılamaz; aktarılabilen şey yöntemdir.

## E · Araçlar ve yazılım (akademik kaynak değil, araç atfı)

Materyal ve yöntem bölümünde ayrı listelenir:

- **mmsegmentation** (OpenMMLab) — referans checkpoint'lerin kaynağı, Apache-2.0
- **cityscapesscripts** — etiket eşleme tablosunun yetkili kaynağı
- **ONNX / ONNX Runtime**, **TensorRT 10.3.0**, **JetPack 6 (L4T 36)**
- **PyTorch 2.1.1**, **NumPy**, **Pillow**

---

## Doğrulama durumu

| durum | sayı |
|---|---|
| ✔ tam künyesiyle doğrulandı | **14** |
| ⏳ doğrulanacak | 0 |
| hedef | ~45–55 |

**Kural:** bu listeye yalnızca açılıp doğrulanmış kaynak girer. `researchs/` klasöründeki
912 kayıt aday havuzudur, kaynakça değildir.

**İkinci kural:** bir kaynağın var olduğunu doğrulamak, ona atfedilen iddianın doğru
olduğunu göstermez. Tezde bir kaynağa dayanarak cümle kuruluyorsa, o cümleyi yazan kişi
kaynağın ilgili bölümünü okumuş olmalıdır.
