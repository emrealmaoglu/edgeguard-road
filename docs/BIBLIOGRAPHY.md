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

**[8]** *(IDD — doğrulanacak)* India Driving Dataset. Kendi çok-domainli eğitimimizde
Cityscapes ile birlikte kullanıldı.

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

**[12]** *(NVIDIA Jetson Orin Nano Super — resmî ürün/teknik dokümanı)*

**[13]** *(NVIDIA TensorRT dokümanı)*

**[14]** *(Jetson termal/güç davranışı — MDPI, doğrulanacak)*

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
| ✔ tam künyesiyle doğrulandı | **10** |
| ⏳ doğrulanacak | 4 |
| hedef | ~45–55 |

**Kural:** bu listeye yalnızca açılıp doğrulanmış kaynak girer. `researchs/` klasöründeki
912 kayıt aday havuzudur, kaynakça değildir.

**İkinci kural:** bir kaynağın var olduğunu doğrulamak, ona atfedilen iddianın doğru
olduğunu göstermez. Tezde bir kaynağa dayanarak cümle kuruluyorsa, o cümleyi yazan kişi
kaynağın ilgili bölümünü okumuş olmalıdır.
