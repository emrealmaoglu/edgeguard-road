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

**[3]** *(DDRNet — doğrulanacak)* Deep Dual-resolution Networks for Real-time and
Accurate Semantic Segmentation of Road Scenes.

**[4]** *(BiSeNetV2 — doğrulanacak)* Bilateral Network with Guided Aggregation for
Real-time Semantic Segmentation.

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

**[7]** *(Cityscapes — doğrulanacak)* CORDTS, Marius ve ark. The Cityscapes Dataset for
Semantic Urban Scene Understanding. CVPR, 2016.

**[8]** *(IDD — doğrulanacak)* India Driving Dataset.

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

**[11]** *(SegmentMeIfYouCan — doğrulanacak)* CHAN, Robin ve ark. A Benchmark for Anomaly
Segmentation. NeurIPS Datasets and Benchmarks, 2021.

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
| ✔ tam künyesiyle doğrulandı | **6** |
| ⏳ doğrulanacak | 8 |
| hedef | ~45–55 |

**Kural:** bu listeye yalnızca açılıp doğrulanmış kaynak girer. `researchs/` klasöründeki
912 kayıt aday havuzudur, kaynakça değildir.

**İkinci kural:** bir kaynağın var olduğunu doğrulamak, ona atfedilen iddianın doğru
olduğunu göstermez. Tezde bir kaynağa dayanarak cümle kuruluyorsa, o cümleyi yazan kişi
kaynağın ilgili bölümünü okumuş olmalıdır.
