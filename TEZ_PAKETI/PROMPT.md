# Başka bir yapay zekâya verilecek prompt

> Aşağıdaki metnin tamamını kopyalayıp, bu klasördeki dosyalarla birlikte yapay zekâya
> ver. `---` çizgileri arasındaki kısım prompt'un kendisidir.

---

Bir lisans bitirme tezi raporunu, verilen Word şablonuna yerleştirmeni istiyorum. Bütün
içerik hazır; senin işin **yazmak değil, yerleştirmek ve biçimlendirmek.**

## Sana verilenler

- `sablon/BM498_tez_sablonu.docx` — boş şablon (Düzce Üniversitesi BM498)
- `sablon/ONISLENMIS_taslak.docx` — metnin şablona önceden yerleştirilmiş hâli;
  isteğe bağlı başlangıç noktası
- `bolumler/*.md` — raporun dokuz bölümü, sırasıyla numaralı
- `sekiller/*.pdf` ve `*.png` — 16 şekil, **dosya adları tez şekil numaralarıdır**
- `yonerge/BICIM_KURALLARI.md` — şablonun biçim kuralları

## Yapman gereken

`bolumler/` içindeki metni şablona yerleştir, şekilleri ve tabloları doğru yerlere koy,
biçim kurallarına uy ve tek bir `.docx` üret.

## MUTLAK KURALLAR — bunları ihlal etme

**1. Hiçbir sayıyı değiştirme, yuvarlama veya "düzeltme".**
Metindeki her sayı gerçek bir ölçümden gelir ve ölçüm kayıtlarıyla eşleştiği otomatik
olarak doğrulanmıştır. `0,6847` gördüğünde `0,68` yapma. `+0,0000` gördüğünde "sıfır" diye
yazma — o bir ölçüm sonucudur ve güven aralığıyla birlikte anlam taşır.

**2. Hiçbir bulguyu ekleme, çıkarma veya güçlendirme.**
Metinde "ayırt edilemez", "ölçülmedi", "geri çekildi" gibi ifadeler bilinçlidir. Bunları
"başarılı", "en iyi", "kanıtlandı" gibi ifadelere çevirme. Rapor kasıtlı olarak
**"en doğru model" ifadesini kullanmaz**; bu ifadeyi metne sokma.

**3. Eksik bölümleri kendin doldurma.**
Metinde doldurulmamış bir yer varsa (sayfa sayısı, öğrenci bilgileri, teşekkür) orayı
`[DOLDURULACAK: ...]` olarak bırak. Uydurma.

**4. Ön kısma dokunma.**
Kapak, değerlendirme tutanağı, BEYAN ve ÜRETKEN YAPAY ZEKA KULLANIM BEYANI sayfaları
şablonda hazırdır ve **öğrenci tarafından imzalanacaktır**. İçeriklerini değiştirme,
silme, yeniden yazma. Yalnızca kişisel bilgi alanlarını `[DOLDURULACAK]` olarak işaretle.

**5. Kaynakçaya kaynak ekleme.**
19 kaynağın tamamı birincil kaynağından doğrulanmıştır. Listeye ekleme yapma, sıralamayı
bozma. Metin içi atıflar `[1]`, `[2]` biçiminde **numaralıdır** — şablonun kuralı budur.

## Biçim kuralları

Ayrıntısı `yonerge/BICIM_KURALLARI.md` dosyasındadır. Özeti:

| Kural | Değer |
|---|---|
| Sayfa | A4, kenar boşlukları üst/sağ/alt 2,5 cm, **sol 3,5 cm** (ciltleme payı) |
| Font | Times New Roman |
| **Şekil başlığı** | Şeklin **ALTINDA**, `Şekil 2.1. Sistem mimarisi.` (sonunda nokta) |
| **Çizelge başlığı** | Tablonun **ÜSTÜNDE**, `Çizelge 2.1. Model karşılaştırması.` |
| Şekle atıf | Metinde **şekilden ÖNCE** geçer |
| Numaralandırma | Bölüme göre (`2.1`, `2.2`, `5.1`…), Word'ün otomatik alanlarıyla |
| Başlık düzeyleri | `ALT BAŞLIK 1` büyük harf · `Alt Başlık 2` · `Alt Başlık 3` |

Şablonun stilleri doğru ayarlıdır. Metni yapıştırırken **"Yalnızca metni koru"** kullan,
sonra şablonun stilini uygula — doğrudan yapıştırma fontu ve satır aralığını bozar.

## Şekillerin yerleşimi

Dosya adları şekil numaralarıdır. `bolumler/` içindeki metinde
`**[Şekil 2.1: ...]**` biçiminde yer tutucular vardır; her birini karşılık gelen dosyayla
değiştir ve altına şekil başlığını koy.

| Dosya | Bölüm | Başlık |
|---|---|---|
| `Sekil_1.1_literatur_tarama_akisi.pdf` | 1.3 | Literatür tarama akışı (PRISMA). |
| `Sekil_2.1_sistem_mimarisi.pdf` | 2.3 | Görüntüden risk sıralamasına sinyal yolu. |
| `Sekil_2.2_olcum_protokolu.pdf` | 2.4 | Ölçüm protokolü: hangi veri hangi soruyu nerede yanıtlıyor. |
| `Sekil_2.3_uc_eksenli_cerceve.pdf` | 2.4 | Üç eksenli değerlendirme çerçevesi. |
| `Sekil_4.1_kare_butcesi_akisi.pdf` | 4.1 | Kare bütçesinin aşamalara dağılımı ve optimizasyonun etkisi. |
| `Sekil_4.2_kare_butcesi.pdf` | 4.5 | Optimizasyon öncesi ve sonrası kare bütçesi. |
| `Sekil_5.1_yayin_vs_olculen_miou.pdf` | 5.1 | Yayımlanmış ve ölçülen mIoU karşılaştırması. |
| `Sekil_5.2_sinif_bazli_iou.pdf` | 5.2 | Dağıtım çözünürlüğünde sınıf bazlı IoU. |
| `Sekil_5.3_guvenilirlik_diyagramlari.pdf` | 5.4 | Güvenilirlik diyagramları. |
| `Sekil_5.4_acdc_kosullari.pdf` | 5.8 | Gerçek olumsuz koşullarda doğruluk ve kalibrasyon. |
| `Sekil_5.5_open_set.pdf` | 5.6 | Open-set skorları ve tehlike türüne göre kırılım. |
| `Sekil_5.6_pareto.pdf` | 5.7 | Doğruluk ↔ enerji ve open-set ↔ gecikme ödünleşimleri. |
| `Sekil_5.7_asiri_guven.pdf` | 5.4 | Kesin yanlış piksellerde güven düzeyi. |
| `Sekil_5.8_sentetik_bozulma.pdf` | 5.8 | Sentetik bozulmaya belirsizlik tepkisi. |
| `Sekil_5.9_bes_mimari_ayni_kare.png` | 5.9 | Aynı karede beş mimarinin çıktısı. |
| `Sekil_5.10_kosullar_pidnet_s.png` | 5.9 | PIDNet-S çıktısı: temiz, sis ve gece koşulları. |

**Not:** Şekiller PDF biçimindedir çünkü vektöreldir — baskıda ve yakınlaştırmada
bozulmaz. Word'e `Ekle → Resim` ile eklenebilirler. İkisi PNG'dir (nitel görseller).

## Tabloların yerleşimi

Bölüm dosyalarındaki Markdown tabloları Word tablosuna çevir. Her tablonun **üstünde**
zaten `Çizelge X.Y. Başlık.` satırı vardır; onu tablonun başlığı olarak kullan.

Tabloları şablondaki mevcut tablo stiliyle biçimlendir — yanındaki tablolardan farklı
görünmemeleri önemlidir. Geniş tabloları sayfaya sığdır; gerekiyorsa font bir punto
küçültülebilir, ancak tablo sayfa dışına taşmamalıdır.

## Bölüm sırası

```
00_ozet_abstract.md   → ÖZET ve ABSTRACT sayfaları (şablonda hazır yerleri var)
01_giris.md           → 1. GİRİŞ
02_materyal_ve_yontem.md → 2. MATERYAL VE YÖNTEM
03_uygulama_ve_sorunlar.md → 3. UYGULAMA VE KARŞILAŞILAN SORUNLAR
04_optimizasyon.md    → 4. OPTİMİZASYON
05_bulgular_ve_tartisma.md → 5. BULGULAR VE TARTIŞMA
06_sonuclar_ve_oneriler.md → 6. SONUÇLAR VE ÖNERİLER
07_kaynaklar.md       → KAYNAKLAR
08_ekler.md           → EKLER
```

Şablonun sonunda bir **ÖZGEÇMİŞ** bölümü vardır; içeriğini `[DOLDURULACAK: öğrenci
özgeçmişi]` olarak bırak.

## Markdown'daki işaretler ne anlama geliyor

- `> Şablon notu:` ile başlayan alıntı blokları **bana yazılmış açıklamalardır**, rapora
  girmez — bunları at.
- `**[Şekil X.Y: dosya]**` → o şeklin geleceği yer.
- `**kalın**` → Word'de kalın yap.
- `` `kod` `` → düz metin olarak yaz, kod biçimlendirmesi gerekmez.
- `---` → bölüm ayracı, rapora girmez.

## Bitirdiğinde

Tek bir `.docx` üret ve şunları yaptığını doğrula:

1. İçindekiler güncellendi (Word'de alanı güncelleştir)
2. Şekil başlıkları **altta**, çizelge başlıkları **üstte**
3. Her şekle metin içinde şekilden **önce** atıf var
4. Sayfa numaraları kesintisiz (ön kısım romen, gövde arap rakamı)
5. Hiçbir tablo sayfa dışına taşmıyor
6. Hiçbir başlık sayfa sonunda tek başına kalmıyor
7. Doldurulmamış yerler `[DOLDURULACAK: ...]` olarak işaretli

Son olarak, **değiştirdiğin veya emin olamadığın her şeyi bir liste hâlinde bildir.**
Özellikle bir sayıyı veya ifadeyi değiştirmek zorunda kaldıysan bunu mutlaka söyle.

---
