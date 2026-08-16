# Tez raporu paketi — EdgeGuard-Road / BM498

Bu klasör, raporu üretmek için gereken **her şeyi** içerir. Başka bir yapay zekâya
devredilmek üzere hazırlanmıştır.

## Nasıl kullanılır

1. **`PROMPT.md`** dosyasını aç, `---` çizgileri arasındaki metni kopyala.
2. Bu klasörün tamamını (dosyalarıyla birlikte) yapay zekâya ver.
3. Kopyaladığın prompt'u yapıştır.

Hepsi bu. Prompt kendi kendine yeterlidir; yapay zekânın bu projeyi tanımasına gerek yok.

## Klasörde ne var

```
TEZ_PAKETI/
├── PROMPT.md                     ← yapay zekâya verilecek metin
├── OKU_BENI.md                   ← bu dosya
├── sablon/
│   ├── BM498_tez_sablonu.docx    ← boş şablon (Düzce Üniversitesi)
│   └── ONISLENMIS_taslak.docx    ← metin şablona önceden yerleştirilmiş hâli
├── bolumler/                     ← raporun dokuz bölümü (Markdown)
├── sekiller/                     ← 16 şekil, dosya adları tez numaralarıdır
└── yonerge/
    └── BICIM_KURALLARI.md        ← şablonun biçim kuralları
```

### İki şablon dosyası neden var

`ONISLENMIS_taslak.docx`, metnin şablona zaten yerleştirilmiş hâlidir; şekil ve tablolar
`[BURAYA ŞEKİL: ...]` biçiminde yer tutucu olarak durur. Yapay zekâ isterse buradan devam
edebilir; isterse boş şablondan başlayabilir. İkisi de kabul.

### Şekiller

Dosya adları doğrudan tez numarasıdır: `Sekil_5.1_yayin_vs_olculen_miou.pdf` → raporda
**Şekil 5.1**. Böylece hangisinin nereye gideceği konusunda belirsizlik kalmaz.

PDF olmalarının nedeni vektörel olmalarıdır — baskıda ve yakınlaştırmada bozulmazlar.
İkisi PNG'dir (nitel görseller, zaten raster).

---

## SENİN YAPMAN GEREKENLER

Yapay zekâ bunları **yapamaz**, sen yapacaksın:

### 1. İki beyanı imzala

Şablonda iki beyan sayfası vardır ve rubrikte 15 puanlık "Etik ve Mesleki Sorumluluk"
bloğuna bağlıdırlar:

- **BEYAN** (intihal beyanı)
- **ÜRETKEN YAPAY ZEKA KULLANIM BEYANI**

İkincisi hakkında açık konuşmak gerekir: **bu projede yapay zekâ yoğun biçimde
kullanılmıştır.** Beyanda bunu dürüstçe belirt. Şablonun kendisi bu beyanı zorunlu
tutmakta ve arkasına "beyana aykırı durum saptanması hâlinde…" maddesini koymaktadır —
yani gizlemek beyanı ihlal eder.

Elinde beyanı destekleyen kanıt vardır: `docs/AI_USAGE_LOG.md` her materyal değişikliği
tarihi, dosyası, doğrulaması ve gerekçesiyle tutan **ekle-sadece** bir kayıttır. Böyle bir
log tutmak, "kullandım" demeyi zayıflık olmaktan çıkarıp süreç disiplini kanıtına
dönüştürür. EK olarak eklemeni öneririm.

Beyanda önerdiğim ayrım: **yapay zekânın yaptığı** (kod, ölçüm altyapısı, doküman
taslakları) ile **senin yaptığın** (proje kararları, cihazın çalıştırılması, sonuçların
yorumlanması) açıkça ayrılsın.

### 2. Kişisel bilgileri doldur

- Kapak: ad, öğrenci numarası, ders sorumlusu, dönem
- ÖZGEÇMİŞ sayfası
- TEŞEKKÜR (serbest)
- ÖZET sayfasındaki **sayfa sayısı** (rapor bitince belli olur)

### 3. Son kontrolü kendin yap

Yapay zekânın ürettiği dosyayı Word'de aç ve **baştan sona bir kez göz gezdir.** Kayan
tablo, taşan şekil, sayfa sonunda tek kalmış başlık en çok burada yakalanır. Bunu yapay
zekâya güvenerek atlamak, "göze batan düzensizlik" riskini alır.

---

## RAPORDA DEĞİŞTİRİLMEMESİ GEREKENLER

Bu rapor, ölçülmüş sonuçlar üzerine kuruludur ve bazı ifadeleri bilinçlidir. Yapay zekâya
verdiğin prompt bunları koruması için uyarıyor, ama sen de bil:

- **Sayılar ölçümdür.** Yuvarlanmamalı, "düzeltilmemeli". Metindeki dört ondalıklı her
  sayının bir ölçüm kaydında karşılığı olduğu otomatik olarak doğrulanmıştır.
- **"En doğru model" ifadesi kasıtlı olarak kullanılmaz.** İlk iki model istatistiksel
  olarak ayırt edilemediği için. Bu, raporun bulgularından biridir.
- **Geri çekilen iki sonuç raporda durmaktadır.** Bunlar silinmemeli — ölçüm yapan bir
  çalışmanın kendi hatasını düzeltmesi güçlü yanıdır, zayıf yanı değil.
- **Ölçülmeyen şeyler "ölçülmedi" diye yazılıdır.** Bunlar başarı gibi sunulmamalı.
