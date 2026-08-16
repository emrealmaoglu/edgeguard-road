# Raporu şablona geçirme kılavuzu

Hedef net: **şablona uysun, intihal olmasın, göze batan düzensizlik olmasın.** Bu dosya
onu iki parçaya ayırıyor — biçim kuralları (puanı doğrudan etkiliyor) ve hangi bölüme
hangi içeriğin gireceği.

Şablondan çıkarılan gerçek değerler aşağıda; tahmin yok, `document.xml`'den okundu.

---

## 1 · Biçim — dokunma, olduğu gibi kalsın

Şablonun kendisi zaten doğru ayarlı. **En büyük risk, kendi metnini yapıştırırken bu
ayarların bozulması.** Her yapıştırmada Word'de **"Yalnızca metni koru"** (Ctrl+Shift+V)
kullan, sonra şablonun stilini uygula. Doğrudan yapıştırma fontu ve aralığı taşır.

| ayar | değer |
|---|---|
| sayfa | A4 (11909 × 16834 twip) |
| kenar boşlukları | üst/sağ/alt **2,5 cm**, **sol 3,5 cm** (ciltleme payı) |
| font | **Times New Roman** |
| üstbilgi/altbilgi | 1,25 cm |

> Sol kenarın diğerlerinden geniş olması hata değil, ciltleme payıdır. Eşitlemeye çalışma.

### Numaralandırma ve başlıklar

Şablon üç düzey kullanıyor ve **büyük/küçük harf düzeyi belirtiyor**:

- `ALT BAŞLIK 1` → tamamı büyük harf
- `Alt Başlık 2` → her kelime büyük harfle başlar
- `Alt Başlık 3` → aynı

Bölüm numaraları otomatik. Kendin `2.1`, `2.2` yazma — şablonun stilini kullan, numara
kendiliğinden gelir ve araya bölüm eklersen hepsi güncellenir.

### Şekil ve çizelge — en sık yapılan hata burada

| öğe | başlık nerede | numara | örnek |
|---|---|---|---|
| **Şekil** | **altında** | bölüme göre | `Şekil 2.1. Sistem mimarisi.` |
| **Çizelge** | **üstünde** | bölüme göre | `Çizelge 2.1. Model karşılaştırması.` |
| Harita | altında | bölüme göre | `Harita 2.1. …` |
| Denklem | sağda, parantezli | bölüme göre | `(2.1)` |

Üç kural daha, şablonun kendi metninden:

1. **Şekle atıf, şekilden önce gelir.** Yani önce "Şekil 2.1 sistem mimarisini
   göstermektedir." cümlesi, sonra şekil.
2. Metin ile şekil arasında **1,5 satır aralık**.
3. Tek başlıkta birden fazla şekil varsa `a)`, `b)`, `c)` diye isimlendirilir ve başlıkta
   hepsi açıklanır: *"Şekil 2.4. … a) İstanbul b) Eminönü c) Galata köprüsü."*

Başlıkların sonunda **nokta var** — `Şekil 2.1. Sistem mimarisi.` Şablon böyle, sen de öyle
yap.

### Çapraz başvuru

Şekil/çizelge/denklem numaralarını metne **elle yazma.** Word'de
**Ekle → Çapraz başvuru** ile ekle. Sebep pratik: sonradan araya bir şekil eklersen
numaralar kendiliğinden kayar, elle yazdıysan hepsi yanlış kalır ve bu **"göze batan
düzensizlik"** tanımına birebir uyar.

---

## 2 · Beyanlar — bunu atlama

Şablonda iki zorunlu beyan var ve rubrikte **15 puanlık "Etik ve Mesleki Sorumluluk"**
bloğuna bağlılar.

### 2.1 · İntihal beyanı

Metin şablonda hazır, imzalanıp tarihlenecek. Rubrikteki karşılığı: *"☐ İntihal beyanı
mevcut"*.

### 2.2 · Üretken yapay zekâ kullanım beyanı — dikkat

Şablon şunu istiyor: *"Bu tez çalışmasını hazırlarken … üretken yapay zekâ
programlarından … yararlandım/yararlanmadım"* ve arkasından şu cümle geliyor:
*"beyana aykırı bir durumun saptanması durumunda, ortaya çıkacak …"*.

**Bunu dürüst doldur.** Bu projede yapay zekâ yoğun biçimde kullanıldı ve bunu gizlemenin
hiçbir faydası yok — üstelik gizlemek beyanı ihlal eder.

İyi haber: elinde **beyanı destekleyen kanıt var.** `docs/AI_USAGE_LOG.md` her materyal
değişikliği tarih, dosya, doğrulama ve gerekçesiyle kaydeden **ekle-sadece** bir kayıt.
Böyle bir log tutmak, "kullandım" demeyi zayıflık olmaktan çıkarıp **süreç disiplini**
kanıtına çevirir.

Beyanda ne yazacağın sana ait, ama önerim: hangi işlerde kullanıldığını (kod, ölçüm
altyapısı, doküman taslakları) ve neyin sana ait olduğunu (proje kararları, cihaz
çalıştırma, sonuçların yorumu) açıkça ayır. `AI_USAGE_LOG.md`'yi **EK olarak** ekleyebilirsin
— rubrikteki "raporlarını düzenli hazırladı mı" (10 puan) maddesine de doğrudan hizmet
eder.

---

## 3 · Hangi bölüme ne girecek

Elindeki dokümanların hepsi hazır; iş onları şablonun bölümlerine dağıtmak.

| Şablon bölümü | Kaynak | Not |
|---|---|---|
| **ÖZET** | `THESIS_SECTIONS.md` §C7 | Tek paragraf olmalı — şablon böyle diyor. Bendeki metin dört bulguyu ayrı cümlelerle veriyor, tek paragrafa birleştir. |
| **ABSTRACT** | `THESIS_SECTIONS.md` §C7 | Aynısının İngilizcesi, hazır. |
| Anahtar sözcükler | §C7 sonu | Şablon **üç** örnek veriyor; 5-8 arası makul. |
| **1. GİRİŞ** *(zorunlu)* | `THESIS_OUTLINE.md` §1 | Şablon üç şey istiyor: literatür özeti, tezin amacı, literatüre katkı. |
| Literatür taraması | `LITERATURE_PLAN.md` §2 + `D5_prisma_flow.png` | **PRISMA şeması Şekil 1.1 olacak.** Rubrikte 5 puan. |
| **2. MATERYAL VE YÖNTEM** | `THESIS_OUTLINE.md` §2 | Veri setleri, modeller, ölçüm protokolü. |
| Blok şeması | `D1_system_architecture.png` | Rubrikte 5 puan, **zorunlu gibi davran.** |
| **3. UYGULAMA VE SORUNLAR** | `THESIS_OUTLINE.md` §3 | Rubrik: *"uyumsuzluklar ve aksaklıklar … çözüm yöntemleri tartışılmış mı"* — 5 puan. Bu bölüm senin en güçlü tarafın, karşılaşılan gerçek hatalar `AI_USAGE_LOG.md`'de tarihli duruyor. |
| **4. OPTİMİZASYON** | `THESIS_EVIDENCE.md` §8 | Kare bütçesi 146 → 94 ms. |
| **5. BULGULAR VE TARTIŞMA** | `THESIS_EVIDENCE.md` §1–§9 | Ana bölüm. Aşağıdaki çizelge listesine bak. |
| **6. SONUÇLAR** | `THESIS_OUTLINE.md` §6 | Üç sonuç cümlesi + sınırlar. |
| Standartlar | `THESIS_SECTIONS.md` §C4 | Rubrikte 5 puan. |
| Girişimcilik | `THESIS_SECTIONS.md` §C5 | 4 puan. |
| Sürdürülebilirlik | `THESIS_SECTIONS.md` §C6 | 4 puan. |
| **KAYNAKLAR** | `BIBLIOGRAPHY.md` | ISO 690, 19 kaynak, hepsi doğrulanmış. |
| **EKLER** | — | EK 1: ölçüm kayıtları, EK 2: AI kullanım logu, EK 3: yeniden üretim talimatı. |

### Şekil ve çizelge listesi

Elindeki hazır görseller (`.local/presentation/figures/`, hem `.png` hem `.pdf`):

**Şemalar** — `D1_system_architecture` · `D2_measurement_protocol` ·
`D3_three_axis_framework` · `D4_frame_budget_flow` · `D5_prisma_flow`

**Sonuç figürleri** — `01_published_vs_measured_miou` · `02_per_class_iou` ·
`03_reliability_diagrams` · `04_acdc_conditions` · `05_open_set` · `06_pareto` ·
`07_frame_budget` · `08_synthetic_shift` · `09_overconfidence`

Rapora **PDF sürümünü** koy, PNG'yi değil — vektörel olduğu için baskıda ve yakınlaştırmada
bozulmaz.

Çizelgeler `THESIS_EVIDENCE.md`'deki tablolardan doğrudan alınabilir: model karşılaştırması,
sınıf bazlı IoU, açık küme skorları, ACDC koşulları, Jetson performansı, sürülebilir alan,
kalibrasyon, FP16 sadakati.

---

## 4 · İntihal — gerçek risk nerede

Ölçüm sonuçların, kodun ve bulguların **senin**; oradan intihal çıkmaz. Risk iki yerde:

1. **Literatür bölümü.** Bir makaleyi özetlerken cümle yapısını taşımak en sık yapılan
   hata. Kaynağı okuduktan sonra kapat, kendi cümlenle yaz, sonra doğruluğunu kontrol et.
   `BIBLIOGRAPHY.md`'deki her kayıt zaten künyesiyle doğrulandı, atıf tarafında sorun yok.

2. **Tanım cümleleri.** "mIoU şöyle hesaplanır", "TensorRT şudur" gibi cümleler internetten
   birebir alınmaya çok müsait. Bunları formülle ver, düz metinle değil.

Şablonun istediği **ISO 690** biçimi `BIBLIOGRAPHY.md`'de zaten uygulanmış — kopyalayıp
yapıştırabilirsin.

---

## 5 · Teslimden önce son kontrol

- [ ] İçindekiler güncellendi mi? (Word'de tabloya sağ tık → **Alanı güncelleştir**)
- [ ] Şekil/çizelge numaraları çapraz başvuruyla mı, elle mi yazıldı?
- [ ] Şekil başlıkları **altta**, çizelge başlıkları **üstte** mi?
- [ ] Her şekle metin içinde, şekilden **önce** atıf var mı?
- [ ] ÖZET tek paragraf mı?
- [ ] İki beyan da imzalı ve tarihli mi?
- [ ] AI beyanı **dürüst** mü?
- [ ] Kaynakçadaki her kaynağa metinde atıf var mı, metindeki her atıf kaynakçada mı?
- [ ] Sayfa numaraları kesintisiz mi? (ön kısım romen, gövde arap rakamı)
- [ ] Baştan sona bir kez PDF'e çevirip **göz gezdirdin mi**? Kayan tablo, taşan şekil,
      yarım kalan başlık en çok burada yakalanır.
